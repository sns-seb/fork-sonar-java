package org.sonar.java.checks.s125model;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * RoBERTa BPE encoder according to the RoBERTa tokenizer implementation from Hugging Face.
 * <p>
 * This class relies on a file (called the merge file) to know:
 * <ul>
 *     <li>the pairs of characters to merge</li>
 *     <li>the rank of these pairs</li>
 * </ul>
 * The path to this file must be provided in the constructor. See {@link #readBPERanks(Path)} for description of the
 * expected content of the merge file.
 * <p>
 */
public class RoBERTaBPEEncoder implements BPEEncoder {
    private static final BPEPair[] NO_BPE_PAIRS = new BPEPair[0];
    private final Map<BPEPair, RankedBPEPair> bpePairsRanks;

    public RoBERTaBPEEncoder(InputStream inputStream) throws IOException {
        this.bpePairsRanks = readBPERanks(inputStream);
    }

    /**
     * Reads the content of the merge file.
     * <p>
     * The content of the merge file is expected:
     * <ul>
     *     <li>to be UTF-8 encoded</li>
     *     <li>1st line starts with {@code #} and will be ignored</li>
     *     <li>all other lines are made of two series of characters (can be a single character) separated by a blank space</li>
     * </ul>
     * <p>
     * The merge file is expected to be one of the trained RoBERTa models available from Hugging Face.
     * <p>
     * Eg, <a href="https://huggingface.co/roberta-base/raw/main/merges.txt">this file</a> for the
     * "roberta-base" model, or <a href="https://huggingface.co/roberta-large/resolve/main/merges.txt">this file</a> for the
     * "roberta-large" model.
     * <p>
     */
    private static Map<BPEPair, RankedBPEPair> readBPERanks(InputStream inputStream) throws IOException {
        Map<BPEPair, RankedBPEPair> res = new HashMap<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream, UTF_8))) {
            String versionLine = reader.readLine();
            if (!versionLine.startsWith("#")) {
                throw new IllegalStateException("Expected merges.txt to start with a version line");
            }
            for (int rank = 0; ; rank++) {
                String line = reader.readLine();
                if (line == null)
                    break;
                String[] substring = line.trim().split(" ");
                BPEPair bpePair = BPEPair.of(substring[0], substring[1]);
                res.put(bpePair, new RankedBPEPair(bpePair, rank));
            }
        }
        return res;
    }

    /**
     * BPE encode the specified {@link CharSequence}.
     * <p>
     * See original source code in
     * <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L251-L291">InferencePipeline#bpe(self, token)</a>.
     * <pre>
     * def bpe(self, token):
     *     if token in self.cache:
     *         return self.cache[token]
     *     word = tuple(token)
     *     pairs = get_pairs(word)
     *
     *     if not pairs:
     *         return token
     *
     *     while True:
     *         bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
     *         if bigram not in self.bpe_ranks:
     *             break
     *         first, second = bigram
     *         new_word = []
     *         i = 0
     *         while i < len(word):
     *             try:
     *                 j = word.index(first, i)
     *             except ValueError:
     *                 new_word.extend(word[i:])
     *                 break
     *             else:
     *                 new_word.extend(word[i:j])
     *                 i = j
     *
     *             if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
     *                 new_word.append(first + second)
     *                 i += 2
     *             else:
     *                 new_word.append(word[i])
     *                 i += 1
     *         new_word = tuple(new_word)
     *         word = new_word
     *         if len(word) == 1:
     *             break
     *         else:
     *             pairs = get_pairs(word)
     *     word = " ".join(word)
     *     self.cache[token] = word
     *     return word
     * </pre>
     *
     * @return an array of the words which result of the BPE compression of {@code text}, effectively words will
     *         be the split of the characters from {@code text} between pairs of characters which do not exist according
     *         to pairs defined and ordered in {@link #bpePairsRanks}.
     */
    @Override
    public String[] bpeEncode(CharSequence text) {
        // at this stage, all words are made of a single character
        String[] words = toArrayOfString(text);
        BPEPair[] bpePairs = toPairs(words);

        if (bpePairs.length == 0) {
            return words;
        }

        while (true) {
            RankedBPEPair[] rankedBPEPairs = toRankedPairs(bpePairs);
            RankedBPEPair rankedBPEPair = Arrays.stream(rankedBPEPairs).min(RankedBPEPairComparator.INSTANCE).get();
            if (!rankedBPEPair.isRanked()) {
                break;
            }

            words = applyBPECompression(words, rankedBPEPair.bpePair);
            if (words.length == 1) {
                break;
            }
            bpePairs = toPairs(words);
        }


        return words;
    }

    /**
     * Converts the specified {@link CharSequence} into an array where each character is represented as a {@link String}
     */
    private static String[] toArrayOfString(CharSequence word) {
        return word.chars().mapToObj(c -> String.valueOf((char) c)).toArray(String[]::new);
    }

    /**
     * Converts the specified array of String into each pair of consecutive String in this array, as {@link BPEPair}
     * objects.
     */
    private static BPEPair[] toPairs(String[] word) {
        if (word.length < 2) {
            return NO_BPE_PAIRS;
        }
        BPEPair[] res = new BPEPair[word.length - 1];
        for (int i = 1; i < word.length; i++) {
            res[i - 1] = BPEPair.of(word[i - 1], word[i]);
        }
        return res;
    }

    /**
     * Builds an array of the same size as {@code pairs}, where each {@link BPEPair} in {@code pairs} is mapped to
     * a {@link BPEPair} with the rank of that {@link BPEPair} in {@link #bpePairsRanks} or an unranked {@link BPEPair}
     * (see {@link RankedBPEPair#unranked(BPEPair)}) if it doesn't exist in {@link #bpePairsRanks}.
     */
    private RankedBPEPair[] toRankedPairs(BPEPair[] pairs) {
        RankedBPEPair[] res = new RankedBPEPair[pairs.length];
        for (int i = 0; i < pairs.length; i++) {
            BPEPair bpePair = pairs[i];
            RankedBPEPair rankedBPEPair = this.bpePairsRanks.get(bpePair);
            res[i] = (rankedBPEPair == null ? RankedBPEPair.unranked(bpePair) : rankedBPEPair);
        }
        return res;
    }

    /**
     * Compress the words in the provided array by merging together the successive words which match the provided
     * {@link BPEPair]}.
     * <p>
     * The resulting array will have the size of {@code words} minus the number of time the {@code target} appears in
     * {@code words}.
     */
    private static String[] applyBPECompression(String[] words, BPEPair target) {
        int compressionCount = 0;
        String[] partialCompressed = new String[words.length];
        for (int i = 0; i < words.length; i++) {
            // last character can not belong to a pair, copy it and let the loop end
            if (i + 1 >= words.length) {
                partialCompressed[i] = words[i];
                continue;
            }

            // if the current char is the left part of the target BPEPair, copy the merge to the position
            // of the current char and skip the next char (effectively leaving a null in its position)
            if (target.equals(words[i], words[i + 1])) {
                partialCompressed[i] = target.merge();
                i++;
                compressionCount++;
            } else {
                partialCompressed[i] = words[i];
            }
        }

        // build the resulting array by copying all non-null values from partialCompressed
        String[] res = new String[words.length - compressionCount];
        int j = 0;
        for (int i = 0; i < words.length; i++) {
            if (partialCompressed[i] == null) {
                continue;
            }
            res[j++] = partialCompressed[i];
        }

        return res;
    }

    /**
     * Implements {@link #equals(Object)} and {@link #hashCode()} in order to be used as a key in a {@link Map}.
     */
    private static class BPEPair {
        private final String left;
        private final String right;
        private final String merge;

        private BPEPair(String left, String right) {
            this.left = left;
            this.right = right;
            this.merge = left + right;
        }

        public String merge() {
            return merge;
        }

        public static BPEPair of(String left, String right) {
            return new BPEPair(left, right);
        }

        public boolean equals(String left, String right) {
            return this.left.equals(left) && this.right.equals(right);
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            BPEPair bpePair = (BPEPair) o;
            return left.equals(bpePair.left) && right.equals(bpePair.right);
        }

        @Override
        public int hashCode() {
            return Objects.hash(left, right);
        }

        @Override
        public String toString() {
            return "{" + left + " " + right + "}";
        }
    }

    private static class RankedBPEPair {

        private static final int NOT_RANKED = -1;
        private final BPEPair bpePair;
        private final int rank;

        private RankedBPEPair(BPEPair bpePair, int rank) {
            this.bpePair = bpePair;
            this.rank = rank;
        }

        public boolean isRanked() {
            return rank >= 0;
        }

        public static RankedBPEPair of(BPEPair bpePair, int rank) {
            return new RankedBPEPair(bpePair, rank);
        }

        public static RankedBPEPair unranked(BPEPair bpePair) {
            return new RankedBPEPair(bpePair, NOT_RANKED);
        }

        @Override
        public String toString() {
            return '{' +
                    bpePair.left + ' ' + bpePair.right + ": " +
                    (isRanked() ? rank : "unranked") +
                    '}';
        }
    }

    /**
     * Intended to be used to find the {@link RankedBPEPair} with the lowest rank, leaving all unranked
     * {@link RankedBPEPair} (see {@link RankedBPEPair#isRanked()}) at the end and with their relative order unchanged.
     */
    private enum RankedBPEPairComparator implements Comparator<RankedBPEPair> { //NOSONAR
        INSTANCE;

        @Override
        public int compare(RankedBPEPair o1, RankedBPEPair o2) {
            if (o1.isRanked() && o2.isRanked()) {
                return o1.rank - o2.rank;
            }
            if (o1.isRanked()) {
                return -1;
            }
            if (o2.isRanked()) {
                return 1;
            }
            return 0;
        }
    }
}
