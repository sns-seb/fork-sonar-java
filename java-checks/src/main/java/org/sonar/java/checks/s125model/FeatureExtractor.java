package org.sonar.java.checks.s125model;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonParser;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Extract features from a comment as a array of tokens (see {@link #extractFrom(String[])}), assuming these tokens
 * are those produced by the RoBERTa tokenizer (see {@link RoBERTaTokenizer}).
 * <p>
 * Feature extraction will take into account a limited number of tokens, provided in the constructor.
 * <p>
 * The features extracted are:
 * <ul>
 *     <li>the count of occurrences of each word, in order, defined in a vocabulary file (see {@link #readVocabulary(String)})</li>
 *     <li>the count of semicolons in all tokens</li>
 *     <li>the frequency of the semicolons compared to the number of tokens</li>
 * </ul>
 */
public class FeatureExtractor {
    private final int maxTokensPerString;
    private final Map<String, Integer> indexByWord;

    protected FeatureExtractor(Map<String, Integer> indexByWord, int maxTokensPerString) {
        this.maxTokensPerString = maxTokensPerString;
        this.indexByWord = indexByWord;
    }

    public static FeatureExtractor create(InputStream vocabFilename, int maxTokensPerString) {
        return new FeatureExtractor(readVocabulary(vocabFilename), maxTokensPerString);
    }

    private static Map<String, Integer> readVocabulary(InputStream vocabFilename) {
        JsonArray jsonArray = JsonParser.parseReader(new InputStreamReader(vocabFilename, UTF_8)).getAsJsonArray();
        Iterator<JsonElement> it = jsonArray.iterator();

        AtomicInteger i = new AtomicInteger(0);
        return Stream.generate(() -> null)
                .takeWhile(x -> it.hasNext())
                .map(n -> it.next())
                .map(JsonElement::getAsString)
                .collect(Collectors.toMap(t -> t, t -> i.getAndIncrement()));
    }

    public double[] extractFrom(String[] tokens) {
        int vocabularySize = this.indexByWord.size();
        double[] res = new double[vocabularySize + 2];
        int tokenCount = Math.min(tokens.length, this.maxTokensPerString);


        long semicolonCount = 0;
        for (int i = 0; i < tokenCount; i++) {
            String token = tokens[i];
            Integer position = this.indexByWord.get(token);
            if (position != null) {
                res[position]++;
            }
            semicolonCount += token.chars().filter(t -> t == ';').count();
        }

        res[vocabularySize] = semicolonCount;
        res[vocabularySize + 1] = semicolonCount / ((double) tokenCount);

        return res;
    }
}
