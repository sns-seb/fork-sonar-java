package org.sonar.java.checks.s125model;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Implementation in Java of the RoBERTa tokenizer from Hugging Face
 * (<a href="https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/models/roberta/tokenization_roberta.py#L107">source</a>).
 * <p>
 * <strong>Limitations</strong> of the port to Java:
 * <ul>
 *     <li>supports only tokenization of text to an array of String. Mapping to integer is left to
 *     responsibility of the caller</li>
 *     <li>AddedToken are not supported (aka. level1 tokenization, see below)</li>
 * </ul>
 * <p>
 * RoBERTa tokenization can be separated into 4 levels, which are applied sequentially in order to achieve the expected
 * tokenization:
 * <ul>
 *     <li>level1: split over AddedTokens (not supported)</li>
 *     <li>level2: split over RegExp (see {@link #level2TokenizationPattern})</li>
 *     <li>level3: byte-level encoding of level2 tokens (see {@link #level3TokenizationImpl(String[])})</li>
 *     <li>level4: BPE encoding of level3 tokens (implemented in {@link BPEEncoder}</li>
 * </ul>
 * <p>
 * This class provides the {@link Cache} interface which can be implemented and provided in the constructor to cache
 * the encoding at any of level2, level3 or level4.
 * <p>
 * This class provides the {@link Listener} interface which can be implemented and provided in the constructor to
 * monitor each intermediate steps of the tokenization.
 */
public class RoBERTaTokenizer {
    private static final Listener NOOP_LISTENER = new Listener() {
        @Override
        public void level1Tokens(String[] level1Tokens) {
            // NO OP
        }

        @Override
        public void level2Tokens(String[] level1Tokens, int level1TokenIndex, String[] level2Tokens) {
            // NO OP
        }

        @Override
        public void level3Tokens(String[] level1Tokens, int level1TokenIndex, String[] level3Tokens) {
            // NO OP
        }

        @Override
        public void level4Tokens(String[] level1Tokens, int level1TokenIndex, String[] level3Tokens, int level3TokenIndex, String[] level4Tokens) {
            // NO OP
        }
    };
    private static final Cache NOOP_CACHE = new Cache() {
        @Override
        public String[] level2Tokenization(String level1Token, Function<String, String[]> level2Tokenizer) {
            return level2Tokenizer.apply(level1Token);
        }

        @Override
        public String[] level3Tokenization(String[] level2Tokens, UnaryOperator<String[]> level3Tokenizer) {
            return level3Tokenizer.apply(level2Tokens);
        }
    };
    /**
     * Sub-patterns of the regular expression splitting the text at level2.
     * <p>
     * The original pattern is defined in
     * <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L242">RobertTokenizer.__init__</a>:
     * <pre>
     * self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
     * </pre>
     */
    private static final List<String> LEVEL_2_TOKENIZATION_SUB_PATTERNS = List.of(
            "'s",
            "'t",
            "'re",
            "'ve",
            "'m",
            "'ll",
            "'d",
            " ?\\p{L}+",
            " ?\\p{N}+",
            " ?[^\\s\\p{L}\\p{N}]+",
            "\\s+(?!\\S)",
            "\\s+"
    );
    private final Pattern level2TokenizationPattern;
    private final char[] level3UnicodeEncodingTable;
    private final BPEEncoder bpeEncoder;
    private final Listener listener;
    private final Cache cache;

    public RoBERTaTokenizer(BPEEncoder bpeEncoder) {
        this(bpeEncoder, NOOP_LISTENER);
    }

    protected RoBERTaTokenizer(BPEEncoder bpeEncoder, Cache cache) {
        this(bpeEncoder, NOOP_LISTENER, cache);
    }

    protected RoBERTaTokenizer(BPEEncoder bpeEncoder, Listener listener) {
        this(bpeEncoder, listener, NOOP_CACHE);
    }

    protected RoBERTaTokenizer(BPEEncoder bpeEncoder, Listener listener, Cache cache) {
        this.level2TokenizationPattern = Pattern.compile(String.join("|", LEVEL_2_TOKENIZATION_SUB_PATTERNS));
        this.level3UnicodeEncodingTable = buildLevel3UnicodeEncodingTable();
        this.bpeEncoder = bpeEncoder;
        this.listener = listener;
        this.cache = cache;
    }

    /**
     * Builds the mapping table used to encode 1-byte characters (ie. ASCII characters) to characters which can all
     * be read as text.
     * <p>
     * In other words, all flavors of invisible characters (including blank space) and escape characters in ASCII-table
     * are mapped to a characters with the same character number shifted by 256.
     * <p>
     * The original method to create this mapping table is defined into
     * <a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/tokenization_roberta.py#L69-L90">InferencePipeline#bytes_to_unicode()</a>:
     * <pre>
     * def bytes_to_unicode():
     *     """
     *     Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
     *     characters the bpe code barfs on.
     *
     *     The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
     *     if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
     *     decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
     *     tables between utf-8 bytes and unicode strings.
     *     """
     *     bs = (
     *         list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
     *     )
     *     cs = bs[:]
     *     n = 0
     *     for b in range(2**8):
     *         if b not in bs:
     *             bs.append(b)
     *             cs.append(2**8 + n)
     *             n += 1
     *     cs = [chr(n) for n in cs]
     *     return dict(zip(bs, cs))
     * </pre>
     */
    private static char[] buildLevel3UnicodeEncodingTable() {
        int asciiTableMaxIndex = (int) Math.pow(2, 8);
        char[] res = new char[asciiTableMaxIndex];
        int n = 0;
        for (int i = 0; i < asciiTableMaxIndex; i++) {
            if ((i >= '!' && i <= '~') || (i >= '¡' && i <= '¬') || (i >= '®' && i <= 'ÿ')) {
                res[i] = (char) i;
            } else {
                char[] chars = Character.toChars(asciiTableMaxIndex + n);
                if (chars.length > 1) {
                    throw new IllegalStateException("Integer number should not translate to more than one char");
                }
                res[i] = chars[0];
                n++;
            }
        }
        return res;
    }

    /**
     * @return the tokens representing the specified {@code text} according to the RoBERTa tokenization algorithm
     *         (see {@link RoBERTaTokenizer} for limitations).
     */
    public String[] tokenize(String text) {
        String[] level1Tokens = new String[]{text};
        listener.level1Tokens(level1Tokens);
        String level1Token = level1Tokens[0];
        String[] level2Tokens = level2Tokenization(level1Token);
        listener.level2Tokens(level1Tokens, 0, level2Tokens);
        String[] level3Tokens = level3Tokenization(level2Tokens);
        listener.level3Tokens(level1Tokens, 0, level3Tokens);

        List<String[]> allLevel4Tokens = new ArrayList<>(level3Tokens.length);
        for (int j = 0; j < level3Tokens.length; j++) {
            var level3Token = level3Tokens[j];
            String[] level4Tokens = level4Tokenization(level3Token);
            listener.level4Tokens(level1Tokens, 0, level3Tokens, j, level4Tokens);
            allLevel4Tokens.add(level4Tokens);
        }

        return allLevel4Tokens.stream()
                .flatMap(Arrays::stream)
                .toArray(String[]::new);
    }

    /**
     * @return the value returned by {@link Cache#level2Tokenization(String, Function)}, providing it
     *         with {@link #level2TokenizationImpl(String)} as the Function.
     */
    private String[] level2Tokenization(String text) {
        return cache.level2Tokenization(text, this::level2TokenizationImpl);
    }

    /**
     * Uses {@link #level2TokenizationPattern} to split the provided {@code text} into substrings limited by matches
     * to the sub-patterns in {@link #level2TokenizationPattern} (see {@link #LEVEL_2_TOKENIZATION_SUB_PATTERNS}),
     * <strong>in order</strong>.
     */
    private String[] level2TokenizationImpl(String text) {
        List<String> s = new ArrayList<>();
        Matcher matcher = this.level2TokenizationPattern.matcher(text);
        int start = 0;
        while (matcher.find()) {
            int end = matcher.start();
            if (end > 0) {
                s.add(text.substring(start, end));
            }
            start = end;
        }
        if (start < text.length()) {
            s.add(text.substring(start));
        }
        return s.toArray(String[]::new);
    }


    /**
     * @return the value returned by {@link Cache#level3Tokenization(String[], UnaryOperator)}, providing it
     *         with {@link #level3TokenizationImpl(String[])} as the Function.
     */
    private String[] level3Tokenization(String[] level2Tokens) {
        return cache.level3Tokenization(level2Tokens, this::level3TokenizationImpl);
    }

    /**
     * @return an array of the same size as {@code level2Tokens}, where each value has been encoded using
     *         {@link #level3Encode(String)}
     */
    private String[] level3TokenizationImpl(String[] level2Tokens) {
        String[] encodedTokens = new String[level2Tokens.length];
        for (int j = 0; j < level2Tokens.length; j++) {
            encodedTokens[j] = level3Encode(level2Tokens[j]);
        }
        return encodedTokens;
    }

    /**
     * Converts {@code token} to UTF-8 bytes and encodes each byte into a char per the mapping table
     * {@link #level3UnicodeEncodingTable}.
     * <p>
     * The returned {@link String} will have the same number, or more characters than {@code token}. All characters
     * encoded over 2 bytes in UTF-8 will be mapped to two 1-byte characters.
     *
     * @return the string made of the concatenated mapped chars of the UTF-8 encoding of {@code token}
     */
    private String level3Encode(String token) {
        byte[] chars = token.getBytes(UTF_8);
        char[] res = new char[chars.length];
        for (int i = 0; i < chars.length; i++) {
            // convert the byte value (which is a signed integer between -128 and 127 inclusive)
            // to an unsigned integer between 0 and 256 and then cast it
            char aChar = (char) (chars[i] & 0xFF);
            res[i] = this.level3UnicodeEncodingTable[aChar];
        }
        return new String(res);
    }

    /**
     * @return {@code level3Token} encoded with {@link BPEEncoder#bpeEncode(String)}
     */
    private String[] level4Tokenization(String level3Token) {
        return this.bpeEncoder.bpeEncode(level3Token);
    }

    /**
     * Interface to be notified of intermediate steps in the tokenization process.
     */
    public interface Listener {
        /**
         * The level1 tokens of the {@code text} passed to {@link #tokenize(String)}.
         * <p>
         * <strong>{@code level1Tokens} is always {@code new String[]{text}}</strong>, see limitations described in
         * {@link RoBERTaTokenizer}.
         */
        void level1Tokens(String[] level1Tokens);

        /**
         * The level2 tokens of the level1 token at index {@code level1TokenIndex} in {@code level1Tokens}.
         * <p>
         * <strong>{@code level1Tokens} is always {@code new String[]{text}}</strong> and <strong>{@code level1TokenIndex} is
         * always {@code 0}</strong>,see limitations described in {@link RoBERTaTokenizer}.
         */
        void level2Tokens(String[] level1Tokens, int level1TokenIndex, String[] level2Tokens);

        /**
         * The level3 tokens of the level1 token at index {@code level1TokenIndex} in {@code level1Tokens}.
         * <p>
         * <strong>{@code level1Tokens} is always {@code new String[]{text}}</strong> and <strong>{@code level1TokenIndex} is
         * always {@code 0}</strong>,see limitations described in {@link RoBERTaTokenizer}.
         */
        void level3Tokens(String[] level1Tokens, int level1TokenIndex, String[] level3Tokens);

        /**
         * The level4 tokens of the level3 token at index {@code level3TokenIndex} in {@code level3Tokens} of the
         * level1 token at index {@code level1TokenIndex} in {@code level1Tokens}.
         * <p>
         * <strong>{@code level1Tokens} is always {@code new String[]{text}}</strong> and <strong>{@code level1TokenIndex} is
         * always {@code 0}</strong>,see limitations described in {@link RoBERTaTokenizer}.
         */
        void level4Tokens(String[] level1Tokens, int level1TokenIndex, String[] level3Tokens, int level3TokenIndex, String[] level4Tokens);
    }

    /**
     * Interface which allows implementing a cache of any of the level2 and level3 tokenization stages.
     * <p>
     * Caching of level4 tokenization can be implemented directly into the {@link BPEEncoder} implementation provided
     * in the constructor.
     */
    public interface Cache {
        /**
         * @return either a cached {@code String[]} or the value returned by calling {@code level2Tokenizer} with
         *         {@code level1Token} as parameter.
         */
        String[] level2Tokenization(String level1Token, Function<String, String[]> level2Tokenizer);

        /**
         * @return either a cached {@code String[]} or the value returned by calling {@code level3Tokenizer} with
         *         {@code level2Tokens} as parameter.
         */
        String[] level3Tokenization(String[] level2Tokens, UnaryOperator<String[]> level3Tokenizer);
    }
}
