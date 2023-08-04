package org.sonar.java.checks.s125model;

import java.util.Set;

/**
 * Feature extraction (see {@link RoBERTaTokenizer} and {@link FeatureExtractor}) should not apply to the comment signs
 * which delimits the comment because these signs are common to all comments, they can't convey any meaning, they can
 * only risk to bias the model's prediction.
 */
public class CommentPreparation {
    // based on https://github.com/SonarSource/rd-s125/blob/molecules-extractor/src/main/resources/Atomizer.g4
    private static final Set<String> JAVADOC_HEADERS = Set.of("/** ", "/**\t", "/**\n", "/**\r\n");
    private static final int JAVADOC_HEADER_SIZE = JAVADOC_HEADERS.iterator().next().length();
    private static final String LINE_COMMENT_HEADER = "//";
    private static final String BLOCK_COMMENT_HEADER = "/*";
    private static final String BLOCK_COMMENT_TRAILER = "*/";

    private enum CommentType {
        JAVADOC, LINE_COMMENTS_BLOCK, COMMENT_BLOCK
    }

    public String stripCommentSigns(String text) {
        return prepareComment(commentType(text), text);
    }

    private static String prepareComment(CommentType type, String text) {
        switch (type) {
            case JAVADOC:
                String s = text.substring(JAVADOC_HEADER_SIZE);
                if (s.endsWith(BLOCK_COMMENT_TRAILER)) {
                    return s.substring(0, s.length() - BLOCK_COMMENT_TRAILER.length());
                }
                return s;
            case LINE_COMMENTS_BLOCK:
                return text.substring(2).replace("\n//", "\n");
            // remove head "//" and then all which follow a line return (careful with platform line returns)
            case COMMENT_BLOCK:
                return text.substring(BLOCK_COMMENT_HEADER.length(), text.length() - BLOCK_COMMENT_TRAILER.length());
        }
        return null;
    }

    private static CommentType commentType(String comment) {
        String header = comment.substring(0, Math.min(JAVADOC_HEADER_SIZE, comment.length()));
        if (JAVADOC_HEADERS.contains(header)) {
            return CommentType.JAVADOC;
        } else if (header.startsWith(LINE_COMMENT_HEADER)) {
            return CommentType.LINE_COMMENTS_BLOCK;
        } else if (header.startsWith(BLOCK_COMMENT_HEADER)) {
            return CommentType.COMMENT_BLOCK;
        }
        throw new IllegalArgumentException("Unrecognized comment starting with \"" + header + "\"");
    }
}
