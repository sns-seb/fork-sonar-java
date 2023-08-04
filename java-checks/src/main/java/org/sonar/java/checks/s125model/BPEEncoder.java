package org.sonar.java.checks.s125model;

/**
 * A class performing the <a href="https://en.wikipedia.org/wiki/Byte_pair_encoding">BPE encoding</a> of a
 * {@link CharSequence}.
 */
public interface BPEEncoder {
    String[] bpeEncode(CharSequence text);
}
