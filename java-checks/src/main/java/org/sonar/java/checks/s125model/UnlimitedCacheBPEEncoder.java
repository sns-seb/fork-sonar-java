package org.sonar.java.checks.s125model;

import java.util.HashMap;
import java.util.Map;

/**
 * A {@link BPEEncoder} which caches all the values computed by a delegate {@link BPEEncoder} implementation.
 * <p>
 * Size of the cache as well as the number of hits and cache misses are provided.
 */
public class UnlimitedCacheBPEEncoder implements BPEEncoder {
    private final BPEEncoder delegate;
    private final Map<CharSequence, String[]> cache = new HashMap<>();
    private long hits = 0;

    public UnlimitedCacheBPEEncoder(BPEEncoder delegate) {
        this.delegate = delegate;
    }

    @Override
    public String[] bpeEncode(CharSequence text) {
        hits++;
        return cache.computeIfAbsent(text, delegate::bpeEncode);
    }

    public int size() {
        return cache.size();
    }

    public long hits() {
        return hits;
    }

    public void clear() {
        this.cache.clear();
        this.hits = 0;
    }
}
