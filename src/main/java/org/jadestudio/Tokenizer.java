package org.jadestudio;

import java.util.List;

public interface Tokenizer {
    /**
     *  tokenize text
     * @param text
     * @return tokenized words
     */
    List<String> tokenize(String text);
}
