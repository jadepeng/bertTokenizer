package org.jadestudio;

import java.util.ArrayList;
import java.util.List;


/**
 * 基础分词器
 * @author jqpeng
 */
public class BasicTokenizer implements Tokenizer {
    private boolean doLowerCase = true;
    private List<String> neverSplit;
    private boolean tokenizeChineseChars = true;

    public BasicTokenizer(boolean doLowerCase, List<String> neverSplit, boolean tokenizeChineseChars) {
        this.doLowerCase = doLowerCase;
        if (neverSplit == null) {
            neverSplit = new ArrayList<String>();
        }
        this.neverSplit = neverSplit;
        this.tokenizeChineseChars = tokenizeChineseChars;
    }

    public BasicTokenizer() {
    }

    @Override
    public List<String> tokenize(String text) {
        text = TokenizerUtils.cleanText(text);
        if (tokenizeChineseChars) {
            text = TokenizerUtils.tokenizeChineseChars(text);
        }
        List<String> originalTokens = TokenizerUtils.whitespaceTokenize(text);

        List<String> splitTokens = new ArrayList<String>();
        for (String token : originalTokens) {
            if (doLowerCase && !neverSplit.contains(token)) {
                token = TokenizerUtils.runStripAccents(token.toLowerCase());
                splitTokens.addAll(TokenizerUtils.runSplitOnPunc(token, neverSplit));
            }
        }
        return TokenizerUtils.whitespaceTokenize(String.join(" ", splitTokens));
    }

}