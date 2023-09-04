# Bert Tokenizer

This repository contains java implementation of Bert Tokenizer. The implementation is referred from https://github.com/ankiteciitkgp/bertTokenizer

Support output onnx tensor for onnx model inference

## Usage

To get tokens from text:
```
String text = "Text to tokenize";
 BertTokenizer bertTokenizer = new BertTokenizer("D:\\model\\vocab.txt");
List<String> tokens = tokenizer.tokenize(text);
```

To get token ids using Bert Vocab:

```
List<Integer> token_ids = tokenizer.convert_tokens_to_ids(tokens);
```

To get :

```
List<Integer> token_ids = tokenizer.convert_tokens_to_ids(tokens);
```

To get onnx tensor

```
var inputMap = bertTokenizer.tokenizeOnnxTensor(Arrays.asList("hello world 你好", "肿瘤治疗未来发展趋势"));
```

Full example:

```java

public class OnnxTests {
    public static void main(String[] args) throws IOException, OrtException {
        BertTokenizer bertTokenizer = new BertTokenizer("D:\\model\\vocab.txt");

        var env = OrtEnvironment.getEnvironment();
        var session = env.createSession("D:\\model\\output\\onnx\\fp16_model.onnx",
                new OrtSession.SessionOptions());

        var inputMap = bertTokenizer.tokenizeOnnxTensor(Arrays.asList("hello world 你好", "肿瘤治疗未来发展趋势"));

        try (var results = session.run(inputMap)) {
            System.out.println(results);
            var embeddings = (float[][])results.get(0).getValue();
            for (var embedding : embeddings) {
                System.out.println(JSON.toJSONString(embedding));
            }
        }

    }
}

```