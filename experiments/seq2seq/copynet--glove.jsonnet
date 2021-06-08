local BASE = import 'copynet.jsonnet';

BASE+ {
  "dataset_reader"+: {
    "source_token_indexers"+: {
      "tokens"+: {
        "lowercase_tokens": false
      },
    },
  },

  "model"+: {
    "source_embedder"+: {
      "token_embedders"+:{
        "tokens"+: {
        "embedding_dim": 300,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "trainable": false
        },
       },
    },

    "encoder"+: {
      "input_size": 300,
    },
  },
 }