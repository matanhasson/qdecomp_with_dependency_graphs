local BASE = import 'copynet.jsonnet';

BASE+ {
  "dataset_reader"+: {
    "source_token_indexers"+: {
      "tokens"+: {
          "lowercase_tokens": true
      },
       "elmo": {
          "type": "elmo_characters"
      }
    },
  },

  "model"+: {
    "source_embedder"+: {
    "token_embedders"+:{
//      "tokens": {
//        "type": "embedding",
//        "embedding_dim": 100,
//        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
//        "trainable": true
//      },
      "tokens"+: {
        "embedding_dim": 300,
        "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.840B.300d.txt.gz",
        "trainable": true
      },
      "elmo": {
        "type": "elmo_token_embedder",
        "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
        "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5",
        "do_layer_norm": false,
        "dropout": 0.5
      }
      },
    },

    "encoder"+: {
      "input_size": 300 + 128 * 2 #(biLM)
    },
  },
}
