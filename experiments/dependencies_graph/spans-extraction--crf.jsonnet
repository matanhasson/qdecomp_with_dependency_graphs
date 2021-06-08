{
  "train_data_path": "datasets/Break/QDMR/train_spans_BIO.txt",
  "validation_data_path": "datasets/Break/QDMR/dev_spans_BIO.txt",

  "dataset_reader":{
    "type":"sequence_tagging",
  },

  "model": {
    "type": "crf_tagger",
    "text_field_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": 300,
          "trainable": true,
        },
      },
    },

    "encoder": {
      "type": "lstm",
      "input_size": 300,
      "hidden_size": 150,
      "num_layers": 2,
      "dropout": 0.2,
      "bidirectional": true
    },
    "label_encoding": "BIO",
    "constrain_crf_decoding": true,
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size": 32
    }
  },

  "trainer": {
    "cuda_device": 0,
    "num_epochs": 25,
    "optimizer": {
      "type": "adam",
      "lr": 0.001
    },
    "patience": 10,
    "validation_metric": "+f1-measure-overall",
    "checkpointer": {
      "num_serialized_models_to_keep": 2
    },
  }
}