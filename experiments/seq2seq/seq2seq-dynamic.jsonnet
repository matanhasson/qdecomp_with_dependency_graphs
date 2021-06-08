{
  "dataset_reader":{
    "type":"break_seq2seq",
    "dynamic_vocab": true,

    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens"
      }
    },

    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "tokens"
      }
    }
  },

  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "model": {
    "type": "simple_seq2seq_dynamic",
    "source_embedder": {
      "token_embedders":{
          "tokens": {
            "type": "embedding",
            "embedding_dim": 150,
            "vocab_namespace": "tokens",
            "trainable": true,
          },
      },
    },

    "encoder": {
      "type": "lstm",
      "input_size": 150,
      "hidden_size": 150,
      "num_layers": 5,
      "dropout": 0.2
    },

    "max_decoding_steps": 200,
    "attention": "dot_product",
    "beam_size": 5
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["source_tokens"],
        "batch_size": 32
       }
  },

  "trainer": {
    "num_epochs": 80,
    "patience": 15,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}