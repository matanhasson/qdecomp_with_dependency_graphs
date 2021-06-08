{
  "dataset_reader":{
    "type":"break_seq2seq",

    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    },

    "target_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "target_tokens"
      }
    }
  },

  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "model": {
    "type": "simple_seq2seq_custom",
    "source_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": 150,
          "vocab_namespace": "source_tokens",
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
    "target_namespace": "target_tokens",
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
    "num_epochs": 25,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    }
  }
}