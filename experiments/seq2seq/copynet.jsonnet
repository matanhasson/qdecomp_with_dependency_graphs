local embedding_dim = std.parseInt(std.extVar("embedding_dim")); //150
local logical_form_em = true;

{
  "dataset_reader":{
    "type":"break_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      }
    }
  },

  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "model": {
    "type": "custom_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "vocab_namespace": "source_tokens",
          "trainable": true,
        },
      },
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim,
      "hidden_size": 150,
      "num_layers": 5,
      "dropout": 0.2,
      "bidirectional": false
    },

    "max_decoding_steps": 200,
    "attention": "dot_product",
    "beam_size": 5,

    [if logical_form_em then "token_based_metric"]: {
       "type": "logical_form_em_for_seq2seq"
     },
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
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 2
    },
    [if logical_form_em then "validation_metric"]: "+logical_form_em",
  }
}