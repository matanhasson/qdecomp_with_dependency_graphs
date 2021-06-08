local transformer_model= "bert-base-uncased";
local embedding_dim = 768;
local max_length = 128;
local logical_form_em = true;

local seed = std.parseInt(std.extVar("seed")); //13370
local rat_num_layers = 8;
local rat_num_heads = 8;

local relation_encoder = {
 "type": "feedforward",
 "feedforward": {
   "input_dim": embedding_dim,
   "num_layers": 3,
   "hidden_dims": embedding_dim/rat_num_heads,
   "activations": "relu",
   "dropout": 0.1
 }
};

{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": "datasets/Break/QDMR/train_seq2seq_with_graphs.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq_with_graphs.csv",

  "dataset_reader":{
//    "max_instances": 100,
    "type":"break_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length,
        "namespace": "source_tokens"
      },
    }
  },

  "model": {
    "type": "custom_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_embedder": {
      "token_embedders":{
        "tokens": {
          "type":"pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length,
          "train_parameters": true,
        },
      },
    },

    "encoder": {
      "type": "latent_relation_aware_transformer",
      "num_layers": rat_num_layers,
      "num_heads": rat_num_heads,
      "hidden_size": embedding_dim,
      "ff_size" : embedding_dim,
      "tie_layers": false,
      "dropout": 0.1,
      "relation_k_encoder": relation_encoder,
      "relation_v_encoder": relation_encoder,
    },

    "max_decoding_steps": 200,
    "attention": "dot_product",
    "beam_size": 5,

    [if logical_form_em then "token_based_metric"]: {
       "type": "logical_form_em_for_seq2seq"
     },

    "initializer": {
      "regexes": [
        ["_encoder.*self_attn.*weight", {"type": "xavier_uniform"}],
        ["_encoder.*self_attn.*bias", {"type": "zero"}],
        ["_encoder.*feed_forward.*weight", {"type": "xavier_uniform"}],
        ["_encoder.*feed_forward.*bias", {"type": "zero"}],
      ]
    }
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
    "cuda_device": 0,
    "grad_norm": 1.0,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["_source_embedder.*transformer.*"], {"lr": 5e-5}]
      ]
    },
    "learning_rate_scheduler": {
       "type": "slanted_triangular",
       "cut_frac": 0.06
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    [if logical_form_em then "validation_metric"]: "+logical_form_em",
  }
}