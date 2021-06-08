/*
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python scripts/train/run_experiments.py train --experiment experiments/hybrid/copynet--rat-encoder.jsonnet -s tmp/hybrid -o seed:13370 > _logs/hybrid.txt &
*/

local transformer_model= "bert-base-uncased";
local embedding_dim = 768;
local max_length = 128;
local logical_form_em = true;

local seed = std.parseInt(std.extVar("seed")); //13370

{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": "datasets/Break/QDMR/train_seq2seq_with_graphs.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq_with_graphs.csv",

  "dataset_reader":{
    "type":"break_copynet_seq2seq_rat",
    "dependencies_namespace": "relations_tags",
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
    "type": "custom_copynet_seq2seq_for_rat",
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
      "type": "relation_aware_transformer",
      "relations_namespace": "relations_tags",
      "num_layers": 8,
      "num_heads": 8,
      "hidden_size": embedding_dim,
      "ff_size" : embedding_dim,
      "tie_layers": false,
      "dropout": 0.1,
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