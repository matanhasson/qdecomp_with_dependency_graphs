/*
nohup python scripts/train/run_experiments.py train --experiment experiments/dependencies_graph/pairwised-encoding-graph-parser.jsonnet -s tmp/graph-parser -o LF:false ENC:transformer PENC:none COMB:subtract --no_eval; nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser; nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser --all
*/

local transformer_model = "bert-base-uncased";
local max_length = 128;
local transformer_dim = 768;

local deps_tags_namespace = "labels";

//hyperparameters
local logical_form_em = std.parseJson(std.extVar("LF")); // false;
local encoder_to_use = std.extVar("ENC"); //"transformer"; // "none" | "transformer"
local pairs_encoder_to_use = std.extVar("PENC"); //"none"; // "none" | "ff"
local pair_combination = std.extVar("COMB"); // "subtract"; // "concat" | "subtract"

local seed = 24;
local input_dropout = 0.3;
local dropout = 0.3;
local lr = 1e-3;
local transformer_lr = 1e-5;


// encoders
local encoder_input_dim = transformer_dim;
local pairs_encoder_input_dim = if pair_combination == "concat" then 2*encoder_input_dim else encoder_input_dim;

local encoder = {
  "none": {
    "type": "pass_through",
    "input_dim": encoder_input_dim
  },
  "transformer":{
    "type": "pytorch_transformer",
    "input_dim": encoder_input_dim,
    "feedforward_hidden_dim": encoder_input_dim,
    "num_layers": 5,
    "num_attention_heads":4,
    "dropout_prob": 0.1,
  },
}[encoder_to_use];

local pairs_encoder = {
  "none": {
    "type": "pass_through",
    "input_dim": pairs_encoder_input_dim
  },
  "ff": {
    "type": "feedforward",
    "feedforward": {
      "input_dim": pairs_encoder_input_dim,
      "num_layers": 3,
      "hidden_dims": pairs_encoder_input_dim,
      "activations": "relu",
      "dropout": 0.1
    },
  },
}[pairs_encoder_to_use];



{
  "numpy_seed": seed,
  "pytorch_seed": seed,
  "random_seed": seed,

  "train_data_path": "datasets/Break/QDMR/train_dependencies_graph.json",
  "validation_data_path": "datasets/Break/QDMR/dev_dependencies_graph.json",

  "dataset_reader":{
//    "max_instances": 100,
    "type": "dependencies_graph",
    "fill_none_tags": true,
    "pos_field": "tag",
    "deps_tags_namespace": deps_tags_namespace,
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
  },

  "model": {
     "type": "pairwise_graph_parser",
     "labels_namespace": deps_tags_namespace,
     "input_dropout": input_dropout,
     "dropout": dropout,
     "pair_combination": pair_combination,
     [if logical_form_em then "graph_based_metric"]: {
       "type": "logical_form_em_for_graph"
     },
     "text_field_embedder": {
       "token_embedders": {
         "tokens": {
           "type": "pretrained_transformer_mismatched",
           "model_name": transformer_model,
           "max_length": max_length,
         }
       }
     },
     "encoder": encoder,
     "pairs_encoder": pairs_encoder,
   },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["tokens"],
        "batch_size": 64,
     }
  },

  "trainer": {
    "num_epochs": 80,
    "grad_norm": 5.0,
    "cuda_device": 0,
    "validation_metric": if logical_form_em then "+logical_form_em" else "+arcs_and_tags_micro_fscore",
    "learning_rate_scheduler": {
     "type": "slanted_triangular",
     "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["_text_field_embedder.*transformer.*"], {"lr": transformer_lr}]
      ]
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
  }
}