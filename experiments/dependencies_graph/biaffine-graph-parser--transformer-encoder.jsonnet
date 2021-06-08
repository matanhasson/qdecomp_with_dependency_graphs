/*
nohup python scripts/train/run_experiments.py train --experiment experiments/dependencies_graph/biaffine-graph-parser--transformer-encoder.jsonnet -s tmp/graph-parser -o arc_tags_only:false multi_label:false --no_eval; nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser; nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser --all &
*/
local arc_tags_only = false; // std.parseJson(std.extVar("arc_tags_only"));
local multi_label = false; // std.parseJson(std.extVar("multi_label"));
local logical_form_em = true;

local transformer_model = "bert-base-uncased";
local max_length = 128;
local transformer_dim = 768;

local lr_scheduler = std.extVar("lr_scheduler"); // "slanted" | "linear" | 'polynomial' ...

//hyperparameters
local pos_embedding_dim = 100;
local input_dropout = 0.6;
local dropout = 0.3;
local arc_representation_dim = 300;
local tag_representation_dim = 300;
local arc_num_layers = 3;
local tag_num_layers = 3;

local seed = 24;
local lr = 1e-3;
local transformer_lr = 5e-5;
local batch_size = 32;
local use_amp = true;

local learning_rate_scheduler=
    if lr_scheduler == 'slanted' then {
      "type": "slanted_triangular",
      "cut_frac": 0.06,
    } else if lr_scheduler == 'custom_slanted' then {
      "type": "custom_slanted_triangular",
      "cut_frac": 0.06,
      "freeze_while_warmup": [0]
    } else if lr_scheduler == 'linear' then {
      "type": "linear_with_warmup",
      "warmup_steps": 100
    } else if lr_scheduler == 'polynomial' then{
      "type": "polynomial_decay",
      "power": 1.0,
      "warmup_steps": 0,
    } else {
      "type": lr_scheduler
    };


local embedding_dim = transformer_dim + pos_embedding_dim;

{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": "datasets/Break/QDMR/train_dependencies_graph.json",
  "validation_data_path": "datasets/Break/QDMR/dev_dependencies_graph.json",
  "test_data_path": "datasets/Break/QDMR/test_dependencies_graph__questions_only.json",

  "dataset_reader": {
    "type": "dependencies_graph",
    "fill_none_tags": arc_tags_only && !multi_label,
    "multi_label": multi_label,
    "deps_tags_namespace": "labels",
    "word_field": "text",
    "pos_field": "tag",
    "pos_tags_namespace": "pos_tags",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length
      },
    },
  },
  "model": {
    "type": "biaffine_graph_parser",
    "arc_tags_only": arc_tags_only,
    "multi_label": multi_label,
    [if logical_form_em then "graph_based_metric"]: {
      "type": "logical_form_em_for_graph"
    },
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length,
//        "train_parameters": false
        }
      }
    },
    "pos_tag_embedding":{
       "embedding_dim": pos_embedding_dim,
       "vocab_namespace": "pos_tags",
       "sparse": false  # huggingface_adamw cannot work with sparse
    },
    "encoder": {
     "type": "pass_through",
     "input_dim": embedding_dim,
    },
    "arc_representation_dim": arc_representation_dim,
    "tag_representation_dim": tag_representation_dim,
    "arc_feedforward": {
      "input_dim": embedding_dim,
      "num_layers": arc_num_layers,
      "hidden_dims": arc_representation_dim,
      "activations": "elu",
    },
    "tag_feedforward": {
      "input_dim": embedding_dim,
      "num_layers": tag_num_layers,
      "hidden_dims": tag_representation_dim,
      "activations": "elu",
    },
    "dropout": dropout,
    "input_dropout": input_dropout, // "dropout" is enough (no encoder) ?
    "initializer": {
      "regexes": [
        [".*feedforward.*weight", {"type": "xavier_uniform"}],
        [".*feedforward.*bias", {"type": "zero"}],
        [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
        [".*tag_bilinear.*bias", {"type": "zero"}],
      ]
    }
   },
   "data_loader": {
      "batch_sampler": {
        "type": "bucket",
         "batch_size": batch_size,
     }
  },
  "trainer": {
    "use_amp": use_amp,
    "num_epochs": 80,
    "grad_norm": 1.0,
    "cuda_device": 0,
    "validation_metric": if logical_form_em then "+logical_form_em" else (if multi_label then "+arcs_and_tags_f1" else "+arcs_and_tags_micro_fscore"),
    "learning_rate_scheduler": learning_rate_scheduler,
    "optimizer": {
       "type": "huggingface_adamw",
       "lr": lr,
       "weight_decay": 0.01,
       "parameter_groups": [
         [[".*transformer.*"], {"lr": transformer_lr}]
       ]
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
   },

//   "distributed": {
//    "cuda_devices": [0, 1, 2, 3]
//   }
}