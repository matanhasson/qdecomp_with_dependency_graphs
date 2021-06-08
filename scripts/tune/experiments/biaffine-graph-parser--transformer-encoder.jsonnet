/*
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python scripts/train/run_experiments.py train --experiment experiments/dependencies_graph/biaffine-graph-parser--transformer-encoder.jsonnet --dataset datasets/dependencies_graphs/2020-10-17 -s tmp/graph-parser -o arc_tags_only:false multi_label:false batch_size:64  --no_eval; PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser; PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python dependencies_graph/evaluation/evaluate_dep_graph.py graph -r tmp/graph-parser --all &
*/

local arc_tags_only = std.parseJson(std.extVar("arc_tags_only"));
local multi_label = std.parseJson(std.extVar("multi_label"));
local logical_form_em = true;

local transformer_model = std.extVar('transformer_model'); //"bert-base-uncased"
local max_length = std.parseInt(std.extVar('max_length')); // 128
local transformer_dim = std.parseInt(std.extVar('transformer_dim')); // 768

//hyperparameters
local pos_embedding_dim = std.parseInt(std.extVar('pos_embedding_dim')); //100
local input_dropout = std.parseJson(std.extVar('input_dropout')); //0.3
local dropout = std.parseJson(std.extVar('dropout')); //0.3
local arc_representation_dim = std.parseInt(std.extVar('arc_representation_dim')); //500
local tag_representation_dim = std.parseInt(std.extVar('tag_representation_dim')); //100
local arc_num_layers = std.parseInt(std.extVar('arc_num_layers')); //1
local tag_num_layers = std.parseInt(std.extVar('tag_num_layers')); //1

local seed = std.parseInt(std.extVar("seed")); //13370
local lr = std.parseJson(std.extVar('lr')); //1e-3
local transformer_lr = std.parseJson(std.extVar('transformer_lr')); //1e-5

local embedding_dim = transformer_dim + pos_embedding_dim;

{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": "datasets/Break/QDMR/train_dependencies_graph.json",
  "validation_data_path": "datasets/Break/QDMR/dev_dependencies_graph.json",

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
         "batch_size": 32
     }
  },
  "trainer"+: {
    "num_epochs": 80,
    "grad_norm": 5.0,
    "patience": 20,
    "cuda_device": 0,
    "validation_metric": if logical_form_em then "+logical_form_em" else (if multi_label then "+arcs_and_tags_f1" else "+arcs_and_tags_micro_fscore"),
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
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
   }
}
+
{
  trainer+:{
    epoch_callbacks: [
      {
        type: 'optuna_pruner',
      },
    ],
  }
}