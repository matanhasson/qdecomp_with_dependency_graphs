local transformer_model = "bert-base-uncased";
local max_length = 128;
local transformer_dim = 768;

local logical_form_em = true;
local decode_strategy = "operators_mask";  // "operators_mask" | "probs_multiplication"
local tags_namespace = "labels";
local operators_namespace = "operators_labels";

//hyperparameters
local seed = 24;
local dropout = 0.3;
local input_dropout = 0.3;
local pos_embedding_dim = 100;
local operator_representation_dim = 300;
local tag_representation_dim = 300;
local operator_ff_num_layers = 1;
local tag_ff_num_layers = 1;
local operator_embeddings_dim = std.parseInt(std.extVar("operator_embeddings_dim")); // 100;  // set 0 for ignore

local lr = std.parseJson(std.extVar("lr_")); // 1e-3;
local transformer_lr = std.parseJson(std.extVar("transformer_lr")); // 5e-5;



// components
local encoder_dim = transformer_dim + pos_embedding_dim;
local operator_feedforward = {
  "input_dim": encoder_dim,
  "num_layers": operator_ff_num_layers,
  "hidden_dims": operator_representation_dim,
  "activations": "elu",
//  "dropout": 0.1
};

local tag_feedforward = {
  "input_dim": encoder_dim + operator_embeddings_dim,
  "num_layers": tag_ff_num_layers,
  "hidden_dims": tag_representation_dim,
  "activations": "elu",
//  "dropout": 0.1
};

{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": "datasets/Break/QDMR/train_dependencies_graph.json",
  "validation_data_path": "datasets/Break/QDMR/dev_dependencies_graph.json",

  "dataset_reader":{
//    "max_instances": 100,
    "type": "dependencies_graph_with_operators",
    "operators_namespace": operators_namespace,
    "deps_tags_namespace": tags_namespace,
    "pos_field": "tag",
    "token_indexers": {
       "tokens": {
         "type": "pretrained_transformer_mismatched",
         "model_name": transformer_model,
         "max_length": max_length,
      },
      "pos": {
        "type": "single_id",
        "feature_name": "tag_",
        "namespace": "pos_tags"
      }
    },
  },
  "model": {
     "type": "operators_aware_biaffine_graph_parser",
     "operators_namespace": operators_namespace,
     "tags_namespace": tags_namespace,
     "dropout": dropout,
     "input_dropout": input_dropout,
     "text_field_embedder": {
       "token_embedders": {
         "tokens": {
           "type": "pretrained_transformer_mismatched",
           "model_name": transformer_model,
           "max_length": max_length,
         },
         "pos": {
           "type": "embedding",
           "embedding_dim": pos_embedding_dim,
           "vocab_namespace": "pos_tags",
         }
       }
     },
     "encoder": {
       "type": "pass_through",
       "input_dim": encoder_dim,
     },
     "operator_representation_dim": operator_representation_dim,
     "operator_feedforward": operator_feedforward,
     "tag_representation_dim": tag_representation_dim,
     "tag_feedforward": tag_feedforward,
     [if operator_embeddings_dim>0 then "operator_embeddings_dim"]: operator_embeddings_dim,
     "decode_strategy": decode_strategy,
     [if logical_form_em then "graph_based_metric"]: {
       "type": "logical_form_em_for_graph"
     },
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

  "trainer": {
    "num_epochs": 80,
    "grad_norm": 1.0,
    "validation_metric": if logical_form_em then "+logical_form_em" else "-loss",
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