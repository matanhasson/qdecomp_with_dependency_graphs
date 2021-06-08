local transformer_model= "bert-base-uncased";
local transformer_dim = 768;
local max_length = 128;

local tags_namespace = "labels";
local scheduler = std.extVar("lr_scheduler"); // "slanted" | "linear"

// hyperparameters
local seed = 24; //std.parseInt(std.extVar("seed")); //13370
local lr = 1e-3;
local transformer_lr = 5e-5;
local batch_size = 32;
local use_amp = true;

// seq2seq
local rat_num_layers = 2;
local rat_num_heads = 2;
// graph
local graph_logical_form_em = true;
local graph_input_dropout = 0.6;
local graph_dropout = 0.3;
local arc_representation_dim = 300;
local tag_representation_dim = 300;
local arc_num_layers = 3;
local tag_num_layers = 3;


local seq2seq_model = {
  "type": "custom_copynet_seq2seq_for_soft_rat",
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
    "type": "soft_relation_aware_transformer",
    "relations_namespace": tags_namespace,
    "num_layers": rat_num_layers,
    "num_heads": rat_num_heads,
    "hidden_size": transformer_dim,
    "ff_size" : transformer_dim,
    "tie_layers": true,
    "dropout": 0.1,
  },

  "max_decoding_steps": 200,
  "attention": "dot_product",
  "beam_size": 5,

  "token_based_metric": {
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
};

local graph_parser_model = {
  "type": "biaffine_graph_parser",
  "arc_tags_only": true,
  "multi_label": false,
  [if graph_logical_form_em then "graph_based_metric"]: {
    "type": "logical_form_em_for_graph"
  },
  "dropout": graph_dropout,
  "input_dropout": graph_input_dropout,
  "text_field_embedder": {
    "token_embedders": {
      "tokens": {
        "type":"pretrained_transformer_mismatched",
        "model_name": transformer_model,
        "max_length": max_length,
        "train_parameters": true,
      }
    }
  },

  "encoder": {
    "type": "pass_through",
    "input_dim": transformer_dim,
  },

  "arc_representation_dim": arc_representation_dim,
  "tag_representation_dim": tag_representation_dim,
  "arc_feedforward": {
    "input_dim": transformer_dim,
    "num_layers": arc_num_layers,
    "hidden_dims": arc_representation_dim,
    "activations": "elu",
  },
  "tag_feedforward": {
    "input_dim": transformer_dim,
    "num_layers": tag_num_layers,
    "hidden_dims": tag_representation_dim,
    "activations": "elu",
  },

  "initializer": {
    "regexes": [
      [".*feedforward.*weight", {"type": "xavier_uniform"}],
      [".*feedforward.*bias", {"type": "zero"}],
      [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
      [".*tag_bilinear.*bias", {"type": "zero"}],
    ]
  }
};


{
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "train_data_path": {
    "seq2seq": "datasets/Break/QDMR/train_seq2seq.csv",
    "graph_parser": "datasets/Break/QDMR/train_dependencies_graph.json"
  },
  "validation_data_path": {
    "seq2seq": "datasets/Break/QDMR/dev_seq2seq.csv",
    "graph_parser": "datasets/Break/QDMR/dev_dependencies_graph.json"
  },

  "dataset_reader":{
    "type": "custom_multitask",
    "readers": {
        "seq2seq": {
//          "max_instances": 100,
          "type":"break_copynet_seq2seq",
          "target_namespace": "target_tokens",
          "source_token_indexers": {
            "tokens": {
              "type": "pretrained_transformer_mismatched",
              "model_name": transformer_model,
              "max_length": max_length,
            }
          }
        },
        "graph_parser": {
//          "max_instances": 100,
          "type": "dependencies_graph",
          "fill_none_tags": true,
          "multi_label": false,
          "deps_tags_namespace": tags_namespace,
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
    },
  },

  "model": {
    "type": "multitask_soft_rat",
    "tags_namespace": tags_namespace,
    "zero_nones": true,
    "models":{
      "seq2seq": seq2seq_model,
      "graph_parser": graph_parser_model,
    },
    "tie_modules":[
      ["seq2seq._source_embedder", "graph_parser.text_field_embedder"],
    ],
    "loss_weights": {
      "graph_parser": 1000
    },
  },

  "data_loader": {
    "type": "multitask",
    "scheduler":{
      "batch_size": batch_size,
    },
  },

  "trainer": {
    "num_epochs": 25,
    "cuda_device": 0,
    "grad_norm": 1.0,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*_source_embedder.*transformer.*"], {"lr": transformer_lr}]
      ]
    },
    [if scheduler == "slanted" then "learning_rate_scheduler"]: {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    [if scheduler == "linear" then "learning_rate_scheduler"]: {
      "type": "linear_with_warmup",
      "warmup_steps": 100
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "validation_metric": "+maximal_logical_form_em",
  }
}
