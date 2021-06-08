/*
nohup python scripts/train/run_experiments.py train --experiment experiments/hybrid/multitask--bart--latent-rat-encoder.jsonnet -s tmp/multitask-latent-rat -o separate_kv_classification:false rat_num_layers:2 rat_num_heads:2 feedforward.num_layers:3 scheduler:slanted batch_size:32
*/
local transformer_model= std.extVar("transformer_model"); // "facebook/bart-base" | "facebook/bart-large"
local embedding_dim = std.parseInt(std.extVar("transformer_size")); // 768 | 1024
local max_length = 128;
local special_tokens = ["@@SEP@@", "@@sep@@"] + ["@@%d@@"%x for x in std.range(1, 30)];


local pretrained_file = ""; //std.extVar("pretrained");
local lr_scheduler = std.extVar("lr_scheduler"); // "slanted" | "linear"

// hyperparameters
local data_rate_graph = std.parseJson(std.extVar("data_rate_graph"));
local graph_logical_form_em = true;
local separate_kv_classification = true;
local tie_rat_layers = true;
local combination_strategy = "multiply"; // "multiply" | "max"
local rat_num_layers = std.parseInt(std.extVar("rat_num_layers")); // 2;
local rat_num_heads = std.parseInt(std.extVar("rat_num_heads")); // 2;

local seed = 24; //std.parseInt(std.extVar("seed")); //13370
local lr = std.parseJson(std.extVar("lr_")); //1e-3
local lr_embedder = std.parseJson(std.extVar("lr_embedder")); //5e-5
local lr_encoder = std.parseJson(std.extVar("lr_encoder")); //1e-3
local lr_decoder = std.parseJson(std.extVar("lr_decoder")); //5e-5
local batch_size = 16;
//local effective_batch_size = std.parseInt(std.extVar("effective_batch_size"));
//local num_gradient_accumulation_steps = effective_batch_size/batch_size;
local use_amp = false; // doesn't work well


local train_data_loader = if data_rate_graph==1 then {} else {
  "instances_per_epoch": 44322,
  "sampler":{
    "type": "weighted",
    "weights": {
      "seq2seq": 1-data_rate_graph,
      "graph_parser": data_rate_graph,
    },
  }
};

local relation_encoding_dim = embedding_dim/rat_num_heads;
local relation_encoder = {
 "type": "feedforward",
 "feedforward": {
   "input_dim": embedding_dim,
   "num_layers": 3,
   "hidden_dims": relation_encoding_dim,
   "activations": "relu",
   "dropout": 0.1
 }
};

local learning_rate_scheduler=
    if lr_scheduler == 'slanted' then {
      "type": "slanted_triangular",
      "cut_frac": 0.06
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
  "test_data_path": {
    "seq2seq": "datasets/Break/QDMR/test_seq2seq.csv",
    "graph_parser": "datasets/Break/QDMR/test_dependencies_graph__questions_only.json"
  },

  "dataset_reader":{
    "type": "custom_multitask",
    "readers": {
        "seq2seq": {
//          "max_instances": 100,
          "type":"break_seq2seq",
          "source_add_start_token": false,
          "source_add_end_token": false,
          "target_add_start_token": false,
          "target_add_end_token": false,
          "source_token_indexers": {
            "tokens": {
              "type": "pretrained_transformer_mismatched",
              "model_name": transformer_model,
              //"max_length": max_length,
              "namespace": "source_tokens"
            }
          },
          "target_tokenizer": {
            "type": "pretrained_transformer",
            "model_name": transformer_model,
            //"max_length": max_length,
            "tokenizer_kwargs": {
              "additional_special_tokens": special_tokens
            }
          },
          "target_token_indexers": {
            "tokens": {
              "type": "pretrained_transformer",
              "model_name": transformer_model,
              //"max_length": max_length,
              "namespace": "target_tokens",
              "tokenizer_kwargs": {
                "additional_special_tokens": special_tokens
              }
            }
          },
        },
        "graph_parser": {
//          "max_instances": 100,
          "type": "dependencies_graph",
          "fill_none_tags": true,
          "multi_label": false,
          "deps_tags_namespace": "labels",
          "word_field": "text",
          "pos_field": "tag",
          "pos_tags_namespace": "pos_tags",
          "token_indexers": {
            "tokens": {
              "type": "pretrained_transformer_mismatched",
              "model_name": transformer_model,
//              "max_length": max_length,
              "namespace": "source_tokens"
            },
          },
        },
    },
  },

  "model": {
    "type": "multitask_rat",
    "tags_namespace": "labels",
    "separate_kv_classification": separate_kv_classification,
    "combination_strategy": combination_strategy,
    "relations_encoding_dim": relation_encoding_dim,
    "graph_loss_weight": 100,
    [if graph_logical_form_em then "graph_based_metric"]: {
      "type": "logical_form_em_for_graph"
    },
    "seq2seq_model":{
      "type": "custom_bart",
      "model_name": transformer_model,
      "indexer": {
        "model_name": transformer_model,
        //"max_length": max_length,
        "namespace": "target_tokens",
        "tokenizer_kwargs": {
          "additional_special_tokens": special_tokens
        }
      },
      "embedder": {
        "token_embedders":{
          "tokens":{
            "type": "custom_pretrained_transformer_mismatched",
            "model_name": transformer_model,
            //"max_length": max_length,
            "sub_module": "encoder",
          }
        }
      },
      "encoder": {
        "type": "latent_relation_aware_transformer",
        "num_layers": rat_num_layers,
        "num_heads": rat_num_heads,
        "hidden_size": embedding_dim,
        "ff_size" : embedding_dim,
        "tie_layers": tie_rat_layers,
        "dropout": 0.1,
        "relation_k_encoder": relation_encoder,
        "relation_v_encoder": relation_encoder,
      },

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
    },
    [if pretrained_file=="" then "initializer"]:{
      "regexes": [
        ["_classification_layer_.*weight", {"type": "xavier_uniform"}],
        ["_classification_layer_.*bias", {"type": "zero"}],
      ]
    },
    [if pretrained_file!="" then "initializer"]: {
      "regexes": [
        [".*", {"type": "pretrained", "weights_file_path": pretrained_file, "parameter_name_overrides": {}}],
      ]
    },
  },

  "data_loader": {
    "type": "multitask",
    "scheduler":{
      "batch_size": batch_size,
    },
  }+train_data_loader,
  "validation_data_loader": {
    "type": "multitask",
    "scheduler":{
      "batch_size": batch_size,
    },
  },

  "trainer": {
    "use_amp": use_amp,
    "num_epochs": 25,
    "grad_norm": 1.0,
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr,
      "weight_decay": 0.01,
      "parameter_groups": [
//        [["_seq2seq_model.*bart.*", "_seq2seq_model._source_embedder.*"], {"lr": transformer_lr}]
//        [[".*_source_embedder.*transformer.*"], {"lr": transformer_lr}]
        [["_embedder.*"], {"lr": lr_embedder}],
        [["_encoder.*"], {"lr": lr_encoder}],
        [["_seq2seq_model.*bart.*"], {"lr": lr_decoder}],
      ]
    },
    "learning_rate_scheduler": learning_rate_scheduler,
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    "validation_metric": "+maximal_logical_form_em",
//    [if num_gradient_accumulation_steps>1 then "num_gradient_accumulation_steps"]: num_gradient_accumulation_steps,
  }
}