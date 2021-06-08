local model_name = "facebook/bart-base";
local model_size = 768;
local special_tokens = ["@@SEP@@", "@@sep@@"] + ["@@%d@@"%x for x in std.range(1, 30)];

local seed = 24; //std.parseInt(std.extVar("seed"));
local lr_embedder = std.parseJson(std.extVar("lr_embedder")); //1e-5
//local lr_encoder = std.parseJson(std.extVar("lr_encoder")); //1e-5
local lr_decoder = std.parseJson(std.extVar("lr_decoder")); //1e-5

local lr_scheduler = std.extVar("lr_scheduler"); // "slanted" | "linear"
local learning_rate_scheduler=
    if lr_scheduler == 'slanted' then {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    } else if lr_scheduler == 'linear' then {
      "type": "linear_with_warmup",
      "warmup_steps": 0
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

  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "dataset_reader":{
//    "max_instances": 100,
    "type":"break_seq2seq",
    "source_add_start_token": false,
    "source_add_end_token": false,
    "target_add_start_token": false,
    "target_add_end_token": false,
    "source_token_indexers": {
       "tokens": {
         "type": "pretrained_transformer_mismatched",
         "model_name": model_name,
         "namespace": "source_tokens_tags",
       }
    },
    "target_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name,
      "tokenizer_kwargs": {
        "additional_special_tokens": special_tokens
      }
    },
    "target_token_indexers": {
       "tokens": {
         "type": "pretrained_transformer",
         "model_name": model_name,
         "namespace": "target_tokens_tags",
         "tokenizer_kwargs": {
           "additional_special_tokens": special_tokens
         }
       }
    },
  },

  "model":{
    "type": "custom_bart",
    "model_name": model_name,
    "indexer": {
      "model_name": model_name,
      "namespace": "target_tokens_tags",
      "tokenizer_kwargs": {
        "additional_special_tokens": special_tokens
      }
    },
    "embedder": {
      "token_embedders":{
        "tokens":{
          "type": "custom_pretrained_transformer_mismatched",
          "model_name": model_name,
          "sub_module": "encoder",
          "aggregation": "average",
        }
      }
    },
    "encoder": {
      "type": "pass_through",
      "input_dim": model_size,
    },
    "token_based_metric": {
      "type": "logical_form_em_for_seq2seq"
    },
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["source_tokens"],
      "batch_size": 16
     }
  },

  "trainer": {
    "num_epochs": 10,
    "patience": 10,
    "validation_metric": "+logical_form_em",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": lr_decoder,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "correct_bias": true,
      "weight_decay": 0.01,
      "parameter_groups": [
        [["_source_embedder.*"], {"lr": lr_embedder}],
//        [["_encoder.*"], {"lr": lr_encoder}],
      ],
    },
   "learning_rate_scheduler": learning_rate_scheduler,
//    "learning_rate_scheduler": {
//      "type": "polynomial_decay",
//    },
    "grad_norm": 1.0,
  }
}