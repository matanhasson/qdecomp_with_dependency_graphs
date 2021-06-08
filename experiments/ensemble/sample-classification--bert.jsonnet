/*
nohup python scripts/train/run_experiments.py train --experiment experiments/ensemble/sample-classification--bert.jsonnet -s tmp/ensemble
*/

local transformer_model= "bert-base-uncased";
local hidden_size = 768;
local max_length = 128;
local tokenizer_kwargs = {
  "additional_special_tokens": ["@@SEP@@", "@@sep@@"] + ["@@%d@@"%x for x in std.range(1, 30)],
  };
local tokens_namespace = "tags";
local scheduler = std.extVar("scheduler"); // "slanted" | "none"

// hyperparameters
local num_epochs = 100;
local dropout = 0.3;
local pooler_dropout = 0.3;


{
  "train_data_path": "datasets/ensemble/2021-01-18/train.json",
  "validation_data_path": "datasets/ensemble/2021-01-18/dev.json",

  "dataset_reader":{
//    "max_instances": 100,
    "type": "text_classification_json",
    "token_indexers":{
      "tokens":{
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": max_length,
        "tokenizer_kwargs" : tokenizer_kwargs,
        "namespace": tokens_namespace,
      }
    },
    "tokenizer":{
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": max_length,
      "tokenizer_kwargs" : tokenizer_kwargs,
    }
  },

  "model": {
     "type": "basic_classifier",
     "text_field_embedder": {
       "token_embedders": {
         "tokens": {
           "type":"pretrained_transformer",
           "model_name": transformer_model,
           "max_length": max_length,
           "train_parameters": true,
         }
       }
     },
     "seq2vec_encoder":{
       "type": "bert_pooler",
       "pretrained_model": transformer_model,
       "dropout": pooler_dropout,
     },
     "dropout": dropout,
     "namespace": tokens_namespace,
   },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
         "batch_size": 128
     }
  },

  "trainer": {
    "num_epochs": num_epochs,
    "validation_metric": "+accuracy",
    "grad_norm": 1.0,
    "optimizer":{
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    [if scheduler=="slanted" then "learning_rate_scheduler"]: {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
  }
}