local BASE = import 'copynet.jsonnet';

local seed = std.parseInt(std.extVar("seed")); //13370
local num_epochs = std.parseInt(std.extVar("num_epochs")); //25
//local train_examples = std.parseInt(std.extVar("train_examples"));
//local batch_size =  std.extVar("BATCH");
//local gradient_accumulation_steps = 1;
local transformer_model= "bert-base-uncased";
local hidden_size = 768;
local max_length = 128;


BASE+ {
  "random_seed": seed,
  "numpy_seed" : std.floor($.random_seed / 10),
  "pytorch_seed" : std.floor($.numpy_seed / 10),

  "dataset_reader"+:{
    "source_token_indexers": {
      "tokens":{
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
//          "max_length": max_length,
      }
    }
  },

  "model"+: {
    "source_embedder": {
        "token_embedders": {
            "tokens": {
                "type":"pretrained_transformer_mismatched",
                "model_name": transformer_model,
//                "max_length": max_length,
//                "train_parameters": false,
            },
         },
     },
    "encoder": {
      "type": "pass_through",
      "input_dim": hidden_size,
    },
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        //"sorting_keys": [["source_tokens", "tokens___tokens"]],
        "batch_size": 32
    }
  },

  "trainer"+:{
    "num_epochs": num_epochs,
    "patience":: null, // because of slanted
    "grad_norm": 1.0,
    "optimizer":{
          "type": "huggingface_adamw",
          "lr": 1e-3,
          "weight_decay": 0.01,
          "parameter_groups": [
            [[".*transformer.*"], {"lr": 1e-5}]
          ]
    },
    "learning_rate_scheduler": {
       "type": "slanted_triangular",
       "cut_frac": 0.06
    },
   }
 }