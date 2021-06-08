/*
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python scripts/train/run_experiments.py train --experiment experiments/hybrid/copynet--transformer-encoder--graph-embedder.jsonnet -s tmp/hybrid -o embedding_dim:768 seed:13370 num_epochs:25 > _logs/hybrid.txt &
*/
local BASE = import 'copynet--graph-embedder.jsonnet';

// embedding_dim - should be set to 768
local seed = std.parseInt(std.extVar("seed")); //13370
local transformer_model= "bert-base-uncased";
//local graph_embedding_dim = (BASE.model.encoder.input_size - BASE.model.source_embedder.token_embedders.tokens.embedding_size);
//local embedding_dim = graph_embedding_dim + 768;
local max_length = 128;

local bilinear = std.parseJson(std.extVar("bilinear")); //false;

BASE+ {
  "random_seed": seed,
  "numpy_seed" : seed,
  "pytorch_seed" : seed,

  "dataset_reader"+:{
    "source_token_indexers"+: {
      "tokens"+:{
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
          "max_length": max_length,
      },
      "graph"+:{
        "original_sequence_mask_keyword": "original_sequence_mask" // prevent conflict with 'mask' of pretrained_transformer_tokens
      }
    }
  },

  "model"+: {
    "source_embedder"+: {
        "token_embedders"+: {
            "tokens": {
                "type":"pretrained_transformer_mismatched",
                "model_name": transformer_model,
                "max_length": max_length,
//                "train_parameters": false,
            },
            "graph"+:{
              "original_sequence_mask_keyword": "original_sequence_mask"
            }
         },
     },
    "encoder": {
      "type": "pass_through",
      "input_dim": BASE.model.encoder.input_size,
    },
    [if bilinear then "attention"]: {
      "type": "bilinear",
      "vector_dim": BASE.model.encoder.input_size,
      "matrix_dim": BASE.model.encoder.input_size
    },
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["source_tokens"],
        "batch_size": 16
    }
  },

  "trainer"+:{
    "patience":: null, // because of slanted
    "grad_norm": 1.0,
    "optimizer":{
          "type": "huggingface_adamw",
          "lr": 1e-3,
          "weight_decay": 0.01,
          "parameter_groups": [
            [[".*transformer.*"], {"lr": 5e-5}]
          ]
    },
     "learning_rate_scheduler": {
       "type": "slanted_triangular",
       "cut_frac": 0.06
    },
   }
 }