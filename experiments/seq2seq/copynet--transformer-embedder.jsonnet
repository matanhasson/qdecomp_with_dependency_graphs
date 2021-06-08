/*
nohup python scripts/train/run_experiments.py train --experiment experiments/seq2seq/copynet--transformer-embedder.jsonnet -s tmp/copynet-transformer-embedder -o transformer_only:false embedding_dim:450 hidden_size:450 num_layers:3 optimizer.lr:1e-3
*/
local BASE = import 'copynet.jsonnet';

//local seed = std.parseInt(std.extVar("seed")); //13370
//local num_epochs = std.parseInt(std.extVar("num_epochs")); //25
//local train_examples = std.parseInt(std.extVar("train_examples"));
//local batch_size =  std.extVar("BATCH");
//local gradient_accumulation_steps = 1;
local transformer_only = std.parseJson(std.extVar("transformer_only"));

local transformer_model= "bert-base-uncased";
local hidden_size = 768;
local max_length = 128;
local train_parameters = false;

local indexer_name = if transformer_only then "tokens" else "tokens_transformer";
local input_size = if transformer_only then hidden_size else BASE.model.encoder.input_size + hidden_size;


BASE+ {
//  "random_seed": seed,
//  "numpy_seed" : std.floor($.random_seed / 10),
//  "pytorch_seed" : std.floor($.numpy_seed / 10),

  "dataset_reader"+:{
    "source_token_indexers"+: {
      [indexer_name]:{
          "type": "pretrained_transformer_mismatched",
          "model_name": transformer_model,
//          "max_length": max_length,
      }
    }
  },

  "model"+: {
    "source_embedder"+: {
        "token_embedders"+: {
            [indexer_name]: {
                "type":"pretrained_transformer_mismatched",
                "model_name": transformer_model,
//                "max_length": max_length,
                "train_parameters": train_parameters,
            },
         },
     },
    "encoder"+: {
      "input_size": input_size,
    },
  },
 }