local BASE = import 'copynet.jsonnet';

local seed = std.parseInt(std.extVar("seed")); //13370
local num_epochs = std.parseInt(std.extVar("num_epochs")); //4
local train_examples = std.parseInt(std.extVar("train_examples"));
//local batch_size =  std.extVar("BATCH");
local gradient_accumulation_steps = 1;
local pretrained_dir=std.extVar("pretrained_dir"); // "tmp/test-bert/data_old_version__seq2seq/seq2seq-bert_freeze/";
local tune = (pretrained_dir!="");
local train = tune;
local transformer_model= "bert-base-uncased";
local max_length = 150;


BASE+ {
  "random_seed": seed,
  "numpy_seed" : std.floor($.random_seed / 10),
  "pytorch_seed" : std.floor($.numpy_seed / 10),

  "dataset_reader"+:{
    "target_namespace": "tokens",
    "source_token_indexers"+: {
      "tokens"+: {
        "namespace": "tokens"
      }
    }
  },

  "model"+: {
    "target_namespace": "tokens",
    "source_embedder": {
        "token_embedders": {
            "tokens": {
                "type":"aggregated_pretrained_transformer",
                "model_name": transformer_model,
                "train_parameters": tune || train,
                "vocab_namespace": "tokens",
//                "vocab_namespace": "source_tokens",
            },
         },
     },
    "encoder": {
      "type": "pass_through",
      "input_dim": 768,
    },

     // tune
     [if tune then "initializer"]: {
        "regexes": [
            [".*",
                {
                    "type": "pretrained",
                    "weights_file_path": pretrained_dir+"best.th",
                    "parameter_name_overrides": {}
                }
            ],
        ],
      },
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        //"sorting_keys": [["source_tokens", "tokens___tokens"]],
        "batch_size": if tune then 16 else 32
    }
  },

  "trainer"+:{
    "num_epochs": num_epochs,
    "patience": 5,
    [if tune then "grad_norm"]: 1.0,
    [if tune then "optimizer"]+:{
       "type": "huggingface_adamw",
       "lr": 2e-5,
       "correct_bias": false,
       "weight_decay": 0.01,
       // warmup???
    },
    [if tune then "learning_rate_scheduler"]: {
            "type": "slanted_triangular",
            "num_epochs": num_epochs,
            "num_steps_per_epoch": (std.ceil(train_examples / $.data_loader.batch_sampler.batch_size) / gradient_accumulation_steps),
    },
   }
 }