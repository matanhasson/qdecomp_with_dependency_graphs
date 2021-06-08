local BASE = import 'seq2seq.jsonnet';

local seed = std.parseInt(std.extVar("seed")); //13370
local num_epochs = std.parseInt(std.extVar("num_epochs")); // 4
local train_examples = std.parseInt(std.extVar("train_examples")); // 19500
//local batch_size =  std.extVar("batch_size");
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
  "dataset_reader"+: {
    "source_add_start_token": false,
    "source_add_end_token": false,
    "source_tokenizer":{
       "type":"pretrained_transformer",
       "model_name": transformer_model,
       "max_length": max_length,
    },
    "target_tokenizer":{
       "type":"spacy",
    },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        "max_length": max_length,
        //"use_starting_offsets": true,
      },
    },
  },

  "model"+: {
    [if tune then "type"]:"simple_seq2seq_custom",
    "source_embedder": {
        "token_embedders": {
            "tokens": {
//                "type": "transformers_pass_through",
//                "hidden_dim": 768,
                "type":"pretrained_transformer",
                "model_name": transformer_model,
                "train_parameters": tune || train,
            },
        },
    },
    "target_embedding_dim": 450,
    "encoder": {
//      "type": "pretrained_transformer",
//      "model_name": transformer_model,
//      "trainable": tune || train,
        "type": "pass_through",
        "input_dim": 768,
    },

    [if tune then "initializer"]: {
        "regexes":[
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
        //"sorting_keys": ["source_tokens"],
        "batch_size": 32
    }
  },

  "trainer"+:{
    "num_epochs": num_epochs,
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