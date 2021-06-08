local transformer_model = "bert-base-uncased";

local seed = std.parseInt(std.extVar("seed")); //13370
local num_epochs = std.parseInt(std.extVar("num_epochs")); // 4
local train_examples = std.parseInt(std.extVar("train_examples")); // 19500
//local batch_size =  std.extVar("batch_size");
local gradient_accumulation_steps = 1;

{
  "random_seed": seed,
  "numpy_seed" : std.floor($.random_seed / 10),
  "pytorch_seed" : std.floor($.numpy_seed / 10),

  "dataset_reader": {
    "type":"break_seq2seq",
    "source_add_start_token": false,
    "source_add_end_token": false,
    "target_add_start_token": false,
    "target_add_end_token": false,
    "source_tokenizer":{
       "type":"pretrained_transformer",
       "model_name": transformer_model,
    },
    "source_token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
        //"max_length": 150,
      },
    },
  },

  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "model": {
    "type":"pretrained-transformers-seq2seq",
    "encoder_model_name": transformer_model,
    "beam_size": 5,
    "trainable":true,
  },


  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size": 8,
    }
  },

  "trainer"+:{
    "num_epochs": num_epochs,
    "cuda_device": 0,
    //"patience": 10,
    "grad_norm": 1.0,
    "optimizer":{
       "type": "huggingface_adamw",
       "lr": 1e-5,
       "correct_bias": false,
       "weight_decay": 0.01,
    },
    "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "num_epochs": num_epochs,
            "num_steps_per_epoch": (std.ceil(train_examples / $.data_loader.batch_sampler.batch_size) / gradient_accumulation_steps),
    },
   }
  }
