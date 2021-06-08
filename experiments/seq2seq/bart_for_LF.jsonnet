local data_format = std.extVar("data_format"); // "special_tokens" | "sectors_rep"
local custom_special_tokens =
    if data_format == "special_tokens" then [
    //prop
    "@@%s_prop@@"%x for x in ["aggregate","arithmetic","boolean","comparative","comparison","discard","duplicate","filter","group","intersection","project","sort","span","superlative","union"]
    ]+[
    //args
    "@@%s@@"%x for x in ["aggregate_arg","arithmetic_arg","arithmetic_left","arithmetic_right","boolean_condition","boolean_sub","comparative_attribute","comparative_condition","comparative_sub","comparison_arg","discard_exclude","discard_sub","duplicate","filter_condition","filter_sub","group_key","group_value","intersection_intersection","intersection_projection","project_sub","sort_order","sort_order&sort_sub","sort_sub","span","superlative_attribute","superlative_sub","union_sub"]
    ]
    else if data_format == "sectors_rep" then ["@@PROP@@", "@@ARG@@", "@@ARG_VAL@@"]
    else [];

local model_name = "facebook/bart-base";
local special_tokens = ["@@SEP@@", "@@sep@@"] + ["@@%d@@"%x for x in std.range(1, 30)] + custom_special_tokens;

local seed = 24; //std.parseInt(std.extVar("seed"));
local use_config = false; //std.parseJson(std.extVar("use_config")); // true | false

local batch_size = 16; // base:16, large:8

local lr_scheduler = "slanted"; //std.extVar("lr_scheduler"); // "slanted" | "linear"
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

  "train_data_path": "datasets/Break/QDMR/train_lf_"+data_format+".csv",
//  "validation_data_path": "datasets/Break/QDMR/dev_lf_"+data_format+".csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",

  "dataset_reader":{
//    "max_instances": 100,
    "type":"break_seq2seq",
    "source_add_start_token": false,
    "source_add_end_token": false,
    "target_add_start_token": false,
    "target_add_end_token": false,
    "source_tokenizer": {
      "type": "pretrained_transformer",
      "model_name": model_name,
      "tokenizer_kwargs": {
        "additional_special_tokens": special_tokens
      }
    },
    "source_token_indexers": {
       "tokens": {
         "type": "pretrained_transformer",
         "model_name": model_name,
         "namespace": "tokens",
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
      "namespace": "tokens",
      "tokenizer_kwargs": {
        "additional_special_tokens": special_tokens
      }
    },
    "token_based_metric": {
      "type": "logical_form_em_for_LF_seq2seq",
      "formatter": data_format
    },
    [if use_config then "model_config"]:{
      "activation_dropout": 0.0,
      "activation_function": "gelu",
      "add_bias_logits": false,
      "add_final_layer_norm": false,
      "architectures": [
        "BartForConditionalGeneration"
      ],
      "attention_dropout": 0.0,
      "bos_token_id": 0,
      "classif_dropout": 0.0,
      "d_model": 768,
      "decoder_attention_heads": 12,
      "decoder_ffn_dim": 3072,
      "decoder_layerdrop": 0.0,
      "decoder_layers": 6,
      "decoder_start_token_id": 2,
      "dropout": 0.1,
      "encoder_attention_heads": 12,
      "encoder_ffn_dim": 3072,
      "encoder_layerdrop": 0.0,
      "encoder_layers": 6,
      "eos_token_id": 2,
//      "extra_pos_embeddings": 2,
//      "force_bos_token_to_be_generated": false,
      "id2label": {
        "0": "LABEL_0",
        "1": "LABEL_1",
        "2": "LABEL_2"
      },
      "init_std": 0.02,
      "is_encoder_decoder": true,
      "label2id": {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_2": 2
      },
      "max_position_embeddings": 1024,
      "model_type": "bart",
      "normalize_before": false,
      "normalize_embedding": true,
      "num_hidden_layers": 6,
      "pad_token_id": 1,
//      "save_step": 9,
      "scale_embedding": false,
//      "static_position_embeddings": false,
      "vocab_size": 50265
    }
  },

  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["source_tokens"],
      "batch_size": batch_size
     }
  },

  "trainer": {
    "num_epochs": 10,
    "patience": 10,
    "validation_metric": "+logical_form_em",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "correct_bias": true
    },
    "learning_rate_scheduler": learning_rate_scheduler,
//    "learning_rate_scheduler": {
//      "type": "linear_with_warmup",
//      "warmup_steps": 0,
//    },
    "grad_norm": 1.0,
  }
}