/*
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 nohup python scripts/train/run_experiments.py train --experiment experiments/hybrid/copynet--graph-embedder.jsonnet -s tmp/hybrid -o embedding_dim:300 > _logs/hybrid.txt &
*/

local embedding_dim = std.parseInt(std.extVar("embedding_dim")); //150
local logical_form_em = true;

local graph_pretrained_dir = std.extVar("graph_pretrained_dir");
local graph_pretrained_model = graph_pretrained_dir + "model.tar.gz";
local GRAPH_PARSER = import '../dependencies_graph/biaffine-graph-parser--transformer-encoder.jsonnet';
local use_ff_encodings = std.parseJson(std.extVar("use_ff_encodings")); //false;
local separate_head_and_child = std.parseJson(std.extVar("separate_head_and_child")); //false;
local append_weighted_neighbors = std.parseJson(std.extVar("append_weighted_neighbors")); //true;
local graph_freeze = std.parseJson(std.extVar("graph_freeze")); //true;

local graph_encoding_dim =
    if !use_ff_encodings then 768 + 100 // bert + POS
    else 500 + 100; // arc_representation_dim + tag_representation_dim
local arc_tags_embedding_dim = if append_weighted_neighbors then 100 else 0;
local graph_embedding_dim =
    if !append_weighted_neighbors then graph_encoding_dim
    else graph_encoding_dim + 2*(graph_encoding_dim + arc_tags_embedding_dim); // [e(u); e_in(u); e_out(u)]

{
  "train_data_path": "datasets/Break/QDMR/train_seq2seq.csv",
  "validation_data_path": "datasets/Break/QDMR/dev_seq2seq.csv",
//  "test_data_path": "datasets/Break/QDMR/test_seq2seq.csv",


  "dataset_reader":{
    "type":"break_copynet_seq2seq",
//    "cache_directory": "_cached_dataset_readers/break_copynet_seq2seq/graph_embedder",
//    "max_instances": 100,
    "target_namespace": "target_tokens",
    "source_tokenizer":{
      "type": "spacy",
      "pos_tags": true,
    },
    "source_token_indexers": {
      "tokens": {
        "type": "single_id",
        "namespace": "source_tokens"
      },
      "graph": {
        "type": "dependencies_graph",
        "pos_tags_namespace": "pos_tags",
        "indexers": GRAPH_PARSER.dataset_reader.token_indexers,
      },
    }
  },

  "vocabulary": {
//    "type": "from_files",
    "type": "extend",
    "directory": graph_pretrained_model,
  },

  "model": {
    "type": "custom_copynet_seq2seq",
    "target_namespace": "target_tokens",
    "source_embedder": {
      "token_embedders":{
        "tokens": {
          "type": "embedding",
          "embedding_dim": embedding_dim,
          "vocab_namespace": "source_tokens",
          "trainable": true,
        },
        "graph": {
          "type": "dependencies_graph_embedder",
          "append_weighted_neighbors": append_weighted_neighbors,
          "use_ff_encodings": use_ff_encodings,
          "separate_head_and_child": separate_head_and_child,
          "arc_tags_embedding_dim": arc_tags_embedding_dim,
          "model": {
            "_pretrained": {
                "archive_file": graph_pretrained_model,
                "module_path": "",
                "freeze": graph_freeze,
            }
          },
        },
      },
    },

    "encoder": {
      "type": "lstm",
      "input_size": embedding_dim + graph_embedding_dim,
      "hidden_size": 150,
      "num_layers": 5,
      "dropout": 0.2,
      "bidirectional": false
    },

    "max_decoding_steps": 200,
    "attention": "dot_product",
    "beam_size": 5,

    [if logical_form_em then "token_based_metric"]: {
       "type": "logical_form_em_for_seq2seq"
     },
  },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "sorting_keys": ["source_tokens"],
        "batch_size": 32
    }
  },

  "trainer": {
    "num_epochs": 25,
    "patience": 10,
    "cuda_device": 0,
    "optimizer": {
      "type": "adam",
      "lr": 0.01
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
    [if logical_form_em then "validation_metric"]: "+logical_form_em",
  }
}