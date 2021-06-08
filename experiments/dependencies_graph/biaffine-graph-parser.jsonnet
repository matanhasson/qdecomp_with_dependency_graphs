local arc_tags_only = std.parseJson(std.extVar("arc_tags_only"));
local multi_label = std.parseJson(std.extVar("multi_label"));
local logical_form_em = true;

//hyperparameters
local pretrained_file = "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz";
local embedding_dim = 100;
local pos_embedding_dim = 100;
local input_dropout = 0.3;
local dropout = 0.3;
local hidden_size = 400;
local num_layers = 3;
local arc_representation_dim = 500;
local tag_representation_dim = 100;


{
  "train_data_path": "datasets/Break/QDMR/train_dependencies_graph.json",
  "validation_data_path": "datasets/Break/QDMR/dev_dependencies_graph.json",

  "dataset_reader":{
    "type": "dependencies_graph",
    "fill_none_tags": arc_tags_only && !multi_label,
    "multi_label": multi_label,
    "deps_tags_namespace": "labels",
    "word_field": "text",
    "pos_field": "tag",
    "pos_tags_namespace": "pos_tags",
//    "bio_field": "bio",
//   "bio_tags_namespace": "bio_tags",
  },
  "model": {
     "type": "biaffine_graph_parser",
     "arc_tags_only": arc_tags_only,
     "multi_label": multi_label,
     [if logical_form_em then "graph_based_metric"]: {
       "type": "logical_form_em_for_graph"
     },
     "text_field_embedder": {
       "token_embedders": {
         "tokens": {
           "type": "embedding",
           "embedding_dim": embedding_dim,
           "pretrained_file": pretrained_file,
           "trainable": true,
           "sparse": true
         }
       }
     },
     "pos_tag_embedding":{
       "embedding_dim": pos_embedding_dim,
       "vocab_namespace": "pos_tags",
       "sparse": true
     },
     "encoder": {
       "type": "stacked_bidirectional_lstm",
       "input_size": embedding_dim + pos_embedding_dim,
       "hidden_size": hidden_size,
       "num_layers": num_layers,
       "recurrent_dropout_probability": 0.3,
       "use_highway": true
     },
     "arc_representation_dim": arc_representation_dim,
     "tag_representation_dim": tag_representation_dim,
     "dropout": dropout,
     "input_dropout": input_dropout,
     "initializer": {
       "regexes": [
         [".*feedforward.*weight", {"type": "xavier_uniform"}],
         [".*feedforward.*bias", {"type": "zero"}],
         [".*tag_bilinear.*weight", {"type": "xavier_uniform"}],
         [".*tag_bilinear.*bias", {"type": "zero"}],
         ["encoder.*input.*weight", {"type": "xavier_uniform"}],
         ["encoder.*state.*weight", {"type": "orthogonal"}],
         ["encoder.*input.*bias", {"type": "zero"}],
         ["encoder.*state.*bias", {"type": "lstm_hidden_bias"}]]
       }
   },

  "data_loader": {
      "batch_sampler": {
        "type": "bucket",
         "batch_size": 128
     }
  },

  "trainer": {
    "num_epochs": 80,
    "grad_norm": 5.0,
    "patience": 10,
    "cuda_device": 0,
    "validation_metric": if logical_form_em then "+logical_form_em" else (if multi_label then "+arcs_and_tags_f1" else "+arcs_and_tags_micro_fscore"),
    "optimizer": {
      "type": "dense_sparse_adam",
      "betas": [0.9, 0.9]
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 1
    },
  }
}