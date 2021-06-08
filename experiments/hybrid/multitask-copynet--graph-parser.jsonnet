local SEQ2SEQ = import '../seq2seq/copynet--transformer-encoder.jsonnet';
local GRAPH_PARSER = import '../dependencies_graph/biaffine-graph-parser--transformer-encoder.jsonnet';

local batch_size = 64;

{
  "train_data_path": {
    "seq2seq": SEQ2SEQ.train_data_path,
    "graph_parser": GRAPH_PARSER.train_data_path
  },
  "validation_data_path": {
    "seq2seq": SEQ2SEQ.validation_data_path,
    "graph_parser": GRAPH_PARSER.validation_data_path
  },


  "dataset_reader":{
    "type": "custom_multitask",
    "readers": {
        "seq2seq": SEQ2SEQ.dataset_reader,
        "graph_parser": GRAPH_PARSER.dataset_reader,
    },
  },

  "model":{
    "type": "custom_multitask",
    "models":{
        "seq2seq": SEQ2SEQ.model,
        "graph_parser": GRAPH_PARSER.model,
    },
    "tie_modules":[
        ["seq2seq._source_embedder", "graph_parser.text_field_embedder"],
        ["seq2seq._encoder", "graph_parser.encoder"],
    ]
  },

  "data_loader": {
      "type": "multitask",
      "scheduler":{
        "batch_size": 32,
      },
  },

"trainer": {
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0,
    "grad_norm": 1.0,
    "validation_metric": "-seq2seq-loss",
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 1e-3,
      "weight_decay": 0.01,
      "parameter_groups": [
        [[".*transformer.*"], {"lr": 1e-5}]
      ]
    },
    "checkpointer": {
      "num_serialized_models_to_keep": 0
    },
  }
}