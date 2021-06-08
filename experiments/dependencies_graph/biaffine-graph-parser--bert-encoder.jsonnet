local BASE = import 'biaffine-graph-parser.jsonnet';

local transformer_model = "bert-base-uncased";
local max_length = 128;
local transformer_dim = 768;

BASE+{
//  "dataset_reader"+: {
//     "token_indexers": {
//      "tokens": {
//        "type": "pretrained_transformer_mismatched",
//        "model_name": transformer_model,
//        "max_length": max_length
//      },
//    },
//  },
  "model"+: {
     "text_field_embedder": {
        "token_embedders": {
            "tokens": {
//                "type": "pretrained_transformer_mismatched",
//                "type": "pretrained_transformer_mismatched_embedder",
//                "model_name": transformer_model,
//                "max_length": max_length,
//                "train_parameters": false

                "type":"aggregated_pretrained_transformer",
                "model_name": transformer_model,
                "train_parameters": true,
                "vocab_namespace": "tokens",
            }
        }
     },
     "pos_tag_embedding"+:{
       "sparse": false  # huggingface_adamw cannot work with sparse
     },
//    "pos_tag_embedding": null,
     "encoder": {
      "type": "pass_through",
      "input_dim": transformer_dim + $.model.pos_tag_embedding.embedding_dim
     },
   },
   "data_loader": {
      "batch_sampler": {
        "type": "bucket",
         "batch_size": 32
     }
  },
   "trainer"+: {
       "grad_norm": 1.0,
       "optimizer": {
          "type": "huggingface_adamw",
          "lr": 1e-3,
          "weight_decay": 0.01,
          "parameter_groups": [
            [[".*transformer.*"], {"lr": 1e-5}]
          ]
        }
   }
}