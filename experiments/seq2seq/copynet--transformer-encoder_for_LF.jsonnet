local data_format = std.extVar("data_format"); // "special_tokens" | "sectors_rep"
local logical_form_em = std.parseJson(std.extVar("logical_form_em")); // true | false


{
    "dataset_reader": {
//        "max_instances": 100,
        "type": "break_copynet_seq2seq",
        "source_token_indexers": {
            "tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "bert-base-uncased"
            }
        },
        "target_namespace": "target_tokens"
    },
    "model": {
        "type": "custom_copynet_seq2seq",
        "attention": "dot_product",
        "beam_size": 5,
        "encoder": {
            "type": "pass_through",
            "input_dim": 768
        },
        "max_decoding_steps": 200,
        "source_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "pretrained_transformer_mismatched",
                    "model_name": "bert-base-uncased"
                }
            }
        },
        "target_namespace": "target_tokens",
        [if logical_form_em then "token_based_metric"]: {
            "type": "logical_form_em_for_LF_seq2seq",
            "formatter": data_format
        }
    },
    "train_data_path": "datasets/Break/QDMR/train_lf_"+data_format+".csv",
    "validation_data_path": "datasets/Break/QDMR/dev_lf_"+data_format+".csv",
    "trainer": {
        "checkpointer": {
            "num_serialized_models_to_keep": 1
        },
        "cuda_device": 0,
        "grad_norm": 1,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.06
        },
        "num_epochs": 25,
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1e-3,
            "parameter_groups": [
                [
                    [
                        ".*transformer.*"
                    ],
                    {
                        "lr": 5e-05
                    }
                ]
            ],
            "weight_decay": 0.01
        },
        "patience": 10,
        [if logical_form_em then "validation_metric"]: "+logical_form_em",
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 32
        }
    },
    "numpy_seed": 1337,
    "pytorch_seed": 133,
    "random_seed": 13370
}