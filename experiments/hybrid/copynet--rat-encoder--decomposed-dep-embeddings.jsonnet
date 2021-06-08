/*
nohup python scripts/train/run_experiments.py train --experiment experiments/hybrid/copynet--rat-encoder--decomposed-dep-embeddings.jsonnet -s tmp/rat-decomp -o seed:13370
*/

local BASE = import 'copynet--rat-encoder.jsonnet';
local embedding_dim = 768;

local dependencies_namespace = "relations_tags";

BASE+{
  "dataset_reader"+:{
    "decompose_dependencies": true,
    "dependencies_namespace": dependencies_namespace
  },

  "model"+:{
    "decomposed_dependencies": true,
    "encoder"+: {
      "relation_k_embedder": {
        "type": "fields_list",
        "embedding_dim": $.model.encoder.hidden_size/ $.model.encoder.num_heads,
        "namespaces": [
          "operator_" + dependencies_namespace,
          "arg_" + dependencies_namespace,
          "properties_" + dependencies_namespace,
        ],
        "concat_parts": false,
      },
      "relation_v_embedder": $.model.encoder.relation_k_embedder,
    },
  }
}