local BASE = import 'multitask--copynet--latent-rat-encoder.jsonnet';

BASE+{
  "train_data_path"+: {
    "graph_parser": "datasets/dependencies_graphs/2021-05-03_no_special/train_dependencies_graph.json"
  },
  "validation_data_path"+: {
    "graph_parser": "datasets/dependencies_graphs/2021-05-03_no_special/dev_dependencies_graph__questions_only.json"
  },
  "test_data_path"+: {
    "graph_parser": "datasets/dependencies_graphs/2021-05-03_no_special/test_dependencies_graph__questions_only.json"
  },
  "model"+: {
    "graph_based_metric"::null
  }
}
