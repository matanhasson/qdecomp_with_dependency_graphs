local BASE = import 'copynet--bert-encoder.jsonnet';

BASE+{
  "dataset_reader"+:{
    "type":"break_copynet_seq2seq",
    "dynamic_vocab": true
  },

  "model"+: {
    "type": "copynet_seq2seq_dynamic",
    "tie_weights": false,
   },
}