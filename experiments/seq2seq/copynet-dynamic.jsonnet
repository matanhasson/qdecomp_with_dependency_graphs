local BASE = import 'copynet.jsonnet';
local logical_form_em = true;

BASE+{
  "dataset_reader"+:{
    "type":"break_copynet_seq2seq",
    "dynamic_vocab": true
  },

  "model"+: {
    "type": "copynet_seq2seq_dynamic",
    "tie_weights": true,
    [if logical_form_em then "token_based_metric"]: {
       "type": "logical_form_em_for_seq2seq"
     },
    },

    "trainer"+:{
      [if logical_form_em then "validation_metric"]: "+logical_form_em"
    }
}