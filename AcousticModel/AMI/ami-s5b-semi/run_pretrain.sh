#!/bin/bash
. ./cmd.sh
. ./path.sh

nj=80
dir=../../pretrain_model/exp/chain/tdnn_7b
data_root=data/ihm/semisup
exp_root=exp/ihm/semisup_20k
nnet3_affix=_semi20k_80k
graph_dir=$dir/graph_poco
stage=1



if [ $stage -le 1 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  # diff.
  # utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco_unk $dir $graph_dir
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test_poco $dir $graph_dir
fi

if [ $stage -le 2 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd --num_threads 4" \
          --online-ivector-dir $exp_root/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir $data_root/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
exit 0
