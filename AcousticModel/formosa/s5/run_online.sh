#!/bin/bash
#
# Copyright 2018, Yuan-Fu Liao, National Taipei University of Technology, yfliao@mail.ntut.edu.tw
#
# Before you run this recips, please apply, download and put or make a link of the corpus under this folder (folder name: "NER-Trs-Vol1").
# For more detail, please check:
# 1. Formosa Speech in the Wild (FSW) project (https://sites.google.com/speech.ntut.edu.tw/fsw/home/corpus)
# 2. Formosa Speech Recognition Challenge (FSW) 2018 (https://sites.google.com/speech.ntut.edu.tw/fsw/home/challenge)
. ./cmd.sh
. ./path.sh

stage=-2
train_stage=-10
num_jobs=20
data_dir=/share/corpus/MATBN_GrandChallenge/NER-Trs-Vol1-Eval
data_root=data/online
exp_root=exp
dir=
graph_dir=
# shell options
set -euo pipefail

. ./utils/parse_options.sh


if [ -z $graph_dir ]; then
  graph_dir=$dir/graph
fi

# configure number of jobs running in parallel, you should adjust these numbers according to your machines
# data preparation
if [ $stage -le -2 ]; then
  # Data Preparation
  echo "$0: Data Preparation"
  local/prepare_data_online.sh $data_dir || exit 1;
fi

# Now make MFCC plus pitch features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.

# mfcc
if [ $stage -le -1 ]; then
  echo "$0: Making mfccs"
  for x in Eval; do
    steps/make_mfcc_pitch.sh --cmd "$train_cmd" --nj $num_jobs $data_root/$x || exit 1;
    steps/compute_cmvn_stats.sh $data_root/$x || exit 1;
    utils/fix_data_dir.sh $data_root/$x || exit 1;
  done
fi
<<WORD
if [ $stage -le 0 ]; then
  echo
  for datadir in Clean-test Other-test; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
        $data_root/${datadir}_hires $exp_root/$ivector_dir/extractor \
        $exp_root/$ivector_dir/ivectors_${datadir}_hires
  done
fi

if [ $stage -le 1]; then
  for decode_set in Clean-test Other-test; do
    (
    decode_dir=${dir}/decode_${decode_set}
    steps/nnet3/decode.sh --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir $exp_root/$ivector_dir/ivectors_${decode_set}_hires \
         $graph_dir $data_root/${decode_set}_hires $decode_dir
    ) &
  done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: error detected during decoding"
    exit 1
  fi
fi

WORD
echo "$0: all done"

exit 0;
