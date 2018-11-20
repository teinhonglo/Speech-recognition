#!/bin/bash
# This script extracts mfcc features using mfcc_config and trains ubm model and
# ivector extractor and extracts ivector for train and test.
. ./cmd.sh


stage=1
nnet3_affix=_online
extractor=exp/nnet3${nnet_affix}/extractor
ivector_dim=50
nj=80
mfcc_config=conf/mfcc_hires.conf
data_root=data
exp_root=exp
train_set=train_sup
use_ivector=true # If false, it skips training ivector extractor and
                 # ivector extraction stages.
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=8
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

if [ $stage -le 1 ] && [ $ivector_dim -gt 0 ]; then
  ivectordir=$exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
  temp_data_root=${ivectordir}
  
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $data_root/${train_set}_sp_hires_comb ${temp_data_root}/${train_set}_sp_hires_comb_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_comb_max2 $extractor \
    $exp_root/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb || exit 1;  
  
  for datadir in dev eval; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
        $data_root/${datadir}_hires $extractor \
        $exp_root/nnet3${nnet3_affix}/ivectors_${datadir}_hires  
  done	
fi	