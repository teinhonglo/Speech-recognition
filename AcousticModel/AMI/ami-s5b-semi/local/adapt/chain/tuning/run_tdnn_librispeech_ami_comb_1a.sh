#!/bin/bash

# This script uses weight transfer as a transfer learning method to transfer
# already trained neural net model on Librispeech to ami.
#
#System tdnn_5n tdnn_librispeech_ami_1a
#WER      2.71     1.68
set -e

data_root=data/ihm/adapt
exp_root=exp/ihm/semisup_20k
gmm=tri3
tree_affix=_comb_1a
ivector_root_dir=exp/ihm/semisup_20k/nnet3_semi20k_80k
min_seg_len=1.5
# configs for 'chain'
stage=1
train_stage=-10
get_egs_stage=-10
xent_regularize=0.1

# Semi-supervised options
transfer_weights=1.5,1     # Weights for supervised, unsupervised data egs.
                              # Can be used to scale down the effect of unsupervised data
                              # by using a smaller scale for it e.g. 1.0,0.3
lm_weights=3,2  # Weights on phone counts from supervised, unsupervised data for denominator FST creation

nnet3_affix=_online_librispeech
chain_affix=_online_librispeech

final_lm=`cat data/local/lm/final_lm`
LM=$final_lm.pr1-7
nj=80
gmm=tri3  # Expect GMM model in $exp/$gmm for alignment
# End configuration section.
ami_egs_dir=
ls_egs_dir=
common_treedir=

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi


# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 4" if you have already
# run those things.
ami_lores_dir=train_ami_sup_sp
ami_hires_dir=${ami_lores_dir}_hires

ls_lores_dir=train_ls_sp
ls_hires_dir=${ls_lores_dir}_hires

gmm_dir=$exp_root/${gmm}
ali_dir=$exp_root/${gmm}_ali_train_sup_sp_comb

ami_ls_lores_dir=train_ami_ls_sp
ami_ls_hires_dir=${ami_ls_lores_dir}_hires
ami_ls_ivector_dir=$ivector_root_dir/ivectors_${ami_ls_hires_dir}_comb
ami_ls_lat_dir=$exp_root/chain${chain_affix}/${gmm}_${ami_ls_lores_dir}_comb_lats

tree_dir=$exp_root/chain${chain_affix}/tree_bi${tree_affix}
lang=data/lang_chain_ls2ami
dir=$exp_root/chain${chain_affix}/tdnn_librispeech_ami_comb_1a

if [ $stage -le 0 ]; then
  # making low-resolution MFCC
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
       $data_root/$ls_lores_dir
  steps/compute_cmvn_stats.sh $data_root/$ls_lores_dir
  utils/fix_data_dir.sh $data_root/$ls_lores_dir

  # making high-resolution MFCC  
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj \
     --mfcc-config conf/mfcc_hires.conf \
     $data_root/$ls_hires_dir
  steps/compute_cmvn_stats.sh $data_root/$ls_hires_dir
  utils/fix_data_dir.sh $data_root/$ls_hires_dir
fi

if [ $stage -le 1 ]; then
  utils/data/combine_data.sh $data_root/$ami_ls_lores_dir $data_root/$ls_lores_dir $data_root/$ami_lores_dir
  utils/fix_data_dir.sh $data_root/$ami_ls_lores_dir
  utils/data/combine_data.sh $data_root/$ami_ls_hires_dir $data_root/$ls_hires_dir $data_root/$ami_hires_dir
  utils/fix_data_dir.sh $data_root/$ami_ls_hires_dir
fi

if [ $stage -le 2 ]; then
  for data_dir in $ami_ls_lores_dir $ami_ls_hires_dir; do
    # we have to combine short segments or we won't be able to train chain models
    # on those segments.
    utils/data/combine_short_segments.sh \
    $data_root/$data_dir $min_seg_len $data_root/${data_dir}_comb

    # just copy over the CMVN to avoid having to recompute it.
    cp $data_root/$data_dir/cmvn.scp $data_root/${data_dir}_comb/
    utils/fix_data_dir.sh $data_root/${data_dir}_comb
  done
fi

if [ $stage -le 3 ]; then
  # extractor i-vector
  for data_dir in  $ami_ls_hires_dir; do
    temp_data_root=$ivector_root_dir/ivectors_${data_dir}_adapt
    utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
      $data_root/${data_dir}_comb $temp_data_root/${data_dir}_adapt_max2_comb

    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      $temp_data_root/${data_dir}_adapt_max2_comb $ivector_root_dir/extractor \
      $ivector_root_dir/ivectors_${data_dir}_comb || exit 1
   done
fi
								  
if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  # generating lattice for ami.
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" --generate_ali_from_lats true $data_root/${ami_ls_lores_dir}_comb \
    data/lang $gmm_dir $ami_ls_lat_dir || exit 1;
  rm $ami_ls_lat_dir/fsts.*.gz 2>/dev/null || true # save space
fi

if [ $stage -le 5 ]; then
  echo "$0: creating lang directory with one state per phone."
  
  rm -rf $lang
  cp -r data/lang $lang	
  #cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ -z "$common_treedir" ]; then
    if [ $stage -le 6 ]; then
    # Build a tree using our new topology.  We know we have alignments for the
    # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
    # those.
    if [ -f $tree_dir/final.mdl ]; then
	  echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
	  exit 1;
    fi
  
    steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
       --context-opts "--context-width=2 --central-position=1" \
       --leftmost-questions-truncate -1 \
       --cmd "$train_cmd" 4200 $data_root/$ami_ls_lores_dir $lang $ami_ls_lat_dir $tree_dir
   fi
else
  tree_dir=$common_treedir
fi
cmvn_opts="--norm-means=false --norm-vars=false"

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=450
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=450
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=450
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=450
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=450
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn7 dim=450 target-rms=0.5
  output-layer name=output input=prefinal-chain include-log-softmax=false dim=$num_targets max-change=1.5
  
  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn7 dim=450 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on ami and librispeech $data_root/$ami_ls_hires_dir set."
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$cuda_cmd" \
    --feat.online-ivector-dir "$ami_ls_ivector_dir" \
    --chain.xent-regularize $xent_regularize \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
    --egs.chunk-width 150 \
    --trainer.num-chunk-per-minibatch=128 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 4 \
    --trainer.optimization.num-jobs-initial 2 \
    --trainer.optimization.num-jobs-final 12 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir $data_root/$ami_ls_hires_dir \
    --tree-dir $tree_dir \
    --lat-dir $ami_ls_lat_dir \
    --dir $dir || exit 1;
fi

graph_dir=$dir/graph_${LM}
if [ $stage -le 8 ]; then
  # Note: it might appear that this data/lang_chain directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${LM} $dir $graph_dir
fi

if [ $stage -le 9 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in dev eval; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $nj --cmd "$decode_cmd" \
          --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
          --scoring-opts "--min-lmwt 5 " \
         $graph_dir data/ihm/semisup/${decode_set}_hires $dir/decode_${decode_set} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit 0;
