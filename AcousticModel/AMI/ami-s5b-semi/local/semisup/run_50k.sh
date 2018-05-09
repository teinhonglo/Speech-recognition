#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 50 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we assume there is data/train/text, from which
# we will exclude the utterances contained in the unsupervised set.
# We use all 300 hours of semi-supervised data for i-vector extractor training.

# This differs from run_100k.sh, which uses only 100 hours supervised data for 
# both i-vector extractor training and LM training.

. ./cmd.sh
. ./path.sh 

set -o pipefail


stage=0
nj=30 # number of parallel jobs,
mic=ihm
train_sup_dir=train_sup
train_unsup_dir=train_unsup80k
exp_root=exp/$mic/semisup_21k
LM=$final_lm.pr1-7

. utils/parse_options.sh

for f in data/$mic/$train_sup_dir/utt2spk data/$mic/$train_unsup_dir/utt2spk \
  data/train/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

###############################################################################
# Prepare the 50 hours supervised set and subsets for initial GMM training
###############################################################################

if [ $stage -le 0 ]; then
  utils/subset_data_dir.sh --shortest data/$mic/$train_sup_dir 15000 data/$mic/{$train_sup_dir}_15k_short || exit 1
  utils/subset_data_dir.sh --speakers data/$mic/$train_sup_dir 20000 data/$mic/{$train_sup_dir}_20k || exit 1
fi

###############################################################################
# GMM system training using 50 hours supervised data
###############################################################################

if [ $stage -le 1 ]; then
  steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/{$train_sup_dir}_15k_short data/lang $exp_root/mono0a || exit 1
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/$mic/{$train_sup_dir}_20k data/lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1	
fi

if [ $stage -le 2 ]; then
  steps/train_deltas.sh --cmd "$train_cmd" \
    5000 80000 data/$mic/{$train_sup_dir}_20k data/lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1
  
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
   data/$mic/{$train_sup_dir}_20k data/lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  graph_dir=$exp_root/tri1/graph_${LM} 
  
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} $exp_root/tri1 $graph_dir 
  # Decode	
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev $exp_root/tri1/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval $exp_root/tri1/decode_eval_${LM}	

fi

if [ $stage -le 3 ]; then
  # LDA_MLLT
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    5000 80000 data/$mic/$train_sup_dir data/lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;
  
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd --num_threads 3" \
    data/$mic/$train_sup_dir data/lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;
  
  graph_dir=$exp_root/tri2/graph_${LM}
  
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} $exp_root/tri2 $graph_dir 
  # Decode	
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev $exp_root/tri2/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval $exp_root/tri2/decode_eval_${LM}	
fi

if [ $stage -le 4 ]; then
  # LDA+MLLT+SAT
  steps/train_sat.sh --cmd "$train_cmd  --num_threads 5" \
    5000 80000 data/$mic/$train_sup_dir data/lang $exp_root/tri2_ali $exp_root/tri3 || exit 1;
	
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd --num_threads 3" \
    data/$mic/$train_sup_dir data/lang $exp_root/tri3 $exp_root/tri3_ali || exit 1;
  
  graph_dir=$exp_root/tri2/graph_${LM}
  
  $decode_cmd --mem 4G $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_${LM} $exp_root/tri3 $graph_dir 
  # Decode	
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/dev $exp_root/tri3/decode_dev_${LM}
  steps/decode.sh --nj $nj --cmd "$decode_cmd" --config conf/decode.conf \
    $graph_dir data/$mic/eval $exp_root/tri3/decode_eval_${LM}	
fi

###############################################################################
# Prepare semi-supervised train set 
###############################################################################

if [ $stage -le 5 ]; then
  utils/combine_data.sh data/semisup50k_21k_80k \
    data/$mic/$train_sup_dir data/$mic/$train_unsup_dir || exit 1
fi

###############################################################################
# Train LM on all the text in data/train/text, but excluding the 
# utterances in the unsupervised set
###############################################################################

if [ $stage -le 7 ]; then
  mkdir -p data/local/$mic/pocolm_ex250k

  utils/filter_scp.pl --exclude data/$mic/$train_unsup_dir/utt2spk \
    data/train/text > data/local/$mic/pocolm_ex250k/text.tmp

  if [ ! -f data/lang_test_poco_ex250k_big/G.carpa ]; then
    local/fisher_train_lms_pocolm.sh \
      --text data/local/$mic/pocolm_ex250k/text.tmp \
      --dir data/local/$mic/pocolm_ex250k

    local/fisher_create_test_lang.sh \
      --arpa-lm data/local/$mic/pocolm_ex250k/data/arpa/4gram_small.arpa.gz \
      --dir data/lang_test_poco_ex250k

    utils/build_const_arpa_lm.sh \
      data/local/$mic/pocolm_ex250k/data/arpa/4gram_big.arpa.gz \
      data/lang_test_poco_ex250k data/lang_test_poco_ex250k_big
  fi
fi

###############################################################################
# Prepare lang directories with UNK modeled using phone LM
###############################################################################

if [ $stage -le 8 ]; then
  local/run_unk_model.sh || exit 1

  for lang_dir in data/lang_test_poco_ex250k; do
    rm -r ${lang_dir}_unk ${lang_dir}_unk_big 2>/dev/null || true
    cp -rT data/lang_unk ${lang_dir}_unk
    cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
    cp -rT data/lang_unk ${lang_dir}_unk_big
    cp ${lang_dir}_big/G.carpa ${lang_dir}_unk_big/G.carpa; 
  done
fi

###############################################################################
# Train seed chain system using 50 hours supervised data.
# Here we train i-vector extractor on combined supervised and unsupervised data
###############################################################################

if [ $stage -le 9 ]; then
  local/semisup/chain/run_tdnn.sh \
    --mic $mic \
    --train-set $train_sup_dir \
    --ivector-train-set semisup21k_80k \
    --nnet3-affix _semi21k_80k \
    --chain-affix _semi21k_80k \
    --tdnn-affix _1a --tree-affix bi_a \
    --gmm tri4a --exp-root $exp_root || exit 1

  # WER on dev                21.41
  # WER on test               21.03
  # Final train prob          -0.1035
  # Final valid prob          -0.1667
  # Final train prob (xent)   -1.5926
  # Final valid prob (xent)   -1.7990
fi

###############################################################################
# Semi-supervised training using 50 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 10 ]; then
  local/semisup/chain/run_tdnn_50k_semisupervised.sh \
    --mic $mic \
    --supervised-set $train_sup_dir \
    --unsupervised-set $train_unsup_dir \
    --sup-chain-dir $exp_root/chain_semi50k_100k_250k/tdnn_1a_sp \
    --sup-lat-dir $exp_root/chain_semi50k_100k_250k/tri4a_train_sup50k_sp_unk_lats \
    --sup-tree-dir $exp_root/chain_semi50k_100k_250k/tree_bi_a \
    --ivector-root-dir $exp_root/nnet3_semi50k_100k_250k \
    --chain-affix _semi21k_80k \
    --exp-root $exp_root || exit 1

  # WER on dev                          18.98
  # WER on test                         18.85
  # Final output-0 train prob           -0.1381
  # Final output-0 valid prob           -0.1723
  # Final output-0 train prob (xent)    -1.3676
  # Final output-0 valid prob (xent)    -1.4589
  # Final output-1 train prob           -0.7671
  # Final output-1 valid prob           -0.7714
  # Final output-1 train prob (xent)    -1.1480
  # Final output-1 valid prob (xent)    -1.2382
fi

###############################################################################
# Oracle system trained on combined 300 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

if [ $stage -le 11 ]; then
  local/semisup/chain/run_tdnn.sh \
    --mic $mic \
    --train-set semisup50k_100k_250k \
    --nnet3-affix _semi50k_100k_250k \
    --chain-affix _semi50k_100k_250k \
    --common-treedir $exp_root/chain_semi50k_100k_250k/tree_bi_a \
    --tdnn-affix 1a_oracle --nj 100 \
    --gmm tri4a --exp-root $exp_root \
    --stage 9 || exit 1

  # WER on dev                          17.55
  # WER on test                         17.72
  # Final output train prob             -0.1155
  # Final output valid prob             -0.1510
  # Final output train prob (xent)      -1.7458
  # Final output valid prob (xent)      -1.9045
fi
