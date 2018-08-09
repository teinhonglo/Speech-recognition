#!/bin/bash

stage=2
nnet3_affix=_semi20k_80k
src_model_root=exp/ihm/semisup_20k/chain_semi20k_80k
exp_root=exp/ihm/semisup_20k
# score_comb 
# tdnn_hyp
# decode_{NULL, semisup_best_phn, semisup, oracle}_{eval, dev}_{NULL, lats}
# tdnn_fusion
# decode_{NULL, semisup_best_phn, semisup, oracle}_{eval, dev}
. ./cmd.sh
. ./path.sh 

set -euo pipefail
cmd=run.pl

. parse_options.sh || exit 1;

ivector_root_dir=$exp_root/nnet3${nnet3_affix}
decode_affix=_poco
dest_decode_affix=_oracle

if [ $stage -le 0 ]; then
  echo "$0, Hypothesis Combine without generate lattice"
  for decode_set in dev eval; do
<<WORD
    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi/decode${decode_affix}_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_${decode_set}

    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c/decode${decode_affix}_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d/decode${decode_affix}_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_semisup_${decode_set}

    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/decode${decode_affix}_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a_best_phn/decode${decode_affix}_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c_best_phn/decode${decode_affix}_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn/decode${decode_affix}_best_phn_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_semisup_phn_${decode_set}

    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1b_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1c_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1d_oracle_sp_bi/decode_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_oracle_${decode_set}
WORD
   local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn_nonpost/decode_poco_best_phn_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp_stage/decode_${decode_set}_nopost
						   
   local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi/decode_${decode_set}:1 \
						   $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn_nonpost/decode_poco_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/decode_poco_best_phn_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp_stage/decode_${decode_set}_best_phn
   
    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi/decode_${decode_set}:1 \
						   $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn_nonpost/decode_poco_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/decode_poco_best_phn_${decode_set}:1 \
						   $src_model_root/tdnn_1a_sp_bi_semisup_1b/decode_poco_${decode_set}:1
                           $exp_root/ensemble/tdnn_hyp_stage/decode_${decode_set}_semi					   
  done
fi

if [ $stage -le 1 ]; then
  echo "$0, Hypothesis Combine and generate lattice"
  for decode_set in dev eval; do
    local/score_combine_lats.sh --cmd "$decode_cmd" --stage 0 \
						   data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi/graph_poco \
                           $src_model_root/tdnn_1a_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi/decode_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_${decode_set}_lats

    local/score_combine_lats.sh --cmd "$decode_cmd" --stage 0 \
	                       data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b/graph_poco \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b/decode_poco_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a/decode_poco_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c/decode_poco_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d/decode_poco_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_semisup_${decode_set}_lats

    local/score_combine_lats.sh --cmd "$decode_cmd" --stage 0 \
	                       data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/graph_poco_best_phn \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/decode_poco_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a_best_phn/decode_poco_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c_best_phn/decode_poco_best_phn_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn/decode_poco_best_phn_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_semisup_phn_${decode_set}_lats

    local/score_combine_lats.sh --cmd "$decode_cmd" --stage 0 \
	                       data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1a_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1b_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1c_oracle_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1d_oracle_sp_bi/decode_${decode_set}:1 \
                           $exp_root/ensemble/tdnn_hyp/decode_oracle_${decode_set}_lats
<<WORD
    local/score_combine_lats.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           $src_model_root/tdnn_1a_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1b_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1c_sp_bi/decode_${decode_set}:1 \
                           $src_model_root/tdnn_1d_sp_bi/decode_${decode_set}:1 \
                           $exp_root/tdnn_hyp/decode${decode_affix}_${decode_set}_lats
WORD
  done
fi

if [ $stage -le 2 ]; then
  echo "$0, Frame Level Combine and generate lattice"
  for decode_set in dev eval; do
<<WORD
    local/score_fusion.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
	                       --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
						   --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi/graph_poco \
                           $src_model_root/tdnn_1a_sp_bi \
                           $src_model_root/tdnn_1b_sp_bi \
                           $src_model_root/tdnn_1c_sp_bi \
                           $src_model_root/tdnn_1d_sp_bi \
                           $exp_root/ensemble/tdnn_fusion/decode_${decode_set}

    local/score_fusion.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
	                       --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
						   --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b/graph_poco \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d \
                           $exp_root/ensemble/tdnn_fusion/decode_semisup_${decode_set}

    local/score_fusion.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
	                       --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
						   --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn/graph_poco_best_phn \
                           $src_model_root/tdnn_1a_sp_bi_semisup_1b_best_phn \
                           $src_model_root/tdnn_1b_sp_bi_semisup_1a_best_phn \
                           $src_model_root/tdnn_1c_sp_bi_semisup_1c_best_phn \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn \
                           $exp_root/ensemble/tdnn_fusion/decode_semisup_phn_${decode_set}

    local/score_fusion.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
	                       --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
						   --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1a_oracle_sp_bi \
                           $src_model_root/tdnn_1b_oracle_sp_bi \
                           $src_model_root/tdnn_1c_oracle_sp_bi \
                           $src_model_root/tdnn_1d_oracle_sp_bi \
                           $exp_root/ensemble/tdnn_fusion/decode_oracle_${decode_set}

    local/score_fusion.sh --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
	                       --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
						   --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1a_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1a_oracle_sp_bi \
                           $src_model_root/tdnn_1b_oracle_sp_bi \
                           $src_model_root/tdnn_1c_oracle_sp_bi \
                           $src_model_root/tdnn_1d_oracle_sp_bi \
                           $exp_root/tdnn_fusion/decode_${decode_set}_lats
WORD

   local/score_fusion.sh   --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
                           --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
                           --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1d_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1d_sp_bi \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn_nonpost \
                           $exp_root/ensemble/tdnn_fusion_stage/decode_1d_${decode_set}_nopost
						   
   local/score_fusion.sh   --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
                           --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
                           --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1d_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1d_sp_bi \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn_nonpost \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn \
                           $exp_root/ensemble/tdnn_fusion_stage/decode_1d_${decode_set}_best_phn
   
    local/score_fusion.sh  --cmd "$decode_cmd" --stage 0 --acwt 1.0 --post-decode-acwt 10.0 \
                           --online-ivector-dir $ivector_root_dir/ivectors_${decode_set}_hires \
                           --num-threads 5 --frame-subsampling-factor 3 \
                           data/ihm/semisup/${decode_set}_hires \
                           $src_model_root/tdnn_1d_oracle_sp_bi/graph_poco \
                           $src_model_root/tdnn_1d_sp_bi \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn_nonpost \
                           $src_model_root/tdnn_1d_sp_bi_semisup_1d_best_phn \
			   $src_model_root/tdnn_1d_sp_bi_semisup_1d \
                           $exp_root/ensemble/tdnn_fusion_stage/decode_1d_${decode_set}_semi
  done
fi  

exit 0;
