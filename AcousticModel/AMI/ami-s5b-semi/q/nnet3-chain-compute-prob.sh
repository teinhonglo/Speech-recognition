#!/bin/bash
cd /share/nas165/teinhonglo/ami-s5b
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
--l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 "nnet3-am-copy --raw=true exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/train_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |" 
EOF
) >nnet3-chain-compute-prob
time1=`date +"%s"`
 ( --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 "nnet3-am-copy --raw=true exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/train_diagnostic.cegs                     ark:- | nnet3-chain-merge-egs --minibatch-size=1:64 ark:- ark:- |"  ) 2>>nnet3-chain-compute-prob >>nnet3-chain-compute-prob
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/compute_prob_train.0.log >>nnet3-chain-compute-prob
echo '#' Finished at `date` with status $ret >>nnet3-chain-compute-prob
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.18960
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/nnet3-chain-compute-prob -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe 4 exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/compute_prob_train.0.log     /share/nas165/teinhonglo/ami-s5b/./q/nnet3-chain-compute-prob.sh >>./q/nnet3-chain-compute-prob 2>&1
