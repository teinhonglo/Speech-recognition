#!/bin/bash
cd /share/nas165/teinhonglo/ami-s5b
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/train.0.2.log nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=1.41421356237 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --l2-regularize-factor=0.5 --srand=0 "nnet3-am-copy --raw=true --learning-rate=0.002 --scale=1.0 exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/cegs.2.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=0 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=64 ark:- ark:- |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/1.2.raw 
EOF
) >1
time1=`date +"%s"`
 ( exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/train.0.2.log nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=1.41421356237 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --l2-regularize-factor=0.5 --srand=0 "nnet3-am-copy --raw=true --learning-rate=0.002 --scale=1.0 exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=2                         ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/cegs.2.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=0 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=64 ark:- ark:- |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/1.2.raw  ) 2>>1 >>1
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=--gpu >>1
echo '#' Finished at `date` with status $ret >>1
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.18964
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/1 -P inf_hcrc_cstr_nst -l h_rt=05:00:00 -pe 4 --gpu     /share/nas165/teinhonglo/ami-s5b/./q/1.sh >>./q/1 2>&1
