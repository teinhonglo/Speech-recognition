#!/bin/bash
cd /share/nas165/teinhonglo/ami-s5b
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
all.q@nv980 --gpu 1 exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/train.0.1.log nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --write-cache=exp/ihm/chain_cleaned/tdnn1e_sp_bi/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=1.41421356237 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --l2-regularize-factor=0.5 --srand=0 "nnet3-am-copy --raw=true --learning-rate=0.002 --scale=1.0 exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=1                         ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/cegs.1.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=0 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=64 ark:- ark:- |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/1.1.raw 
EOF
) >all.q@titan,
time1=`date +"%s"`
 ( all.q@nv980 --gpu 1 exp/ihm/chain_cleaned/tdnn1e_sp_bi/log/train.0.1.log nnet3-chain-train --apply-deriv-weights=False --l2-regularize=5e-05 --leaky-hmm-coefficient=0.1 --write-cache=exp/ihm/chain_cleaned/tdnn1e_sp_bi/cache.1 --xent-regularize=0.1 --print-interval=10 --momentum=0.0 --max-param-change=1.41421356237 --backstitch-training-scale=0.0 --backstitch-training-interval=1 --l2-regularize-factor=0.5 --srand=0 "nnet3-am-copy --raw=true --learning-rate=0.002 --scale=1.0 exp/ihm/chain_cleaned/tdnn1e_sp_bi/0.mdl - |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/den.fst "ark,bg:nnet3-chain-copy-egs                         --frame-shift=1                         ark:exp/ihm/chain_cleaned/tdnn1e_sp_bi/egs/cegs.1.ark ark:- |                         nnet3-chain-shuffle-egs --buffer-size=5000                         --srand=0 ark:- ark:- | nnet3-chain-merge-egs                         --minibatch-size=64 ark:- ark:- |" exp/ihm/chain_cleaned/tdnn1e_sp_bi/1.1.raw  ) 2>>all.q@titan, >>all.q@titan,
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>all.q@titan,
echo '#' Finished at `date` with status $ret >>all.q@titan,
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.19193
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/all.q@titan, -q all.q@nv1080,     /share/nas165/teinhonglo/ami-s5b/./q/all.q@titan,.sh >>./q/all.q@titan, 2>&1
