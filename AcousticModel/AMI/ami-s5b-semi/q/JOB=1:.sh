#!/bin/bash
cd /share/nas165/teinhonglo/AcousticModel/AMI/ami-s5b-semi
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascoring/log/get_ctm.10.JOB.log mkdir -p exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascore_10/ && lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/lat.JOB.gz|" ark:- | lattice-limit-depth ark:- ark:- | lattice-push --push-strings=false ark:- ark:- | lattice-align-words-lexicon --max-expand=10.0 data/lang_ami.o3g.kn.pr1-7/phones/align_lexicon.int exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/../final.mdl ark:- ark:- | lattice-to-ctm-conf --frame-shift=0.03 --decode-mbr=true ark:- - | utils/int2sym.pl -f 5 data/lang_ami.o3g.kn.pr1-7/words.txt | utils/convert_ctm.pl data/ihm/semisup/eval_hires/segments data/ihm/semisup/eval_hires/reco2file_and_channel > exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascore_10/eval_hires.JOB.ctm 
EOF
) >JOB=1:
time1=`date +"%s"`
 ( exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascoring/log/get_ctm.10.JOB.log mkdir -p exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascore_10/ && lattice-scale --inv-acoustic-scale=10 "ark:gunzip -c exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/lat.JOB.gz|" ark:- | lattice-limit-depth ark:- ark:- | lattice-push --push-strings=false ark:- ark:- | lattice-align-words-lexicon --max-expand=10.0 data/lang_ami.o3g.kn.pr1-7/phones/align_lexicon.int exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/../final.mdl ark:- ark:- | lattice-to-ctm-conf --frame-shift=0.03 --decode-mbr=true ark:- - | utils/int2sym.pl -f 5 data/lang_ami.o3g.kn.pr1-7/words.txt | utils/convert_ctm.pl data/ihm/semisup/eval_hires/segments data/ihm/semisup/eval_hires/reco2file_and_channel > exp/ihm/tdnn_combine/tdnn_reg/decode_eval_lats/ascore_10/eval_hires.JOB.ctm  ) 2>>JOB=1: >>JOB=1:
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>JOB=1:
echo '#' Finished at `date` with status $ret >>JOB=1:
[ $ret -eq 137 ] && exit 100;
touch ./q/sync/done.25563
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -l arch=*64* -o ./q/JOB=1:    /share/nas165/teinhonglo/AcousticModel/AMI/ami-s5b-semi/./q/JOB=1:.sh >>./q/JOB=1: 2>&1
