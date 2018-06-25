#!/bin/bash

# Copyright 2012-2013  Arnab Ghoshal
#                      Johns Hopkins University (authors: Daniel Povey, Sanjeev Khudanpur)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.


# Script for system combination using minimum Bayes risk decoding.
# This calls lattice-combine to create a union of lattices that have been 
# normalized by removing the total forward cost from them. The resulting lattice
# is used as input to lattice-mbr-decode. This should not be put in steps/ or 
# utils/ since the scores on the combined lattice must not be scaled.

# begin configuration section.
cmd=run.pl
beam=15.0 # prune the lattices prior to MBR decoding, for speed.
stage=0
cer=0
decode_mbr=true
lat_weights=
word_ins_penalty=0.0
min_lmwt=5
max_lmwt=15
parallel_opts="--num-threads 3"
skip_scoring=false
ctm_name=
average=true
iter=final
nj=80

acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
post_decode_acwt=1.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                      # regular scoring script works.
frames_per_chunk=50
max_active=7000
min_active=200
ivector_scale=1.0
lattice_beam=8.0 # Beam we use in lattice generation.
num_threads=1 # if >1, will use latgen-faster--map-parallel
scoring_opts=
skip_diagnostics=false
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
minimize=false
#end configuration section.

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 5 ]; then
  echo "Usage: $0 [options] <data-dir> <graph-dir> <nnet3-dir> <nnet3-dir2> [<nnet3-dir3> ... ] <output-dir>"
  echo "e.g.:   steps/nnet3/compute_output.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet3/ivectors_test_eval92 \\"
  echo "    data/test_eval92_hires exp/nnet3/tdnn exp/nnet3/tdnn/output"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  exit 1;
fi

data=$1
graphdir=$2
dir=${@: -1}  # last argument to the script
shift 2;
model_dirs=( $@ )  # read the remaining arguments into an array
unset model_dirs[${#model_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#model_dirs[@]}  # number of systems to combine

#Let the user to set the CTM file name 
#use the data-dir name in case the user doesn't care
if [ -z ${ctm_name} ] ; then
  ctm_name=`basename $data`
fi

for f in $lang/words.txt $lang/phones/word_boundary.int ; do
  [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
done

mkdir -p $dir/log

[ ! -z "$online_ivector_dir" ] && \
   extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"

# Possibly use multi-threaded decoder
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"   
   
if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi


# convert $dir to absolute pathname
fdir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

for i in `seq 0 $[num_sys-1]`; do
  model=${model_dirs[$i]}/$iter.raw
  if [ ! -f ${model_dirs[$i]}/$iter.raw ]; then
    echo "$0: WARNING: no such file $srcdir/$iter.raw. Trying $srcdir/$iter.mdl instead." && exit 1
    model=${model_dirs[$i]}/$iter.mdl
  fi

  for f in $data/feats.scp $model $extra_files; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done

  if [ ! -z "$output_name" ] && [ "$output_name" != "output" ]; then
    echo "$0: Using output-name $output_name"
    model="nnet3-copy --edits='remove-output-nodes name=output;rename-node old-name=$output_name new-name=output' $model - |"
  fi

  sdata=$data/split$nj;
  cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;

  [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
  echo $nj > $dir/num_jobs

  ## Set up features.
  if [ -f $srcdir/final.mat ]; then
    echo "$0: ERROR: lda feature type is no longer supported." && exit 1
  fi
  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"

  if [ ! -z "$online_ivector_dir" ]; then
    ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
  fi

  frame_subsampling_opt=
  if [ $frame_subsampling_factor -ne 1 ]; then
    # e.g. for 'chain' systems
    frame_subsampling_opt="--frame-subsampling-factor=$frame_subsampling_factor"
  fi

  if $apply_exp; then
    output_wspecifier="ark:| copy-matrix --apply-exp ark:- ark:-"
  else
    output_wspecifier="ark:| copy-feats --compress=$compress ark:- ark:-"
  fi

  gpu_opt="--use-gpu=no"
  gpu_queue_opt=

  if $use_gpu; then
    gpu_queue_opt="--gpu 1"
    gpu_opt="--use-gpu=yes"
  fi
  model_dir=${model_dirs[$i]}
  offset=`echo $model_dir | cut -d: -s -f2` # add this to the lm-weight.
  model_dir=`echo $model_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=0
  
  model=`$model_dir`/final.mdl  # model one level up from decode dir
  for f in $model $model_dir/lat.1.gz ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done

  models[$i]="ark,s,cs: nnet3-compute $gpu_opt $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     $model $feats $output_wspecifier |"
done

mkdir -p $dir/log

if [ $stage -le 0 ]; then  
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log  \
    matrix-sum --average=$average "${models[@]}" ark:- \| ark:- | \
	latgen-faster-mapped$thread_string -lattice-beam=$lattice_beam --acoustic-scale=$acwt --allow-partial=true \
	 --minimize=$minimize --max-active=$max_active --min-active=$min_active --beam=$beam\
     --word-symbol-table=$graphdir/words.txt "$model" \
     $graphdir/HCLG.fst "ark:-" "ark:|gzip -c > $dir/lat.JOB.gz"
	 
	nnet3-latgen-faster 
	  --minimize=false --max-active=7000 --min-active=200 --beam=15.0 --lattice-beam=8.0 --acoustic-scale=1.0 --allow-partial=true --word-symbol-table=exp/ihm/chain_cleaned/tdnn1e_sp_bi/graph_ami.o3g.kn.pr1-7/words.txt exp/ihm/chain_cleaned/tdnn1e_sp_bi/final.mdl exp/ihm/chain_cleaned/tdnn1e_sp_bi/graph_ami.o3g.kn.pr1-7/HCLG.fst 'ark,s,cs:apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/ihm/eval_hires/split30/18/utt2spk scp:data/ihm/eval_hires/split30/18/cmvn.scp scp:data/ihm/eval_hires/split30/18/feats.scp ark:- |' 'ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >exp/ihm/chain_cleaned/tdnn1e_sp_bi/decode_eval/lat.18.gz'  
fi


if ! $skip_scoring ; then
  if [ $stage -le 2 ]; then
    [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
	scoring_opts="--min_lmwt $min_lmwt"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi


exit 0
