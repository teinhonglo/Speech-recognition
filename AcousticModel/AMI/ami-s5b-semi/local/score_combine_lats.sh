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
set -euo pipefail
# begin configuration section.
cmd=run.pl
beam=4 # prune the lattices prior to MBR decoding, for speed.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
stage=0
cer=0
decode_mbr=true
lat_weights=
word_ins_penalty=0.0
min_lmwt=5
max_lmwt=15
frame_shift_opts=0.03
parallel_opts="--num-threads 3"
skip_scoring=false
ctm_name=
overlap_spk=4
asclite=false
JOB_str=JOB
post_decode_acwt=1.0
minimize=false
lattice_beam=8.0 # Beam we use in lattice generation.
word_determinize=false  # If set to true, then output lattice does not retain
                        # alternate paths a sequence of words (with alternate pronunciations).
                        # Setting to true is the default in steps/nnet3/decode.sh.
                        # However, setting this to false
                        # is useful for generation w of semi-supervised training
                        # supervision and frame-level confidences.
write_compact=true   # If set to false, then writes the lattice in non-compact format,
                     # retaining the acoustic scores on each arc. This is
                     # required to be false for LM rescoring undeterminized
                     # lattices (when --word-determinize is false)

#end configuration section.

echo "$0 $@"

help_message="Usage: "$(basename $0)" [options] <data-dir> <graph-dir|lang-dir> <decode-dir1>[:lmwt-bias] <decode-dir2>[:lmwt-bias] [<decode-dir3>[:lmwt-bias] ... ] <out-dir>
     E.g. "$(basename $0)" data/test data/lang exp/tri1/decode exp/tri2/decode exp/tri3/decode exp/combine
     or:  "$(basename $0)" data/test data/lang exp/tri1/decode exp/tri2/decode:18 exp/tri3/decode:13 exp/combine
Options:
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --min-lmwt INT                  # minumum LM-weight for lattice rescoring 
  --max-lmwt INT                  # maximum LM-weight for lattice rescoring
  --lat-weights STR               # colon-separated string of lattice weights
  --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes.
  --stage (0|1|2)                 # (createCTM | filterCTM | runSclite).
  --parallel-opts <string>        # extra options to command for combination stage,
                                  # default '--num-threads 3'
  --cer (0|1)                     # compute CER in addition to WER
";

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;


if [ $# -lt 5 ]; then
  printf "$help_message\n";
  exit 1;
fi

data=$1
lang=$2
dir=${@: -1}  # last argument to the script
shift 2;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

#Let the user to set the CTM file name 
#use the data-dir name in case the user doesn't care
if [ -z ${ctm_name} ] ; then
  ctm_name=`basename $data`
fi

mkdir -p $dir/log

for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  decode_dir=`echo $decode_dir | cut -d: -f1`
  [ -z "$offset" ] && offset=0
  
  model=`dirname $decode_dir`/final.mdl  # model one level up from decode dir
  for f in $model $decode_dir/lat.1.gz ; do
    [ ! -f $f ] && echo "$0: expecting file $f to exist" && exit 1;
  done
  if [ $i -eq 0 ]; then
    nj=`cat $decode_dir/num_jobs` || exit 1;
  else
    if [ $nj != `cat $decode_dir/num_jobs` ]; then
      echo "$0: number of decoding jobs mismatches, $nj versus `cat $decode_dir/num_jobs`" 
      exit 1;
    fi
  fi
  lats[$i]="ark:gunzip -c $decode_dir/lat.JOB.gz|"
  echo $decode_dir
done
# assume the nnet trained by 
# the same GMM, frame_shift and frame subsampling factor
mkdir -p $dir/log

model_dir=`dirname $model`

if [ -f $model ]; then
  echo "$0: $model exists, copy model to $dir/../"
  cp $model $dir/../
fi

if [ -f $decode_dir/num_jobs ]; then
  echo "$0: $decode_dir/num_jobs exists, copy model to $dir/"
  cp $decode_dir/num_jobs $dir
fi

if [ -f $model_dir/frame_shift ]; then
  cp $model_dir/frame_shift $dir/../
  echo "$0: $model_dir/frame_shift exists, copy $model_dir/frame_shift to $dir/../"
elif [ -f $model_dir/frame_subsampling_factor ]; then
  cp $model_dir/frame_subsampling_factor $dir/../
  echo "$0: $model_dir/frame_subsampling_factor exists, copy $model_dir/frame_subsampling_factor to $dir/../"
fi

lat_wspecifier="ark:|"
if ! $write_compact; then
  extra_opts="--determinize-lattice=false"
  lat_wspecifier="ark:| lattice-determinize-phone-pruned --beam=$lattice_beam --acoustic-scale=$acwt --minimize=$minimize --word-determinize=$word_determinize --write-compact=false $model ark:- ark:- |"
fi

if [ "$post_decode_acwt" == 1.0 ]; then
  lat_wspecifier="$lat_wspecifier gzip -c >$dir/lat.JOB.gz"
else
  lat_wspecifier="$lat_wspecifier lattice-scale --acoustic-scale=$post_decode_acwt --write-compact=$write_compact ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
fi

# lattice weight
if [ -z "$lat_weights" ]; then
  lat_weights=1.0
  for i in `seq $[$num_sys-1]`; do lat_weights="$lat_weights:1.0"; done
fi

if [ $stage -le 0 ]; then  
  echo "$0, combine lattice (hypothesis combine)"
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode_combine.JOB.log \
    lattice-combine --acoustic-scale=$acwt --lat-weights=$lat_weights "${lats[@]}" "$lat_wspecifier" || exit 1;
	## lattice-determinize-pruned
fi

if ! $skip_scoring ; then
  if [ $stage -le 2 ]; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
	scoring_opts="--min-lmwt 5 "
    local/score.sh $scoring_opts --cmd "$cmd" $data $lang $dir
    echo "score confidence and timing with sclite"
  fi
fi


exit 0
