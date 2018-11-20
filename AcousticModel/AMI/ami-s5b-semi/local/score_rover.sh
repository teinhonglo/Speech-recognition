#!/bin/bash

# Copyright 2012-2013  Arnab Ghoshal
#                      Johns Hopkins University (authors: Daniel Povey, Sanjeev Khudanpur)
#                      Tien-Hong Lo

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
#end configuration section.

help_message="Usage: "$(basename $0)" [options] <data-dir> <decode-dir1>[:lmwt-bias] <decode-dir2>[:lmwt-bias] [<decode-dir3>[:lmwt-bias] ... ] <out-dir>
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

if [ $# -lt 4 ]; then
  printf "$help_message\n";
  exit 1;
fi

echo "$0 $@"
data=$1
dir=${@: -1}  # last argument to the script
shift 1;
decode_dirs=( $@ )  # read the remaining arguments into an array
unset decode_dirs[${#decode_dirs[@]}-1]  # 'pop' the last argument which is odir
num_sys=${#decode_dirs[@]}  # number of systems to combine

#Let the user to set the CTM file name 
#use the data-dir name in case the user doesn't care
if [ -z ${ctm_name} ] ; then
  ctm_name=`basename $data`
fi

hubscr=$KALDI_ROOT/tools/sctk/bin/hubscr.pl
[ ! -f $hubscr ] && echo "Cannot find scoring program at $hubscr" && exit 1;
hubdir=`dirname $hubscr`

if ! $skip_scoring ; then
  for f in  $data/stm; do
    [ ! -f $f ] && echo "$0: file $f does not exist" && exit 1;
  done
fi


mkdir -p $dir/log

for i in `seq 0 $[num_sys-1]`; do
  decode_dir=${decode_dirs[$i]}
  offset=`echo $decode_dir | cut -d: -s -f2` # add this to the lm-weight.
  decode_dir=`echo $decode_dir | cut -d: -f1`
  ctm_file=$decode_dir/ascore_${offset}/${ctm_name}.ctm
  ctms[$i]="-h $ctm_file ctm "
done

mkdir -p $dir/ascoring/log
mkdir -p $dir/ascore_rover
if [ $stage -le 0 ]; then  
  echo "$0 ROVER."
  rover=$KALDI_ROOT/tools/sctk-2.4.10/src/sclite/rover
  $rover ${ctms[@]} -o $dir/ascore_rover/rover.ctm -m oracle
fi


if [ $stage -le 1 ]; then
# Remove some stuff we don't want to score, from the ctm.
# - we remove hesitations here, otherwise the CTM would have a bug!
#   (confidences in place of the removed hesitations),
  x=$dir/ascore_rover/rover.ctm
  [ ! -f $x ] && echo "File $x does not exist! Exiting... " && exit 1
  cp $x $x.bkup1;
  cat $x.bkup1 | grep -i -v -E '\[noise|laughter|vocalized-noise\]' | \
    grep -i -v -E ' (ACH|AH|EEE|EH|ER|EW|HA|HEE|HM|HMM|HUH|MM|OOF|UH|UM) ' | \
    grep -i -v -E '<unk>' > $x;
  cp $x $x.bkup2;
fi

if ! $skip_scoring ; then
  if [ $stage -le 2 ]; then
    if [ "$asclite" == "true" ]; then
      oname=rover
	  echo $ctm_name
      [ ! -z $overlap_spk ] && oname=${ctm_name}_o$overlap_spk
      echo "asclite is starting"
      # Run scoring, meaning of hubscr.pl options:
      # -G .. produce alignment graphs,
      # -v .. verbose,
      # -m .. max-memory in GBs,
      # -o .. max N of overlapping speakers,
      # -a .. use asclite,
      # -C .. compression for asclite,
      # -B .. blocksize for asclite (kBs?),
      # -p .. path for other components,
      # -V .. skip validation of input transcripts,
      # -h rt-stt .. removes non-lexical items from CTM,
      cp $data/stm $dir/ascore_rover/ '&&' \
      cp $dir/ascore_rover/${ctm_name}.ctm $dir/ascore_rover/${oname}.ctm '&&' \
      $hubscr -G -v -m 1:2 -o$overlap_spk -a -C -B 8192 -p $hubdir -V -l english \
        -h rt-stt -g $data/glm -r $dir/ascore_rover/stm $dir/ascore_rover/${oname}.ctm || exit 1
      # Compress some scoring outputs : alignment info and graphs,
      echo -n "compressing asclite outputs "
      ascore=$dir/ascore_rover
      gzip -f $ascore/${oname}.ctm.filt.aligninfo.csv
      cp $ascore/${oname}.ctm.filt.alignments/index.html $ascore/${oname}.ctm.filt.overlap.html
      tar -C $ascore -czf $ascore/${oname}.ctm.filt.alignments.tar.gz ${oname}.ctm.filt.alignments
      rm -r $ascore/${oname}.ctm.filt.alignments
      echo done
    else
      $cmd LMWT=$min_lmwt:$max_lmwt $dir/ascoring/log/score.LMWT.log \
        cp $data/stm $dir/ascore_rover/ '&&' \
        $hubscr -p $hubdir -v -V -l english -h hub5 -g $data/glm -r $dir/ascore_rover/stm $dir/ascore_rover/rover.ctm || exit 1
    fi
  fi
fi


exit 0
