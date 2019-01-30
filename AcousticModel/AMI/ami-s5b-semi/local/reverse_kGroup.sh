#!/bin/bash

. ./cmd.sh
. ./path.sh

src_dir=data/ihm/semisup_10k/train_sup
dest_dir=data/ihm/semisup_10k/train_sup_rev
frame_size=5
num_jobs=30
fname=feats

# Train systems,
. utils/parse_options.sh

if [ ! -d $dest_dir ]; then
    utils/copy_data_dir.sh $src_dir $dest_dir
fi    

if [ ! -d $dest_dir/data ]; then
    mkdir -p $dest_dir/data     
fi

python local/reverse_kGroup.py $frame_size $num_jobs $src_dir/feats.scp $dest_dir $fname
cat $dest_dir/${fname}${frame_size}.*.scp > $dest_dir/${fname}.scp
rm -rf $dest_dir/${fname}${frame_size}.*.scp
