#!/bin/bash

. ./cmd.sh
. ./path.sh 

set -euo pipefail

cmd=run.pl

. parse_options.sh || exit 1;


for decode_set in dev eval; do
    local/score_combine.sh --cmd "$decode_cmd" --stage 0 \
                           data/ihm/semisup/${decode_set}_hires \
                           data/lang_ami.o3g.kn.pr1-7 \
                           exp/ihm/semisup_20k/chain_semi20k_80k/tdnn_1a_sp_bi/decode_${decode_set}:1 \
                           exp/ihm/semisup_20k/chain_semi20k_80k/tdnn_1b_sp_bi/decode_${decode_set}:1 \
                           exp/ihm/semisup_20k/chain_semi20k_80k/tdnn_1c_sp_bi/decode_${decode_set}:1 \
                           exp/ihm/semisup_20k/chain_semi20k_80k/tdnn_1d_sp_bi/decode_${decode_set}:1 \
                           exp/ihm/tdnn_combine/tdnn_reg/decode_${decode_set}
done							  
