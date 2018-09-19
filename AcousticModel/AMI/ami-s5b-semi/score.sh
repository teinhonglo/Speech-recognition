#!/bin/bash
mic=ihm




#for x in exp/$mic/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
#for x in exp/$mic/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
<<WORD
echo ""
echo "Semi-supervised training"
for x in exp/$mic/semisup_20k/nnet3*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
for x in exp/$mic/semisup_20k/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done

echo ""
echo "Chain model training"
for x in exp/$mic/semisup_20k/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
echo ""
echo "Ensemble Learning"
for x in exp/$mic/semisup_20k/ensemble/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
echo "Weighted transfer"
for x in exp/$mic/transfer_20k/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
WORD

echo "Teacher-student training"
for x in exp/$mic/comb_adapt/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
echo ""

echo "Lower-resource"
for x in exp/$mic/semisup_10k/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
echo ""

echo "Transfer Learning"
for x in exp/$mic/semisup_20k/chain_online_librispeech/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
