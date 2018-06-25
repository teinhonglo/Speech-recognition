#!/bin/bash
mic=ihm




for x in exp/$mic/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
for x in exp/$mic/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done

echo ""
echo "Semi-supervised training"
for x in exp/$mic/semisup_20k/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
echo ""
echo "Chain model training"
for x in exp/$mic/semisup_20k/chain*/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
