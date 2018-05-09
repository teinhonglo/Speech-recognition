#!/bin/bash
mic=ihm
a=
for x in $mic $a; do
  echo "abc", $x
done
for x in exp/$mic/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done

echo ""
echo "Semi-supervised training"
for x in exp/$mic/semisup_20k/*/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
