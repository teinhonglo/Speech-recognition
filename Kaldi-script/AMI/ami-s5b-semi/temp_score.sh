#!/bin/bash

for x in exp/ihm/adapt/*/*/tdnn_7b/decode_*; do grep Sum $x/*scor*/*ys | utils/best_wer.sh; done
