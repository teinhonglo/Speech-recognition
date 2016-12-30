#!/usr/bin/env python
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("bounce.wav")
mfcc_feat = mfcc(sig, rate)
fbank_feat = logfbank(sig, rate)

print(fbank_feat[0:2, :])

