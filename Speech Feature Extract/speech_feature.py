#!/usr/bin/env python
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import numpy as np

def run():
	files = ["chia", "joy","PR"]
	ratelist=[]
	siglist=[]
	anslist=[]
	for fileone in files:
		for doc_item in os.listdir(fileone):
			# join dir path and file name
			doc_item_path = os.path.join(fileone, doc_item)
			# check whether a file exists before read

			if os.path.isfile(doc_item_path):
				if doc_item[0]=='y':
					anslist.append(1)
				else:
					anslist.append(0)
				(rate,sig) = wav.read(doc_item_path)
				ratelist.append(rate)
				siglist.append(sig)
							
	mfcc_feat_list = []
	count =0
	for i in range(len(ratelist)):
		mfcc_feat = mfcc(siglist[i], ratelist[i])
		mfcc_feat_list.append(mfcc_feat)
		
		if count<len(mfcc_feat[:, 0]):
			count =len(mfcc_feat[:, 0])
	extended_mfcc_feat_list = []
	#	fbank_feat = logfbank(sig, rate)
	for mfcc_feat in mfcc_feat_list:
		new_mfcc_feat = []
		for i in range(0,count-len(mfcc_feat)):
			new_mfcc_feat.append([0] * 13)
	
		extended_mfcc_feat_list.append(np.array(mfcc_feat.tolist() + new_mfcc_feat))
		#print mfcc_feat.shape
	
	return [np.array(extended_mfcc_feat_list), np.array(anslist), count]
