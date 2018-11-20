#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import kaldi_io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")


def convert_entropy_from_exp(kmat):
   kent = dict()
   for key, mat in kmat:
      pmat = mat/mat.sum(axis=1, keepdims=True)
      entropy=(-pmat * np.log2(pmat)).sum(axis=1)
      kent[key] = entropy
   return kent

def save_tmp_results(dest_dir, fname, kent):
    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:'+dest_dir+'/'+fname+'.ark,'+dest_dir+'/'+fname+'.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for key,mat in kent: 
            kaldi_io.write_mat(f, mat, key=key)

def plot_cmp_kent(fst_kent, snd_kent, dest_dir, plot_name):
    x = []
    y = []
    num_utt = len(fst_kent.keys())
    for key, fst_ent in fst_kent.items():
        snd_ent = snd_kent[key]
        if fst_ent.shape[0] != snd_ent.shape[0]:
            print "Dimension mismatch. Something wrong with preprocess."
            break
        for frame_idx in xrange(fst_ent.shape[0]):
            x.append(fst_ent[frame_idx])
            y.append(snd_ent[frame_idx])
       
    x = np.asarray(x)
    y = np.asarray(y)
    #rg = sns.regplot(x=x, y=y, color="g", ci=68)
    plt.scatter(x, y, s=10, alpha=0.8)
    # calc the trendline
   
    #z = np.polyfit(x, y, 1)
    #p = np.poly1d(z)
    # the line equation:
    #print "y=%.6fx+(%.6f)"%(z[0],z[1])
    #plt.plot(x,p(x),"r--")
    print "save image to ", plot_name
    plt.savefig(dest_dir + "/" + plot_name)  
    #rg.figure.savefig(dest_dir + "/reg_" + plot_name)

if __name__ == "__main__":
    fst_mdl_scp_fname, snd_mdl_scp_fname = sys.argv[1], sys.argv[2]
    dest_dir, plot_name = sys.argv[3], sys.argv[4]
    print "read kaldi-mat from scp file"
    fst_mdl_kmat = kaldi_io.read_mat_scp(fst_mdl_scp_fname)
    snd_mdl_kmat = kaldi_io.read_mat_scp(snd_mdl_scp_fname)
    print "(1st) convert entropy from exp format"
    if not os.path.isfile(dest_dir + "/fst.tmp.npy") or True:
        fst_kent = convert_entropy_from_exp(fst_mdl_kmat)
        np.save(dest_dir + "/fst.tmp", fst_kent)
    else:
        fst_kent = np.load(dest_dir + "/fst.tmp.npy")
    #save_tmp_results(dest_dir, "fst", fst_kent)
    print "(2nd) convert entropy from exp format"
    if not os.path.isfile(dest_dir + "/snd.tmp.npy") or True:
        snd_kent = convert_entropy_from_exp(snd_mdl_kmat)
        np.save(dest_dir + "/snd.tmp", snd_kent)
    else:
        snd_kent = np.load(dest_dir + "/snd.tmp.npy")
    #save_tmp_results(dest_dir, "snd", fst_kent)
    print "compare two entropy dimension"
    plot_cmp_kent(fst_kent, snd_kent, dest_dir, plot_name)


