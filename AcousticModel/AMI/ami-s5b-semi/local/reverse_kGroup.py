#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import kaldi_io
import numpy as np
import os.path


def reverse_kGroup_from_feats(K, kmat):
    kmat_rev = {}
    for key, mat in kmat:
        st_p = 0
        num_frames = mat.shape[0]
        while st_p + K < num_frames:
            mid_p = K / 2
            end_p = st_p + K - 1
            for i in xrange(mid_p):
                mat[st_p + i], mat[end_p - i] = np.copy(mat[end_p - i]), np.copy(mat[st_p + i])
            st_p += K
        kmat_rev[key] = mat
    return kmat_rev

def save_results(dest_dir, fname, kmat_rev):
    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:'+dest_dir+'/'+fname+'.ark,'+dest_dir+'/'+fname+'.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for key, mat in kmat_rev.iteritems(): 
            #print key, mat
            kaldi_io.write_mat(f, mat, key=key)

if __name__ == "__main__":
    feats_scp_fname = sys.argv[1]
    dest_feat_dir = sys.argv[2]
    K = int(sys.argv[3])
    num_jobs = int(sys.argv[4])
    fname = "new_feat" + str(K)
    print "read kaldi-mat from scp file"
    feats_kmat = kaldi_io.read_mat_scp(feats_scp_fname)
    feats_kmat_rev = reverse_kGroup_from_feats(K, feats_kmat)
    num_utts = len(feats_kmat_rev.keys())
    save_results(dest_feat_dir, fname, feats_kmat_rev)
