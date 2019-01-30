#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
import kaldi_io
import numpy as np
import os.path

def reverse_kGroup_from_feats(K, kmat):
    print "reverse kGroup, k = " + str(K)
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
        kmat_rev[key] = np.copy(mat)
    return kmat_rev

def save_results(dest_dir, fname, kmat):
    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:'+dest_dir+'/data/'+fname+'.ark,'+dest_dir+'/'+fname+'.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for key, mat in kmat.iteritems(): 
            kaldi_io.write_mat(f, mat, key=key)
    return 0

def save_results_nj(num_jobs, dest_dir, fname, kmat):
    num_utts = len(kmat.keys())
    num_utts_nj = num_utts / num_jobs
    kmat_nj = {}
    count_nj = 0
    job = 1
    for key, mat in kmat.iteritems():
        kmat_nj[key] = mat
        count_nj += 1
        if count_nj % num_utts_nj == 0 or count_nj == num_utts:
            save_results(dest_dir, fname + "." + str(job), kmat_nj)
            kmat_nj = {}
            job += 1        
    return 0

if __name__ == "__main__":
    K = int(sys.argv[1])
    num_jobs = int(sys.argv[2])
    feats_scp_fname = sys.argv[3]
    dest_feat_dir = sys.argv[4]
    fname = sys.argv[5]
    fname += str(K)
    print "read kaldi-mat from scp file"
    feats_kmat = kaldi_io.read_mat_scp(feats_scp_fname)
    feats_kmat_rev = reverse_kGroup_from_feats(K, feats_kmat)
    save_results_nj(num_jobs, dest_feat_dir, fname, feats_kmat_rev)
