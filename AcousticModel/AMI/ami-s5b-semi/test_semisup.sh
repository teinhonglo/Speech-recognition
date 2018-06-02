
mic=ihm
data=data/$mic/semisup
data_root=data/$mic/semisup
exp_root=exp/$mic
st_exp_root=exp/$mic/test_semi
train_sup_dir=train_sup
train_unsup_dir=train_unsup_86k
gmm=tri3_cleaned

local/semisup/chain/run_tdnn_20k_semisupervised.sh \
    --mic $mic \
    --supervised-set ${train_sup_dir} \
    --unsupervised-set ${train_unsup_dir} \
    --sup-chain-dir $exp_root/chain_cleaned/tdnn1e_sp_bi \
    --sup-lat-dir $exp_root/chain_cleaned/${gmm}_train_cleaned_sp_comb_lats \
    --sup-tree-dir $exp_root/chain_cleaned/tree_bi \
    --ivector-root-dir $exp_root/nnet3_cleaned \
    --chain-affix _semi20k_80k \
    --data_root $data_root \
    --exp-root $st_exp_root --stage 3
