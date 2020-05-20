echo "begin base exp..."
bash ./base_exp.sh >> base_exp.log
echo "begin sp exp..."
bash ./super_paras_exp.sh >> super_paras_exp.log
echo "begin config exp..."
bash ./config_exp.sh >> config_exp.log
echo "begin hds exp..."
bash ./hidden_dims_exp.sh >> hidden_dims_exp.log
echo "begin layers exp..."
bash ./layers_exp.sh >> layers_exp.log
echo "begin sparse feats exp..."
bash ./sparse_feats_exp.sh >> sparse_feats_exp.log
echo "begin graph exp..."
bash ./graph_exp.sh >> graph_exp.log
