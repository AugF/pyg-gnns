echo "begin config exp"
date
bash config_exp.sh >config_exp.log 2>&1
date
echo "begin degree exp"
bash degree_exp.sh >degree_exp.log 2>&1