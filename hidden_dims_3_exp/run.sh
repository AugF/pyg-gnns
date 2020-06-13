git commit -am "add ..."
git checkout master
date
bash non_head_exp.sh >>non_head_exp.log 2>&1
echo "finish non_head_exp.sh"
date
bash multi_head_exp.sh >>multi_head_exp.log 2>&1
echo "finish multi_head_exp.sh"
date
git commit -am "add .."
git checkout memory
bash non_head_memory.sh >>non_head_memory.log 2>&1
echo "finish non_head_memory.sh"
date
bash multi_head_memory.sh >>multi_head_memory.log 2>&1
echo "finish multi_head_memory.sh"
date