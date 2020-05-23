"""
convert.py
获取cuda的nvtx的运行时

input: .sqlite:
    获取方法:
    nsys profile -t cuda,osrt,nvtx -o report -w true python main.py
    nsys export --type sqlite report.qdrep
output:
    查看方法：https://jsoneditoronline.org/
    格式：Node{key: {0: (start_time, end_time}, 1: {Node, ..., Node}}
    file_name_cpu.json
    file_name_cuda.json
"""
import os
import sys
import json
import sqlite3


class Node:
    def __init__(self, key, value, parent=-1):
        self.key = key
        self.value = value
        self.parent = parent # 父亲
        self.child = [] # 孩子列表

    def __repr__(self):
        return "Node(key={}, value=({}, {}))".format(self.key, self.value[0], self.value[1])


def create(input):
    """
    :param input: [(key, value)],  key: str, value: 2-tuple
    :return: roots: first layer
    """
    # 逻辑
    # 如果当前的结束值小于上一级的结束值，则位于上一级的子节点中
    # 如果当前的开始值大于该级的结束值，则一直往上查找，到合适的位置进行插入
    id = 0
    maps = []
    roots = [0, ]  # 存储根节点的孩子，为了dfs
    flag = True
    for line in input:
        key, value = line[2], line[:-1]
        if key == "epochs0": flag = False # start!!!
        if flag: continue
        if not maps:
            maps.append(Node(key, value))
        else:
            if value[0] > maps[id - 1].value[1]:
                p = id - 1
                while p != -1 and value[0] > maps[p].value[1]:
                    p = maps[p].parent

                maps.append(Node(key, value, parent=p))  # 寻找父亲节点
                if p == -1:
                    roots.append(id)
                else:
                    maps[p].child.append(id)
            elif value[1] < maps[id - 1].value[1]: # 小于，一定为父节点的子节点
                maps.append(Node(key, value, parent=id - 1))
                maps[id - 1].child.append(id)
        id += 1
    return roots, maps


def cal_cuda_time(maps, cur):
    """
    cal cuda time
    :param maps:
    :return:
    """
    sql_runtime = "select correlationId from cupti_activity_kind_runtime where start >= ? and end <= ?"
    cnt = 1
    for node in maps:
        key, value = node.key, node.value
        start_time = sys.maxsize
        end_time = 0
        event_ids = cur.execute(sql_runtime, value).fetchall()
        # 3. find event_id
        for event_id in event_ids:
            # print(event_id)
            for table in ['cupti_activity_kind_kernel', 'cupti_activity_kind_memcpy', 'cupti_activity_kind_memset']:
                sql_cuda = "select start, end from {} where correlationId = {}".format(table, event_id[0])
                events = cur.execute(sql_cuda)
                event = events.fetchone()
                while event:
                    start_time = min(start_time, event[0])
                    end_time = max(end_time, event[1])
                    event = events.fetchone()
        node.value = (start_time, end_time)
        cnt += 1
        if cnt % 10 == 0:
            print("finish {}".format(cnt))


def dfs(roots, maps): # 通过dfs来输出层次结构
    if not roots:
        return None
    ans = {}
    for root in roots:
        key, val = maps[root].key, maps[root].value
        ans[key] = [val, dfs(maps[root].child, maps)]
    return ans


def convert(file_path):
    con = sqlite3.connect(file_path + '.sqlite')
    cur = con.cursor()

    nvtx_sql = "select start, end, text from nvtx_events"
    res = cur.execute(nvtx_sql).fetchall()
    roots, maps = create(res)
    # write cpu.json
    ans = dfs(roots, maps)
    with open(file_path + "_cpu.json", "w") as f:
        json.dump(ans, f)

    cal_cuda_time(maps, cur)
    # write cuda.json
    ans = dfs(roots, maps)
    with open(file_path + "_cuda.json", "w") as f:
        json.dump(ans, f)
    con.close()


if __name__ == '__main__':
    file_name = "config0_ggnn_reddit"
    file_path = os.path.join(os.path.dirname(__file__), 'files', file_name)
    # convert(file_name)
    # 后续读取，第一层，第二层读
    with open(file_path + "_cuda.json") as f:
        dicts = json.load(f)

    def get_time(period):
        return period[1] - period[0]

    # epochs
    epochs = []
    forwards = []
    backwards = []
    evals = []
    for key in dicts.keys():
        if 'epochs' not in key: continue
        epochs.append(get_time(dicts[key][0]))
        forwards.append(get_time(dicts[key][1]['train'][1]['forward'][0]))
        backwards.append(get_time(dicts[key][1]['train'][1]['backward'][0]))
        evals.append(get_time(dicts[key][1]['eval'][0]))

    print(epochs)
    print(forwards)
    print(backwards)
    print(evals)
