法一：
一次性标注依赖树。

流程：
先跑 one_time_prompting.py，得到 DAG 标注
python one_time_prompting.py --data {} --model {openrouter_model_name}

再跑 level12_optim.py 
  - level 1:删掉无用叶节点
  - level 2: merge 完全同父、同子节点（即之间没有依赖关系）
messages 字段即为更新后的轨迹
python level12_optim.py --data {} --level {1,2}