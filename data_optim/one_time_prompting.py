"""
把 messages 中的 assistant+user 看作一个 Round，然后标注 Round 之间的依赖关系，输出一个 edge list，格式如下：
[
  {"source": "Round X", "target": "Round Y", "reason": "Brief justification"},
  ...
]
"""

import argparse
import json
import random
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# from call_idealab import call_idealab
from call_openrouter import call_openrouter
from prompt import LOOSE_PROMPT, STRICT_PROMPT


parser = argparse.ArgumentParser(description="Process and optimize multi-round trajectories.")
parser.add_argument("--data", type=str, required=True, help="Input JSONL of multi-turn trajectories.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory. Defaults to dirname(--data).")
parser.add_argument("--model", type=str, default="openai/gpt-5.4")
parser.add_argument("--mp", type=int, default=10)
parser.add_argument("--prompt_version", type=str, default="strict", choices=["loose", "strict"])
parser.add_argument("--sample_size", type=int, default=0, help="If > 0, randomly pick this many samples (seeded). 0 = full data.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_suffix", type=str, default="", help="Append to output filename.")

args = parser.parse_args()

PROMPT = STRICT_PROMPT if args.prompt_version == "strict" else LOOSE_PROMPT

source_file = args.data
output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.data))
model_subdir = args.model.replace("/", "-")  # 替换斜杠以适应文件命名

suffix = f"_{args.output_suffix}" if args.output_suffix else ""
output_file = os.path.join(
    output_dir,
    model_subdir,
    f"dependency_{args.prompt_version}prompt{suffix}_" + os.path.basename(source_file),
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

def _extract_q(messages):
    # 如果第一条消息是 system，返回 index=1 消息的 content 作为问题；
    if messages[0]["role"] == "system":
        return messages[1]["content"]
    
    # 如果第一条消息是 user，直接从 content 中提取问题文本
    elif messages[0]["role"] == "user":
        text = messages[0]["content"]
        q = text.split("Input Question:")[1].split("Input image: ")[0].strip()
        return q


def process_line(line):
    """处理单行数据的函数"""
    data = json.loads(line)
    messages = data["messages"]
    question = _extract_q(messages)

    input_traj_string = f"**Round 1**: \nUser Query: {question}\n\n"

    assistant_user_pairs = [messages[i:i+2] for i in range(1, len(messages), 2)]

    for i, pair in enumerate(assistant_user_pairs):
        if len(pair) == 2:
            input_traj_string += f"**Round {i+2}**: \nAssistant: {pair[0]['content']} \n\nUser: {pair[1]['content']}\n\n"
        elif len(pair) == 1:
            input_traj_string += f"**Round {i+2}**: \nAssistant: {pair[0]['content']}\n\n"
        else:
            raise ValueError(f"invalid pair length {len(pair)}")
    
    prompt = PROMPT.format(INPUT_TRAJ_STRING=input_traj_string)
    
    # completion = call_idealab(prompt, image_url=None, model="gpt-51-1113-global", max_retries=8)
    completion = call_openrouter(prompt, image_url=None, model=args.model, max_retries=8)

    if 'error' not in completion:
        data['dependency_raw'] = completion['choices'][0]['message']['content']
        return data
    return None


def main():
    done = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            done = {json.loads(l)["_idx"] for l in f if l.strip()}
        print(f"已完成 {len(done)} 条")

    with open(source_file) as f:
        all_lines = list(enumerate(f))

    if args.sample_size and args.sample_size > 0:
        rng = random.Random(args.seed)
        picked_indices = sorted(rng.sample(range(len(all_lines)), k=min(args.sample_size, len(all_lines))))
        pending = [(i, ln) for i, ln in all_lines if i in set(picked_indices) and i not in done]
        print(f"随机抽样 {len(picked_indices)} 条 (seed={args.seed})，其中待处理 {len(pending)} 条")
    else:
        pending = [(i, ln) for i, ln in all_lines if i not in done]
        print(f"待处理 {len(pending)} 条")

    lock = Lock()
    with open(output_file, "a", buffering=1) as g, \
        ThreadPoolExecutor(max_workers=args.mp) as ex:
        futures = {ex.submit(process_line, ln): i for i, ln in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), ncols=50):
            result = fut.result()
            if not result:
                continue
            result["_idx"] = futures[fut]
            with lock:
                g.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
