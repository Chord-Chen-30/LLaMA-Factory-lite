from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from openai import OpenAI
import argparse
from tqdm import tqdm
import time
from prompt import sys_prompt, prompt_ins, prompt_ins3
import logging
from pathlib import Path

from utils import load_jsonl, extract_answer, extract_think, extract_tool_call
from agent_wrapper import ToolCallWrapper
import os
import json
from utils import test_infer
from utils import text_wrap, image_wrap

from urllib.parse import urlparse, unquote
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-VL-30B-A3B-Thinking')
parser.add_argument('--data', type=str, default='./data/c4_demo.jsonl')
parser.add_argument('--max_rounds', type=int, default=128)
parser.add_argument('--max_tool_call_num', type=str, default=128)
parser.add_argument('--mp', type=int, default=6)

parser.add_argument('--output_dir', type=str, default=f'./agent_infer/result/')
parser.add_argument('--date', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d"))
parser.add_argument('--overwrite', type=int, default=1)
parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')

output_path = os.path.join(
    args.output_dir, 
    args.date,
    '/'.join(Path(args.model_name_or_path).parts[-2:]),
    os.path.basename(args.data).replace(".jsonl", "_w_tools.jsonl")
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

finished_lines = 0
if os.path.exists(output_path):
    if args.overwrite:
        os.remove(output_path)
    else:
        with open(output_path, 'r') as f:
            finished_lines = len(f.readlines())


client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)

def call_client(message, top_p=0.95, temperature=0.6, max_try=10):
    start_time = time.time()
    ret = {"error": "Init placeholder"}
    for i in range(max_try):
        try:
            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=message,
                stream=False,
                top_p=top_p,
                temperature=temperature,
                max_tokens=2048
            )
            ret = completion
            break
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            ret = {'error': e}
        time.sleep(0.8)

    logging.info(f"call_client() Time taken: {time.time() - start_time:.2f} seconds")
    return ret

tool_caller = ToolCallWrapper()

def test_case():
    image_url = "https://example.com/example.jpg"
    message = [
        {"role": "system", "content": [text_wrap(sys_prompt)]},
        {"role": "user", "content": [
            text_wrap(prompt_ins3.replace("{Question}", "这是哪里，你可以使用相应搜索工具。").replace("{ImageUrl}", image_url)),
            image_wrap(image_url)
        ]}
    ]
    completion = call_client(message)

    if 'error' in completion:
        logging.info(f"Test failed.")
    else:
        logging.info(f"Test passed. Output:\n{completion.choices[0].message}")
# test_case()

data = load_jsonl(args.data)

def process_item(i, d, args):
    question = d.get('question') or d.get('prompt') or d.get('origin_question')
    image = d.get('file_path') or d.get('image_url') or d.get('image_path')

    assert question and image, print("question or image_url is None")
    if not image.startswith("http"):
        image = f"file://{image}"
        
    message = [
        {"role": "system", "content": [text_wrap(sys_prompt)]},
        {"role": "user", "content": [
            text_wrap(prompt_ins3.replace("{Question}", question).replace("{ImageUrl}", image)),
            image_wrap(image),
        ]},
    ]

    # message = [{"role": "system", "content": sys_prompt},{"role": "user", "content": "Dummy"}]

    response_answer = ''
    
    # breakpoint()

    for r in range(args.max_rounds):
        completion = call_client(message)
        if 'error' not in completion:
            output_str = completion.choices[0].message.content
        else:
            output_str = completion['error']

        # A rare bug. Sometimes the output is BadRequestError. Continue for now.  --zhuochen
        if not isinstance(output_str, str):
            logging.warning(f"============\noutput_str is not of type string: {output_str}\n============")
            continue

        # output_str = test_infer(r)

        assistant_str = ""
        user_str = "Your output format is incorrect. Please use <think>, <tool_call>, <answer> tags to wrap your output."

        # Error format
        if ('<think>' not in output_str) and ('<tool_call>' not in output_str) and ('<answer>' not in output_str):
            assistant_str = f"<think>{output_str}</think>"
            user_str = f"You do not invoke any tool. Continue thinking or answering the question. Remember to use <think></think> or <answer></answer> to wrap your response."
        
        # Correct format
        if ('<think>' in output_str) or ('</think>' in output_str):
            response_think = extract_think(output_str)
            assistant_str += f"<think>{response_think}</think>"
        
        if '<tool_call>' in output_str:
            response_tool_call_list = extract_tool_call(output_str)

            # Successfully extract tool calls
            if isinstance(response_tool_call_list, list):
                assistant_str += "<tool_call>"
                assistant_str += "".join([json.dumps(t) for t in response_tool_call_list])
                assistant_str += "</tool_call>"

                tool_response = ''
                for tool_call_params in response_tool_call_list:
                    tool_response += f"From: {tool_call_params.get('name', 'tool_call')}\n"
                    tool_response += tool_caller.call_tools(tool_call_params)
                
                user_str = f"<tool_response>{tool_response}</tool_response>"

            # Failed to extract tool calls. response_tool_call_list is error string.
            else:
                assistant_str = output_str
                user_str = response_tool_call_list

        if '<answer>' in output_str:
            response_answer = extract_answer(output_str)
            message.append({"role": "assistant", "content": f"{assistant_str}<answer>{response_answer}</answer>"})
            break # Break conversation rounds

        message.append({"role": "assistant", "content": [text_wrap(assistant_str)]})
        message.append({"role": "user", "content": [text_wrap(user_str)]})                    
    
    # Reached max round. Force an answer.
    if r == (args.max_rounds-1):
        logging.info("Maximum number of rounds reached. Force an answer.")
        message[-1]['content'][0]['text'] += "\nYou have reached the maximum number of rounds and tool calls. Give an answer surrounded by <answer></answer> now."

        completion = call_client(message)
        if 'error' not in completion:
            output_str = completion.choices[0].message.content
        else:
            output_str = completion['error']
            logging.error(f"{output_str}")
        
        # output_str = test_infer(r)

        if '<answer>' in output_str:
            response_answer = extract_answer(output_str)
            message.append({"role": "assistant", "content": [text_wrap(f"<answer>{response_answer}</answer>")]})
        else:
            logging.warning("Maximum number of rounds reached. No <answer> tag found. Put last output as answer.")
            message.append({"role": "assistant", "content": [text_wrap(f"<answer>{output_str}</answer>")]})

    d['traj'] = message
    d['response_answer'] = response_answer

    return i, d


from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

result_queue = queue.Queue()
finished_count = 0

with (ThreadPoolExecutor(max_workers=args.mp) as executor, open(output_path, 'a', buffering=1) as g):
    
    # 提交所有任务
    futures = {
        executor.submit(process_item, i + finished_lines, d, args): 
        i + finished_lines for i, d in enumerate(data[finished_lines:])
    }
    
    pbar = tqdm(total=len(futures), ncols=100, desc="Inference with tools")
    
    # 按完成顺序处理结果，但按原始顺序写入
    future_to_idx = {future: idx for future, idx in futures.items()}
    next_idx_to_write = finished_lines
    
    # 缓存结果，用于按顺序写入
    result_cache = {}
    
    for future in as_completed(futures):
        idx, new_data = future.result()
        result_cache[idx] = new_data
        
        # 检查是否可以按顺序写入缓存的结果
        while next_idx_to_write in result_cache:
            g.write(json.dumps(result_cache.pop(next_idx_to_write), 
                              ensure_ascii=False) + '\n')
            next_idx_to_write += 1
            finished_count += 1
            pbar.update(1)
    
    pbar.close()

# with open(output_path, 'a', buffering=1) as g:
#     for d in tqdm(data[finished_lines:], ncols=50):
#         new_data = process_item(d, args)
#         g.write(json.dumps(d, ensure_ascii=False) + '\n')