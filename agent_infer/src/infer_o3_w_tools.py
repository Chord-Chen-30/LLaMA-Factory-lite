import argparse
import json
from openai import OpenAI
from o3_prompt import sys_prompt, prompt_ins, tools_list, prompt_ins3
from utils import text_wrap, image_wrap
import os
import time
import requests
from agent_wrapper import ToolCallWrapper
from utils import load_jsonl, extract_answer, extract_think, extract_tool_call
import datetime
from tqdm import tqdm
from tool_image_search import upload_to_oss
from utils import image_to_base64

import logging

from dotenv import load_dotenv
load_dotenv()

IDEALAB_KEY = os.getenv('IDEALAB_API_KEY', None)
if not IDEALAB_KEY:
    raise ValueError("IDEALAB_API_KEY not set in environment variables")

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_idealab', type=str, default="o3-0416-global")
parser.add_argument('--data', type=str, default="./data/toolcall_3more_ocr_code_6200_cz_cleaned_1216.jsonl")
parser.add_argument('--max_rounds', type=int, default=128)
parser.add_argument('--mp', type=int, default=6)

parser.add_argument('--output_dir', type=str, default=f'./data')
parser.add_argument('--date', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d"))
parser.add_argument('--overwrite', type=int, default=1)
parser.add_argument('--log_level', type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(level=getattr(args, "log_level"), format='%(asctime)s - %(levelname)s - %(message)s')


if args.data.endswith(".json"):
    with open(args.data, "r") as f:
        data = json.load(f)
elif args.data.endswith(".jsonl"):
    with open(args.data, "r") as f:
        data = [json.loads(line) for line in f]

output_path = os.path.join(
    args.output_dir, 
    args.date,
    args.model_name_idealab,
    os.path.basename(args.data).replace(".jsonl", "_o3_w_tools.jsonl")
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

finished_lines = 0
if os.path.exists(output_path):
    if args.overwrite:
        os.remove(output_path)
    else:
        with open(output_path, 'r') as f:
            finished_lines = len(f.readlines())

def call_idealab(messages, model="o3-0416-global", max_retries=8):
    """调用 Idealab API"""

    url = os.getenv("IDEALAB_API_URL", "")
    if not url:
        raise ValueError("IDEALAB_API_URL is not set")
    headers = {
        "Authorization": f"Bearer {IDEALAB_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "stream": "false",
        "messages": messages
    }
    ret = {"error": "Init placeholder"}
    for attempt in range(max_retries):
        response_raw = None
        try:
            response_raw = requests.post(url, headers=headers, json=data, timeout=60)
            response_raw.raise_for_status()
            
            # print(response_raw.json())
            ret = response_raw.json()
            response = response_raw.json()['choices'][0]['message']['content']
            return ret
            
        except Exception as e:
            logging.warning(f"Idealab API call error! Attempt {attempt + 1}/{max_retries}: {e}")
            logging.error(f"{response_raw}")
            
            if attempt < max_retries - 1:
                sleep_time = 1
                time.sleep(sleep_time)
            else:
                logging.error("Max retries reached for Idealab API")
                ret['error'] = str(e)
    
    return ret

# Test
# image_url = "https://example.com/example.jpg"
image_url = "./agent_infer/downloaded_images/sc.jpg"
if image_url.startswith("file://"):
    image_url = upload_to_oss(image_url)
    if image_url is None:
        print(f"[Error] 上传oss失败：{image_url}。")
    else:
        print(f"[Info] 上传oss成功：{image_url}。")


tools = json.dumps(tools_list, ensure_ascii=False)
message = [
    {"role": "system", "content": [text_wrap(sys_prompt)]},
    {"role": "user", "content": [
        text_wrap(prompt_ins.replace("{Question}", "图中有什么，你可以使用相应搜索工具。").replace("{Image_url}", image_url).replace("{Tools}", tools)),
        image_wrap(image_to_base64(image_url))
    ]}
]
print(call_idealab(message, model=args.model_name_idealab))

tool_caller = ToolCallWrapper()

def process_item(i, d, args):
    question = d.get('question') or d.get('prompt') or d.get('origin_question')
    image = d.get('image_path') or d.get('file_path')

    if isinstance(image, list):
        image = image[0]

    assert question and image, print(f"question: {question}; image: {image}")

    if image.startswith("/") or image.startswith("file://"):
        image_base64 = image_to_base64(image)
    else:
        image_base64 = image
    
    message = [
        {"role": "system", "content": [text_wrap(sys_prompt)]},
        {"role": "user", "content": [
            text_wrap(prompt_ins.replace("{Question}", question).replace("{Image_url}", image).replace("{Tools}", tools)),
            # text_wrap(prompt_ins3.replace("{Question}", question).replace("{ImageUrl}", image)),
            image_wrap(image_base64),
        ]},
    ]

    # message = [{"role": "system", "content": sys_prompt},{"role": "user", "content": "Dummy"}]

    response_answer = ''

    for r in range(args.max_rounds):
        completion = call_idealab(message, model=args.model_name_idealab)
        if 'error' not in completion:
            output_str = completion['choices'][0]['message']['content']
        else:
            output_str = completion['error']

        # A rare bug. Sometimes the output is BadRequestError. Continue for now.  --zhuochen
        if not isinstance(output_str, str):
            logging.warning(f"============\noutput_str is not of type string: {output_str}\n============")
            continue

        assistant_str = ""
        user_str = "Your output format is incorrect. Please use <think></think>, <tool_call></tool_call>, <answer></answer> tags to wrap your output."

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

        completion = call_idealab(message)
        if 'error' not in completion:
            output_str = completion['choices'][0]['message']['content']
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

    d['messages'] = message
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
    
    pbar = tqdm(total=len(futures), ncols=50)
    
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