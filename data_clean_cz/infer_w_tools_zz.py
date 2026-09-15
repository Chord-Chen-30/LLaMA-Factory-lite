# from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from openai import OpenAI
import argparse
from prompt import sys_prompt, prompt_ins, prompt_ins2
import logging
from utils import load_jsonl
from agent_wrapper import ToolCallWrapper

import os
import json
import requests
import datetime
import time
import traceback
from tqdm import tqdm
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-VL-30B-A3B-Thinking')
parser.add_argument('--data', type=str, default='data/bc_level1.jsonl')
parser.add_argument('--max_rounds', type=int, default=128)
parser.add_argument('--max_tool_call_num', type=int, default=128)
parser.add_argument('--backend', type=str, choices=['vllm', 'idealab', 'yuanshen', 'transformers'], default='vllm',
                    help='Backend to use: vllm (localhost), idealab, yuanshen, or transformers')
parser.add_argument('--max_workers', type=int, default=1, help='Number of parallel workers')
parser.add_argument('--sequential', action='store_true', help='Run sequentially instead of parallel')
parser.add_argument('--new_file', action='store_true', help='Create new output file (remove existing)')
parser.add_argument('--tool_file_name', type=str, default='zz_1119_v2')
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--temperature', type=float, default=0.6)
args = parser.parse_args()

# ==================== 从环境变量读取 API Keys ====================
YUANSHEN_KEY = os.getenv('YUANSHEN_API_KEY', '')
IDEALAB_KEY = os.getenv('IDEALAB_API_KEY', '')

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 工具函数 ====================
tools_raw = json.load(open(f"tool_schema/{args.tool_file_name}.json", 'r', encoding='utf-8'))
tools = json.dumps(tools_raw, ensure_ascii=False)

def extract_tags(text, tag):
    """Extract content between XML-like tags"""
    import re
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# ==================== API 调用函数 ====================
def call_yuanshen(messages, model="o3-0416-global", max_retries=8):
    """调用原神 API"""
    if not YUANSHEN_KEY:
        raise ValueError("YUANSHEN_API_KEY not set in environment variables")
    
    url = "https://aiopenapi.1688.com/v1/chat/completions"
    headers = {
        "X-PLATFORM": "idealab",
        "Authorization": f"Bearer {YUANSHEN_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "stream": "false",
        "messages": messages
    }
    
    for attempt in range(max_retries):
        try:
            response_raw = requests.post(url, headers=headers, json=data, timeout=60)
            response_raw.raise_for_status()
            print(response_raw)
            response = response_raw.json()['choices'][0]['message']['content']
            return response, True
            
        except Exception as e:
            logger.warning(f"Yuanshen API call error! Attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt < max_retries - 1:
                sleep_time = 1 * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached for Yuanshen API")
                return None, False
    
    return None, False

def call_idealab(messages, model="o3-0416-global", max_retries=8):
    """调用 Idealab API"""
    if not IDEALAB_KEY:
        raise ValueError("IDEALAB_API_KEY not set in environment variables")
    
    url = os.getenv("IDEALAB_API_URL", "")
    if not url:
        raise ValueError("IDEALAB_API_URL is not set")
    headers = {
        "Authorization": f"Bearer {IDEALAB_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "o3-0416-global",
        "stream": "false",
        "messages": messages
    }
    
    for attempt in range(max_retries):
        try:
            response_raw = requests.post(url, headers=headers, json=data, timeout=60)
            print(response_raw.json())
            response_raw.raise_for_status()
            
            response = response_raw.json()['choices'][0]['message']['content']
            return response, True
            
        except Exception as e:
            logger.warning(f"Idealab API call error! Attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt < max_retries - 1:
                sleep_time = 1 * (2 ** attempt)
                time.sleep(sleep_time)
            else:
                logger.error("Max retries reached for Idealab API")
                return None, False
    
    return None, False

def call_vllm(messages, client, model_name, top_p=0.95, temperature=0.6, max_retries=3):
    """调用 VLLM 本地服务"""
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=False,
                top_p=top_p,
                temperature=temperature
            )
            return completion.choices[0].message.content, True
        except Exception as e:
            logger.warning(f"VLLM call error! Attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None, False
    return None, False

def call_transformers(messages, model, processor, max_new_tokens=4096):
    """调用 Transformers 本地模型"""
    try:
        # 将 messages 转换为模型输入格式
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = processor(
            text=[text],
            return_tensors="pt",
        ).to(model.device)
        
        # 生成
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens
        )
        
        # 解码
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        return output_text, True
        
    except Exception as e:
        logger.error(f"Transformers model error: {e}")
        traceback.print_exc()
        return None, False

# ==================== 统一调用接口 ====================
def call_llm(messages, backend, **kwargs):
    """统一的 LLM 调用接口"""
    if backend == 'yuanshen':
        return call_yuanshen(messages, model=kwargs.get('model', 'o3-0416-global'))
    elif backend == 'idealab':
        return call_idealab(messages, model=kwargs.get('model', 'o3-0416-global'))
    elif backend == 'vllm':
        return call_vllm(
            messages, 
            client=kwargs['client'], 
            model_name=kwargs['model_name'],
            top_p=kwargs.get('top_p', 0.95),
            temperature=kwargs.get('temperature', 0.6)
        )
    elif backend == 'transformers':
        return call_transformers(
            messages,
            model=kwargs['model'],
            processor=kwargs['processor']
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

# ==================== 工具执行 ====================
def execute_tool(llm_generated_response, tool_wrapper):
    """
    执行工具调用
    从 llm_generated_response 中提取 <tool_call>...</tool_call> 的内容
    直接传给 tool_wrapper.call_tools
    """
    # 提取 <tool_call> 标签内的内容
    tool_call_str = extract_tags(llm_generated_response, 'tool_call')
    tool_call_flag = False
    
    if not tool_call_str:
        return None, "No tool call correctly extracted.", tool_call_flag
    
    try:
        # 直接传给 tool_wrapper，它会自动处理 JSON 解析
        logger.info(f"Executing tool call: {tool_call_str[:200]}...")
        print(tool_call_str)
        result = tool_wrapper.call_tools(tool_call_str)
        
        # 检查是否执行成功
        result_str = str(result)
        if result and not result_str.startswith("Invalid") and not result_str.startswith("Error"):
            tool_call_flag = True
            logger.info("Tool execution successful")
        else:
            logger.warning(f"Tool execution failed: {result_str[:200]}")
            
    except Exception as e:
        result = f'Error during tool execution: {e}'
        logger.error(f"Tool execution exception: {e}")
        logger.error(traceback.format_exc())
    
    return tool_call_str, result, tool_call_flag

# ==================== 多轮对话处理 ====================
def process_single_item(item, args, llm_kwargs, tool_wrapper):
    """处理单个数据项"""
    current_date = datetime.datetime.now().strftime("%Y年%m月%d日")
    
    question = item.get('question', item.get('prompt', ''))
    image_url = item.get('image_url', item.get('image_path', item.get('file_path', '')))
    answers = item.get("answers", item.get('answer', ''))

    if not question or not image_url:
        logger.error("question or image_url is None")
        return None
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_ins.replace("{Question}", question).replace("{Image_url}", image_url).replace("{Tools}", tools)
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                }
            ]
        }
    ]

    tool_count = 0
    intermediate_steps = []
    multi_turn_response = ''
    # breakpoint()
    for r in range(args.max_rounds):
        # 调用 LLM
        response, response_flag = call_llm(messages, args.backend, **llm_kwargs)
        print(response)

        if not response_flag:
            logger.warning(f"LLM call failed at round {r}")
            break
        
        multi_turn_response += response
        
        # 提取 thinking
        thinking = extract_tags(response.strip(), 'think')
        
        # 检查是否结束（包含 <answer>）
        if '<answer>' in response:
            # 提取最终答案
            final_answer = extract_tags(response, 'answer')
            result = {
                "thinking": thinking,
                "final_answer": final_answer
            }
            intermediate_steps.append(result)
            break
        
        # 处理 tool_call
        if response.strip().endswith("</tool_call>") or response.strip().endswith("</tool_call"):
            if response.strip().endswith("</tool_call"):
                response = response.strip() + ">"
            
            tool_call, tool_response, tool_call_flag = execute_tool(response, tool_wrapper)
            tool_count += 1
            
        elif response.strip().split("</think>")[-1].strip().startswith("<tool_call>") and not response.strip().endswith(">"):
            response = response.strip() + "</tool_call>"
            tool_call, tool_response, tool_call_flag = execute_tool(response, tool_wrapper)
            tool_count += 1
            
        else:
            # 没有工具调用，也没有 answer，可能出错
            logger.warning(f"No tool call or answer found at round {r}")
            result = {
                "thinking": thinking,
            }
            intermediate_steps.append(result)
            continue
        
        # 检查工具调用次数
        if tool_count >= args.max_tool_call_num:
            logger.warning(f"Max tool calls ({args.max_tool_call_num}) reached")
            break
        
        if not tool_call_flag:
            logger.warning(f"Tool call failed at round {r}")
            continue
        
        # 构建 tool_response 消息
        tmp_tool_response = f"<tool_response>\n{json.dumps(tool_response, ensure_ascii=False)[:32000]}\n</tool_response>"
        print(tmp_tool_response)
        multi_turn_response += tmp_tool_response
        
        # 解析工具信息
        try:
            tool_name = json.loads(tool_call)['name']
            tool_args = json.loads(tool_call)['arguments']
            print(f"工具{tool_name}调用成功！")
        except:
            tool_name = "Unknown"
            tool_args = {}
        
        result = {
            "thinking": thinking,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_response": tmp_tool_response,
            "is_tool_success": tool_call_flag,
        }
        intermediate_steps.append(result)
        
        # 更新 messages
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": tmp_tool_response})
    
    # 提取最终答案
    final_answer = extract_tags(multi_turn_response, 'answer')
    if not final_answer:
        final_answer = multi_turn_response.split('<answer>')[-1].split('</answer>')[0].strip() if '<answer>' in multi_turn_response else ""
    
    # 构建返回结果
    result_item = {
        "question": question,
        "image_url": image_url,
        "answers": answers,
        "final_response": final_answer,
        "num_tool_calls": tool_count,
        "intermediate_steps": {f'Step {i + 1}': step for i, step in enumerate(intermediate_steps)},
        "full_traj": multi_turn_response,
        "date": current_date
    }
    
    return result_item

# ==================== 主函数 ====================
if __name__ == "__main__":
    
    # 初始化后端
    llm_kwargs = {}
    model = None
    processor = None
    client = None
    
    if args.backend == 'vllm':
        client = OpenAI(
            base_url="http://localhost:8001/v1",
            api_key="EMPTY",
        )
        llm_kwargs = {
            'client': client,
            'model_name': args.model_name_or_path,
            'top_p': args.top_p,
            'temperature': args.temperature
        }
        
        # 测试连接
        logger.info("Testing VLLM connection...")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_ins.replace("{Question}", "测试问题").replace("{Image_url}", "http://example.com/test.jpg")},
        ]
        response, flag = call_vllm(messages, client, args.model_name_or_path)
        if flag:
            logger.info("VLLM connection test passed.")
        else:
            logger.error("VLLM connection test failed!")
            
    elif args.backend == 'transformers':
        logger.info(f"Loading model from {args.model_name_or_path}...")
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            args.model_name_or_path,
            torch_dtype="auto",
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        llm_kwargs = {
            'model': model,
            'processor': processor
        }
        logger.info("Model loaded successfully.")
        
    elif args.backend in ['yuanshen', 'idealab']:
        logger.info(f"Using {args.backend} API backend")
        llm_kwargs = {'model': 'o3-0416-global'}
    
    # 初始化 ToolCallWrapper
    tool_wrapper = ToolCallWrapper()
    
    # 加载数据
    logger.info(f"Loading data from {args.data}...")
    data = load_jsonl(args.data)
    logger.info(f"Loaded {len(data)} items")
    
    # 设置输出路径
    cur_time = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = args.data.replace('.jsonl', f'_{cur_time}_output.jsonl')
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # 去重处理
    processed_questions = set()
    if os.path.exists(save_path) and args.new_file:
        os.remove(save_path)
        logger.info(f"Removed existing file: {save_path}")
    
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    final_resp = record.get("final_response", "").strip()
                    num_tool_calls = record.get("num_tool_calls", 0)
                    
                    if final_resp and num_tool_calls >= 1:
                        processed_questions.add(record["question"])
                except Exception:
                    continue
        
        logger.info(f"Found {len(processed_questions)} already processed questions")
    
    # 过滤已处理的数据
    data = [d for d in data if d.get('question') not in processed_questions]
    logger.info(f"Processing {len(data)} remaining items")
    
    # 线程锁
    lock = threading.Lock()
    
    def process_and_save(item):
        """处理单个item并保存"""
        try:
            result = process_single_item(item, args, llm_kwargs, tool_wrapper)
            
            if result is None:
                logger.warning(f"Failed to process item: {item.get('question', 'Unknown')}")
                return
            
            # 保存结果
            with lock:
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    
        except Exception as e:
            logger.error(f"Error processing item: {item.get('question', 'Unknown')}")
            logger.error(f"Exception: {e}")
            logger.error(traceback.format_exc())
    
    # 执行处理
    if args.sequential:
        logger.info("Running in sequential mode...")
        for item in tqdm(data):
            process_and_save(item)
    else:
        logger.info(f"Running with {args.max_workers} workers...")
        executor = ThreadPoolExecutor(max_workers=args.max_workers)
        try:
            futures = [executor.submit(process_and_save, d) for d in data]
            pending = set(futures)
            pbar = tqdm(total=len(futures))
            last_progress = time.time()
            
            while pending:
                done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                if done:
                    for fut in done:
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error(f"Future error: {e}")
                    pbar.update(len(done))
                    last_progress = time.time()
                    
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()
    
    logger.info(f"Processing complete. Results saved to {save_path}")