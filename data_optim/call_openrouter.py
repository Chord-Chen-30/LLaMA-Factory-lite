import argparse
import json
from openai import OpenAI
import os
import time
import requests
import datetime
from tqdm import tqdm

import sys
sys.path.append("../agent_infer/src")
from o3_prompt import sys_prompt, prompt_ins, tools_list, prompt_ins3
from utils import text_wrap, image_wrap
from utils import load_jsonl, extract_answer, extract_think, extract_tool_call
from utils import image_to_base64

import logging

from dotenv import load_dotenv
# Load .env from this module's directory so callers in other cwds still pick it up.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY', None)
if not OPENROUTER_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in environment variables")
NEW_API_KEY = os.getenv('NEW_API', None)

# Optional: override the URL for the cheap key; defaults to the same endpoint.
NEW_API_URL = os.getenv('NEW_API_URL', "https://openrouter.ai/api/v1/chat/completions")
OPENROUTER_URL = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")

# Model names differ by endpoint: NEW_API uses the bare name, OPENROUTER uses the prefixed name.
# Other choices: [qwen3-235b-a22b-thinking-2507]
NEW_API_MODEL = os.getenv('NEW_API_MODEL', "gpt-5.4")

# Other choices: [qwen/qwen3-235b-a22b-thinking-2507]
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', "openai/gpt-5.4")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _post_chat(url, key, data, max_retries, label):
    """Try POST to a chat-completions endpoint with retries. Returns the parsed JSON on success,
    or raises the last Exception on failure."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_exc = None
    for attempt in range(max_retries):
        response_raw = None
        try:
            response_raw = requests.post(url, headers=headers, json=data, timeout=120)
            response_raw.raise_for_status()
            return response_raw.json()
        except Exception as e:
            last_exc = e
            logging.warning(f"{label} API call error! Attempt {attempt + 1}/{max_retries}: {e}")
            if response_raw is not None:
                logging.error(f"{response_raw}")
            if attempt < max_retries - 1:
                time.sleep(1)
            time.sleep(0.3)
    raise last_exc if last_exc else RuntimeError(f"{label} failed with no exception")


def call_openrouter(text, image_url=None, model=None,
                    max_retries=24, cheap_retries=10):
    """Call the chat-completions API.

    Strategy: if NEW_API key is configured, try it first for `cheap_retries` attempts (cheap key).
    Fall back to OPENROUTER_API_KEY for `max_retries` attempts. The caller sees the same return
    shape as before: the full JSON response on success, or `{"error": "..."}` on total failure.

    `model` is accepted for backward compatibility but ignored — each endpoint uses its own
    configured model constant (NEW_API_MODEL / OPENROUTER_MODEL) since their naming differs.
    """
    messages = [
        {"role": "system", "content": [text_wrap(sys_prompt)]},
        {"role": "user", "content": [
            text_wrap(text),
        ]}
    ]

    # Phase 1: cheap key, if available
    if NEW_API_KEY:
        data = {"model": NEW_API_MODEL, "stream": False, "messages": messages}
        try:
            return _post_chat(NEW_API_URL, NEW_API_KEY, data, cheap_retries, label="NEW_API")
        except Exception as e:
            logging.warning(f"NEW_API exhausted after {cheap_retries} retries, falling back to OPENROUTER: {e}")

    # Phase 2: fallback to primary key
    data = {"model": OPENROUTER_MODEL, "stream": False, "messages": messages}
    try:
        return _post_chat(OPENROUTER_URL, OPENROUTER_KEY, data, max_retries, label="OPENROUTER")
    except Exception as e:
        logging.error(f"OPENROUTER fallback failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    print(call_openrouter("Who are you?", ))
    exit()


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
                image_wrap(image_base64),
            ]},
        ]

        response_answer = ''

        for r in range(args.max_rounds):
            completion = call_openrouter(message, model=args.model_name_openrouter)
            if 'error' not in completion:
                output_str = completion['choices'][0]['message']['content']
            else:
                output_str = completion['error']

            if not isinstance(output_str, str):
                logging.warning(f"============\noutput_str is not of type string: {output_str}\n============")
                continue

            assistant_str = ""
            user_str = "Your output format is incorrect. Please use <think></think>, <tool_call></tool_call>, <answer></answer> tags to wrap your output."

            if ('<think>' not in output_str) and ('<tool_call>' not in output_str) and ('<answer>' not in output_str):
                assistant_str = f"<think>{output_str}</think>"
                user_str = f"You do not invoke any tool. Continue thinking or answering the question. Remember to use <think></think> or <answer></answer> to wrap your response."

            if ('<think>' in output_str) or ('</think>' in output_str):
                response_think = extract_think(output_str)
                assistant_str += f"<think>{response_think}</think>"

            if '<tool_call>' in output_str:
                response_tool_call_list = extract_tool_call(output_str)

                if isinstance(response_tool_call_list, list):
                    assistant_str += "<tool_call>"
                    assistant_str += "".join([json.dumps(t) for t in response_tool_call_list])
                    assistant_str += "</tool_call>"

                    tool_response = ''
                    for tool_call_params in response_tool_call_list:
                        tool_response += f"From: {tool_call_params.get('name', 'tool_call')}\n"
                        tool_response += tool_caller.call_tools(tool_call_params)

                    user_str = f"<tool_response>{tool_response}</tool_response>"

                else:
                    assistant_str = output_str
                    user_str = response_tool_call_list

            if '<answer>' in output_str:
                response_answer = extract_answer(output_str)
                message.append({"role": "assistant", "content": f"{assistant_str}<answer>{response_answer}</answer>"})
                break

            message.append({"role": "assistant", "content": [text_wrap(assistant_str)]})
            message.append({"role": "user", "content": [text_wrap(user_str)]})

        if r == (args.max_rounds-1):
            logging.info("Maximum number of rounds reached. Force an answer.")
            message[-1]['content'][0]['text'] += "\nYou have reached the maximum number of rounds and tool calls. Give an answer surrounded by <answer></answer> now."

            completion = call_openrouter(message)
            if 'error' not in completion:
                output_str = completion['choices'][0]['message']['content']
            else:
                output_str = completion['error']
                logging.error(f"{output_str}")

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

        futures = {
            executor.submit(process_item, i + finished_lines, d, args):
            i + finished_lines for i, d in enumerate(data[finished_lines:])
        }

        pbar = tqdm(total=len(futures), ncols=50)

        future_to_idx = {future: idx for future, idx in futures.items()}
        next_idx_to_write = finished_lines

        result_cache = {}

        for future in as_completed(futures):
            idx, new_data = future.result()
            result_cache[idx] = new_data

            while next_idx_to_write in result_cache:
                g.write(json.dumps(result_cache.pop(next_idx_to_write),
                                ensure_ascii=False) + '\n')
                next_idx_to_write += 1
                finished_count += 1
                pbar.update(1)

        pbar.close()
