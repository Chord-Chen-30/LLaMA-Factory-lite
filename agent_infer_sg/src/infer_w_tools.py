"""Agentic inference with tools — ported from agent_infer/src/infer_w_tools_mp_test.py.

Same ReAct loop as the original; only the tool implementations changed
(see agent_wrapper.ToolCallWrapper which now dispatches to tools.py).
"""
from openai import OpenAI
import argparse
from tqdm import tqdm
import time
import logging
from pathlib import Path
import os
import json
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from prompt import sys_prompt, prompt_ins3
from utils import (
    load_jsonl, extract_answer, extract_think, extract_tool_call,
    text_wrap, image_wrap,
)
from agent_wrapper import ToolCallWrapper


parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--max_rounds", type=int, default=64)
parser.add_argument("--max_tool_call_num", type=int, default=128)
parser.add_argument("--mp", type=int, default=12)
parser.add_argument(
    "--output_dir", type=str,
    default="./agent_infer_sg/result/",
)
parser.add_argument("--date", type=str, default=datetime.datetime.now().strftime("%Y-%m-%d"))
parser.add_argument("--overwrite", type=int, default=0)
parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8001/v1")
parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
args = parser.parse_args()

logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s - %(levelname)s - %(message)s")

output_path = os.path.join(
    args.output_dir,
    args.date,
    "/".join(Path(args.model_name_or_path).parts[-2:]),
    os.path.basename(args.data).replace(".jsonl", "_w_tools.jsonl"),
)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

finished_lines = 0
if os.path.exists(output_path):
    if args.overwrite:
        os.remove(output_path)
    else:
        with open(output_path, "r") as f:
            finished_lines = len(f.readlines())

client = OpenAI(base_url=args.vllm_base_url, api_key="EMPTY")


def call_client(message, top_p=0.95, temperature=0.6, max_try=10):
    start_time = time.time()
    ret = {"error": "Init placeholder"}
    for _ in range(max_try):
        try:
            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=message,
                stream=False,
                top_p=top_p,
                temperature=temperature,
                max_tokens=2048,
            )
            ret = completion
            break
        except Exception as e:
            logging.error(f"Error calling vLLM: {e}")
            ret = {"error": str(e)}
        time.sleep(0.3)
    logging.info(f"call_client() Time taken: {time.time()-start_time:.2f}s")
    return ret


tool_caller = ToolCallWrapper()


def process_item(i, d, args):
    question = d.get("question") or d.get("prompt") or d.get("origin_question")
    image = d.get("file_path") or d.get("image_url") or d.get("image_path")
    assert question and image, "question or image is None"
    if isinstance(image, list):
        image = image[0]
    if not image.startswith("http"):
        image = f"file://{image}"

    message = [
        {"role": "system", "content": [text_wrap(sys_prompt)]},
        {"role": "user", "content": [
            text_wrap(prompt_ins3.replace("{Question}", question).replace("{ImageUrl}", image)),
            image_wrap(image),
        ]},
    ]

    response_answer = ""
    r = 0
    for r in range(args.max_rounds):
        completion = call_client(message)
        if "error" not in completion:
            output_str = completion.choices[0].message.content
        else:
            output_str = completion["error"]

        if not isinstance(output_str, str):
            logging.warning(f"output_str not a string: {output_str}")
            continue

        assistant_str = ""
        user_str = "Your output format is incorrect. Please use <think>, <tool_call>, <answer> tags to wrap your output."

        if ("<think>" not in output_str) and ("<tool_call>" not in output_str) and ("<answer>" not in output_str):
            assistant_str = f"<think>{output_str}</think>"
            user_str = "You do not invoke any tool. Continue thinking or answering the question. Remember to use <think></think> or <answer></answer> to wrap your response."

        if ("<think>" in output_str) or ("</think>" in output_str):
            response_think = extract_think(output_str)
            assistant_str += f"<think>{response_think}</think>"

        if "<tool_call>" in output_str:
            tool_calls = extract_tool_call(output_str)
            if isinstance(tool_calls, list):
                assistant_str += "<tool_call>"
                assistant_str += "".join(json.dumps(t) for t in tool_calls)
                assistant_str += "</tool_call>"
                tool_response = ""
                for params in tool_calls:
                    tool_response += f"From: {params.get('name', 'tool_call')}\n"
                    tool_response += tool_caller.call_tools(params)
                user_str = f"<tool_response>{tool_response}</tool_response>"
            else:
                assistant_str = output_str
                user_str = tool_calls  # error string

        if "<answer>" in output_str:
            response_answer = extract_answer(output_str)
            message.append({"role": "assistant", "content": f"{assistant_str}<answer>{response_answer}</answer>"})
            break

        message.append({"role": "assistant", "content": [text_wrap(assistant_str)]})
        message.append({"role": "user", "content": [text_wrap(user_str)]})

    if r == (args.max_rounds - 1) and not response_answer:
        logging.info("Max rounds reached. Forcing an answer.")
        message[-1]["content"][0]["text"] += "\nYou have reached the maximum number of rounds and tool calls. Give an answer surrounded by <answer></answer> now."
        completion = call_client(message)
        if "error" not in completion:
            output_str = completion.choices[0].message.content
        else:
            output_str = completion["error"]
            logging.error(f"{output_str}")

        if "<answer>" in output_str:
            response_answer = extract_answer(output_str)
            message.append({"role": "assistant", "content": [text_wrap(f"<answer>{response_answer}</answer>")]})
        else:
            logging.warning("Max rounds, no <answer>. Putting last output as answer.")
            message.append({"role": "assistant", "content": [text_wrap(f"<answer>{output_str}</answer>")]})

    d["traj"] = message
    d["response_answer"] = response_answer
    return i, d


def main():
    data = load_jsonl(args.data)
    with (ThreadPoolExecutor(max_workers=args.mp) as executor, open(output_path, "a", buffering=1) as g):
        futures = {
            executor.submit(process_item, i + finished_lines, d, args): i + finished_lines
            for i, d in enumerate(data[finished_lines:])
        }
        pbar = tqdm(total=len(futures), ncols=100, desc="Inference with tools")
        next_idx = finished_lines
        cache = {}
        for fut in as_completed(futures):
            idx, new_data = fut.result()
            cache[idx] = new_data
            while next_idx in cache:
                g.write(json.dumps(cache.pop(next_idx), ensure_ascii=False) + "\n")
                next_idx += 1
                pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
