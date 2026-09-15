from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from openai import OpenAI
import argparse
from prompt import sys_prompt, prompt_ins, prompt_ins2
import logging
from utils import load_jsonl, extract_answer, extract_think, extract_tool_call
from agent_wrapper import ToolCallWrapper
import os
import json
from utils import test_infer
from urllib.parse import urlparse, unquote

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-VL-30B-A3B-Thinking')
parser.add_argument('--data', type=str, default='./data/c4_demo.jsonl')
parser.add_argument('--max_rounds', type=int, default=128)
parser.add_argument('--max_tool_call_num', type=str, default=128)

parser.add_argument('--output_dir', type=str, default='./agent_infer/result/')
parser.add_argument('--overwrite', type=int, default=1)
args = parser.parse_args()

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)

def call_client(message, top_p=0.95, temperature=0.6, max_try=10):
    for i in range(max_try):
        try:
            completion = client.chat.completions.create(
                model=args.model_name_or_path,
                messages=message,
                stream=False,
                top_p=top_p,
                temperature=temperature
            )
            return completion
        except Exception as e:
            logging.error(f"Error calling OpenAI API: {e}")
            logging.error(completion)
    return False

tool_caller = ToolCallWrapper()

def test_case():
    message = [
        # {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_ins2.replace("{Question}", "who is he")},
            {"type": "image_url", "image_url": {"url": os.getenv("EXAMPLE_IMAGE_URL", "https://example.com/example.jpg")}}
        ]}
    ]
    completion = call_client(message)

    if completion:
        logging.info(f"Test passed. Output:\n{completion.choices[0].message}")
    else:
        logging.error(f"Test failed.")

test_case()

# data = load_jsonl(args.data)

# with open(output_path, 'a') as g:
#     for d in data[finished_lines:]:
#         question = d.get('question') or d.get('prompt')
#         image = d.get('image_url') or d.get('file_path')

#         parsed_result = urlparse(image)
#         encoded_path = parsed_result.path
#         decoded_path = unquote(encoded_path)

#         image = os.path.join('./images', os.path.basename(decoded_path))
        
#         assert question and image, print("question or image_url is None")
        
#         message = [
#             {"role": "system", "content": [{"type": "text", "text": sys_prompt},]},
#             {"role": "user", "content": [
#                 {"type": "text", "text": prompt_ins2.replace("{Question}", question)},
#                 {"type": "image", "image": image},
#             ]},
#         ]

#         # message = [{"role": "system", "content": sys_prompt},{"role": "user", "content": "Dummy"}]
        
#         breakpoint()

#         response_answer = ''
#         for r in range(args.max_rounds):
#             completion = call_client(message)
#             if completion:
#                 output_str = completion.choices[0].message.content
#             else:
#                 output_str = "Error: Failed to call OpenAI API"

#             # output_str = test_infer(r)

#             assistant_str = ""
#             user_str = ""

#             if '<think>' in output_str:
#                 response_think = extract_think(output_str)
#                 assistant_str += f"<think>{response_think}</think>"
            
#             if '<tool_call>' in output_str:
#                 response_tool_call_list = extract_tool_call(output_str)

#                 # Successfully extract tool calls
#                 if isinstance(response_tool_call_list, list):
#                     assistant_str += "<tool_call>"
#                     assistant_str += "".join([json.dumps(t) for t in response_tool_call_list])
#                     assistant_str += "</tool_call>"

#                     tool_response = ''
#                     for tool_call_params in response_tool_call_list:
#                         tool_response += f"From: {tool_call_params['name']}\n"
#                         tool_response += tool_caller.call_tools(tool_call_params)
                    
#                     user_str = f"<tool_response>{tool_response}</tool_response>"

#                 # Failed to extract tool calls. response_tool_call_list is error string.
#                 else:
#                     assistant_str = output_str
#                     user_str = response_tool_call_list

#             if '<answer>' in output_str:
#                 response_answer = extract_answer(output_str)
#                 message.append({"role": "assistant", "content": f"{assistant_str}<answer>{response_answer}</answer>"})
#                 break # Break conversation rounds

#             message.append({"role": "assistant", "content": assistant_str})
#             message.append({"role": "user", "content": user_str})                    
        
#         # Reached max round. Force an answer.
#         if r == (args.max_rounds-1):
#             logging.info("Maximum number of rounds reached. Force an answer.")
#             message[-1]['content'][0]['text'] += "\nYou have reached the maximum number of rounds and tool calls. Give an answer surrounded by <answer></answer> now."

#             completion = call_client(message)
#             if completion:
#                 output_str = completion.choices[0].message.content
#             else:
#                 output_str = "Error: Failed to call OpenAI API"
#                 logging.error(f"{completion}")
            
#             # output_str = test_infer(r)

#             if '<answer>' in output_str:
#                 response_answer = extract_answer(output_str)
#                 message.append({"role": "assistant", "content": f"<answer>{response_answer}</answer>"})
#             else:
#                 logging.warning("Maximum number of rounds reached. No <answer> tag found. Put last output as answer.")
#                 message.append({"role": "assistant", "content": f"<answer>{output_str}</answer>"})

#         d['traj'] = message
#         d['response_answer'] = response_answer
#         g.write(json.dumps(d, ensure_ascii=False) + '\n')

