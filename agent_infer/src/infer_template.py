from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from openai import OpenAI
import argparse
from prompt import sys_prompt, prompt_ins, prompt_ins2



parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='Qwen/Qwen3-VL-30B-A3B-Thinking')
args = parser.parse_args()


breakpoint()

client = OpenAI(
    base_url="http://localhost:8001/v1",
    api_key="EMPTY",
)


messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": prompt_ins.replace("{Question}", "这是哪里，你可以使用相应搜索工具。").replace("{Image_url}", "https://example.com/example.jpg")},
]

completion = client.chat.completions.create(
    model=args.model_name_or_path,
    messages=messages,
    stream=False,
    top_p=0.95,
    temperature=0.6
)

print(completion.choices[0].message)
