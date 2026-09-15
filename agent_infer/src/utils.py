import json
import logging
import re
from pprint import pprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_jsonl(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            logging.info(f"Loaded {len(data)} from '{file_path}'.")
            return data
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON in file '{file_path}'.")
    except Exception as e:
        logging.error(f"An error occurred while loading JSONL file: {e}")
    return []



_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?```", re.DOTALL)

def loads_llm_json(text):
    """解析 LLM 输出的 JSON，兼容 ```json ... ``` 包裹、前后带说明文字等情况。"""
    if not isinstance(text, str):
        return text
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _CODE_FENCE_RE.search(s)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            s = inner
    # 最后兜底：取最外层的 [...] 或 {...}
    for open_c, close_c in (('[', ']'), ('{', '}')):
        start, end = s.find(open_c), s.rfind(close_c)
        if 0 <= start < end:
            return json.loads(s[start:end + 1])
    raise json.JSONDecodeError("No JSON content found", text, 0)



def text_wrap(text:str):
    return {"type": "text", "text": text}
def image_wrap(path:str):
    return {"type": "image_url", "image_url": {"url": path}}

def extract_answer(text: str) -> str:
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text, re.DOTALL)

    cleaned_matches = [match.strip() for match in matches]
    if len(cleaned_matches) == 0:
        logging.warning(f"<answer> tag detected. No answer found: {text}")
        return text
    return '\n'.join(cleaned_matches)

def extract_think(text: str) -> str:
    """
    前提：一定有<think>或</think>
    提取文本中的思考内容，支持多种格式匹配
    优先级：
    1. 完整匹配 <think>...</think>
    2. 只匹配 </think>，提取前面的内容
    3. 只匹配 <think>，提取后面的内容
    """
    # 情况1：尝试完整匹配 <think>...</think>
    pattern_full = r"<think>(.*?)</think>"
    matches = re.findall(pattern_full, text, re.DOTALL)
    
    if matches:
        cleaned_matches = [match.strip() for match in matches]
        return '\n'.join(cleaned_matches)
    
    # 情况2：尝试匹配 </think>，提取前面的内容
    if "</think>" in text:
        parts = text.split("</think>", 1)
        think_content = parts[0]
        # 检查是否有 <think> 开头
        if "<think>" in think_content:
            think_content = think_content.split("<think>", 1)[-1]
        return think_content.strip()
    
    # 情况3：尝试匹配 <think>，提取后面的所有内容
    if "<think>" in text:
        parts = text.split("<think>", 1)
        return parts[-1].strip()
    
    # 没有任何匹配
    logging.warning(f"No think tag pattern found in text: {text[:100]}...")
    return text

def extract_tool_call(text: str) -> list:
    # Returns a list of tool calls(dict)

    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    try:
        tool_calls = []
        for match in matches:
            tool_call_dict = json.loads(match.strip())
            if not isinstance(tool_call_dict, dict):
                logging.warning(f"Invalid tool call JSON: {match}")
                continue
            tool_calls.append(tool_call_dict)
        return tool_calls
    except json.JSONDecodeError:
        error_text = f"Error decoding JSON in <tool_call>: {text}"
        logging.error(error_text)
    except Exception as e:
        error_text = f"An error occurred while extract <tool_call>: {e}.\n Current content: {text}"
        logging.error(error_text)

    return error_text

def test_infer(r: int):
    predefined_output = [
        "<think>Code</think><tool_call>{\"name\": \"code_interpreter\", \"arguments\": {\"code\": \"print('hello world')\"}}</tool_call>",
    ]*127
    predefined_output += [
        # "<think>I should call search engine to find relevant articles</think><tool_call>{\"name\": \"search\", \"arguments\": {\"query\": \"Beijing\"}}</tool_call>}",
        "<answer>答案,yes!</answer>",
    ]

    return predefined_output[r]


import base64
import mimetypes
def image_to_base64(file_path):
    """
    将图片文件转换为完整格式的base64字符串
    
    Args:
        file_path: 图片文件路径
    
    Returns:
        str: 完整的data URL格式base64字符串
    """

    if file_path.startswith('file://'):
        file_path = file_path.replace('file://', '')

    # 获取文件MIME类型
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        # 如果无法识别，默认使用image/jpeg
        mime_type = "image/jpeg"
    
    # 读取文件并编码
    with open(file_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 构建完整data URL格式
    return f"data:{mime_type};base64,{base64_data}"


if __name__ == '__main__':
    text = "xxx<answer>这是第一个答案</answer>xxx<think>y</think>"
    pprint(extract_answer(text))

    text = "xxx<think>yyythink</think><tool_call>{\"name\": \"get_current_weather\", \"args\": {\"location\": \"Beijing\", \"format\": \"celsius\"}}</tool_call>xxx"
    pprint(extract_think(text))
    pprint(extract_tool_call(text))