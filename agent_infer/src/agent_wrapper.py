# Author: Chen Zhuo
# Date: 2025-11-03
# Feature: Do not depend on qwen_agent
# Tools: Search(text to text), Visit, CodeInterpreter, Image Search (image to image)

from tool_search import SearchTool
from tool_visit import Visit
from tool_code import PythonInterpreter
from tool_image_search import ImageSearcher
import re
import logging
import json
import time

logger = logging.getLogger(__name__)

def find_and_extract_wrapped_content(s: str):
    pattern = r'<(tools|tool_call)>(.*?)<\/\1>'
    match = re.search(pattern, s, re.DOTALL)
    
    if match:
        return match.group(2)
    return None


class ToolCallWrapper:
    def __init__(self):
        self.search_tool = SearchTool()
        self.visit = Visit()
        self.python_interpreter = PythonInterpreter()
        self.image_searcher = ImageSearcher()

    def call_tools(self, params):
        start_time = time.time()
        if isinstance(params, str):
            try:
                params = find_and_extract_wrapped_content(params)
                params = json.loads(params)
            except json.JSONDecodeError:
                return "Invalid JSON"
            except Exception as e:
                logger.error(f"Error processing <tool_call> content: {e}")
                logger.error(f"params: {params}")
                return "Error processing <tool_call> content"
        
        tool_name = params.get("name") or params.get("tool_name")
        tool_params = params.get("arguments") or params.get("parameters")

        if not tool_name:
            return "\"name\" field is required. Specify a tool name"

        if tool_name.lower() in ['web_search', 'search']:
            ret = self.search_tool.call(tool_params)
        elif tool_name.lower() in ['visit']:
            ret = self.visit.call(tool_params)
        elif tool_name.lower() in ['code_interpreter', 'pythoninterpreter']:
            ret = self.python_interpreter.call(tool_params)
        elif tool_name.lower() in ['image_search', 'vlsearchimage']:
            ret = self.image_searcher.call(tool_params)
        else:
            ret = f"Invalid tool name {tool_name}. Available tools: web_search, visit, code_interpreter, image_search"
        
        logger.info(f"call_tools({tool_name}) took {time.time() - start_time:.2f} seconds")
        logger.info(f"  - Return: {str(ret)[:100]}")
        return ret


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    agent = ToolCallWrapper()

    # params = {
    #     "name": "web_search",
    #     "parameters": {
    #         "queries": "What is the capital of France?"
    #     }
    # }

    # params = {
    #     "name": "VLSearchImage",
    #     "parameters": {
    #         "image_urls": "./agent_infer/downloaded_images/sc.jpg"
    #     }
    # }

    params = {
        "name": "visit",
        "parameters":{
            "url": ["https://chord-chen-30.github.io/"],
            "goal": "Give 1-2 sentence TLDR of the webpage"\
        }
    }

    test_code = """
import sympy as sp
X=sp.symbols('X')
poly_factor = (X**2 - (sp.sqrt(34)+sp.sqrt(14))*X + 2*sp.sqrt(119))*(X**2 - 2*(sp.sqrt(11)+sp.sqrt(6))*X + 4*sp.sqrt(66))
poly_original = X**4 - sp.sqrt(34)*X**3 - sp.sqrt(14)*X**3 - 2*sp.sqrt(11)*X**3 - 2*sp.sqrt(6)*X**3 + 2*sp.sqrt(374)*X**2 + 2*sp.sqrt(154)*X**2 + 2*sp.sqrt(119)*X**2 + 4*sp.sqrt(66)*X**2 + 4*sp.sqrt(51)*X**2 + 4*sp.sqrt(21)*X**2 - 4*sp.sqrt(1309)*X - 4*sp.sqrt(714)*X - 8*sp.sqrt(561)*X - 8*sp.sqrt(231)*X + 8*sp.sqrt(7854)
# 由于浮点精度问题，直接比较可能失败，使用simplify来验证
is_match = sp.simplify(poly_factor - poly_original) == 0
print(f'Expanded factor matches original polynomial? {is_match}')
"""
    params = {
        "name": "code_interpreter",
        "parameters":{"code": test_code}
    }

    result = agent.call_tools(params)
    print(result)