import re
import os
import random
import time
import json5
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import Timeout
import logging

logger = logging.getLogger(__name__)

# 模拟 sandbox_fusion 的依赖，以便在没有安装该库的情况下也能运行和理解代码结构
# 在实际环境中，请确保已安装 sandbox_fusion: pip install sandbox_fusion
try:
    from sandbox_fusion import run_code, RunCodeRequest, RunStatus
except ImportError:
    print("Error: 'sandbox_fusion' not found. Run pip install sandbox_fusion")
    exit(-1)


SANDBOX_FUSION_ENDPOINTS = [
    e.strip() for e in os.getenv("SANDBOX_FUSION_ENDPOINT", "http://127.0.0.1:8080").split(",") if e.strip()
]


class PythonInterpreter:
    """
    一个独立的Python代码执行器，通过远程沙箱环境运行代码。
    它支持从多个端点中随机选择并进行重试。
    """

    def __init__(self):
        # Test
        logger.info(f"---- Testing PythonInterpreter ----")
        params = {"code": "print('hello, world!')"}
        logger.info(self.call(params))
        logger.info(f"Python Interpreter 初始化完成，请检查上述结果。")
        logger.info(f"-----------------------------------")


    def _extract_code_from_input(self, params: Union[str, dict]) -> str:
        """从不同格式的输入中提取纯代码字符串。"""
        code = ""
        try:
            # 如果输入是字符串，尝试将其作为JSON解析，否则视为原始代码
            if isinstance(params, str):
                try:
                    params_dict = json5.loads(params)
                    code = params_dict.get('code', '')
                except Exception:
                    code = params

            # 如果输入是字典，直接获取'code'键的值
            elif isinstance(params, dict):
                code = params.get('code', '')

            # 检查代码中是否包含Markdown代码块，并提取其中的内容
            # e.g., ```python\nprint("hello")\n```
            markdown_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if markdown_match:
                code = markdown_match.group(1)

        except Exception as e:
            print(f"Error parsing code input: {e}")
            if isinstance(params, str):
                return params.strip() # 最终回退
        
        return code.strip()

    def call(self, params: Union[str, dict], timeout: int = 50, max_retries: int = 5) -> str:
        """
        执行Python代码，支持自动重试和端点故障转移。

        Args:
            params (Union[str, dict]): 包含待执行代码的字典 (e.g., {'code': 'print(1+1)'}) 或纯代码字符串。
            timeout (int): 每次尝试的客户端超时时间（秒）。
            max_retries (int): 最大尝试次数。

        Returns:
            str: 代码执行的stdout和stderr结果，或错误信息。
        """
        code_to_run = self._extract_code_from_input(params)

        if not code_to_run:
            return '[PythonInterpreter Error]: Code is empty after extraction.'
        
        if not SANDBOX_FUSION_ENDPOINTS:
            return '[PythonInterpreter Error]: No sandbox endpoints configured.'

        last_error = None
        for attempt in range(max_retries):
            # 随机选择一个端点进行尝试
            endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
            print(f"Attempt {attempt + 1}/{max_retries} using endpoint: {endpoint}")
            
            try:
                # 调用沙箱执行代码
                request = RunCodeRequest(code=code_to_run, language='python', run_timeout=timeout)
                code_result = run_code(request, max_attempts=1, client_timeout=timeout, endpoint=endpoint)

                # 格式化输出
                result_parts = []
                if code_result.run_result.stdout:
                    result_parts.append(f"stdout:\n{code_result.run_result.stdout}")
                if code_result.run_result.stderr:
                    result_parts.append(f"stderr:\n{code_result.run_result.stderr}")
                
                result = '\n'.join(result_parts)
                print('✅ Execution successful.')
                # 如果没有输出，返回一个标准成功消息
                return result if result.strip() else 'Finished execution with no output.'

            except Timeout:
                last_error = f'[PythonInterpreter Error] TimeoutError on endpoint {endpoint}.'
                print(f"Timeout on attempt {attempt + 1}: {last_error}")
                continue # 继续下一次尝试
            
            except Exception as e:
                last_error = f'[PythonInterpreter Error] on endpoint {endpoint}: {str(e)}'
                print(f"Error on attempt {attempt + 1}: {last_error}")
                continue # 继续下一次尝试

        return last_error or '[PythonInterpreter Error]: All retry attempts failed.'

    def execute_on_endpoint(self, params: Union[str, dict], endpoint: str, timeout: int = 30) -> tuple:
        """
        在指定的单个端点上执行代码，用于测试。

        Args:
            params (Union[str, dict]): 包含代码的输入。
            endpoint (str): 要测试的特定端点URL。
            timeout (int): 超时时间（秒）。

        Returns:
            tuple: (success: bool, result: str, execution_time: float | None)
        """
        code_to_run = self._extract_code_from_input(params)

        if not code_to_run:
            return False, '[PythonInterpreter Error]: Code is empty.', None

        start_time = time.time()
        try:
            request = RunCodeRequest(code=code_to_run, language='python', run_timeout=timeout)
            code_result = run_code(request, max_attempts=1, client_timeout=timeout, endpoint=endpoint)
            end_time = time.time()
            
            result_parts = []
            if code_result.run_result.stdout:
                result_parts.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result_parts.append(f"stderr:\n{code_result.run_result.stderr}")
            
            result = '\n'.join(result_parts)
            execution_time = end_time - start_time
            output = result.strip() if result.strip() else 'Finished execution with no output.'
            return True, output, execution_time

        except Timeout:
            return False, '[PythonInterpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[PythonInterpreter Error]: {str(e)}', None


# --- 测试功能区 ---
def test_single_endpoint(endpoint: str, test_cases: List[dict], timeout: int = 30) -> dict:
    """使用多个测试用例测试单个端点。"""
    executor = PythonInterpreter()
    results = {
        'endpoint': endpoint, 'status': 'unknown', 'passed_tests': 0,
        'total_tests': len(test_cases), 'test_results': [],
        'avg_execution_time': 0, 'errors': []
    }
    execution_times = []
    
    print(f"\n🧪 Testing endpoint: {endpoint}")
    
    for i, test_case in enumerate(test_cases):
        test_name = test_case['name']
        print(f"  ├─ Running test {i+1}/{len(test_cases)}: {test_name}")
        
        success, result, exec_time = executor.execute_on_endpoint(
            {"code": test_case['code']}, endpoint, timeout
        )
        
        test_result_details = {'name': test_name, 'success': success, 'result': result, 'execution_time': exec_time}
        
        if success and exec_time is not None:
            execution_times.append(exec_time)
            expected_output = test_case.get('expected_output')
            
            if expected_output:
                actual_output = result.replace('stdout:\n', '').strip()
                if expected_output in actual_output:
                    results['passed_tests'] += 1
                    print(f"  │  ✅ PASSED ({exec_time:.2f}s)")
                else:
                    print(f"  │  ❌ OUTPUT MISMATCH ({exec_time:.2f}s)")
                    print(f"  │     Expected to contain: '{expected_output}'")
                    print(f"  │     Got: '{actual_output}'")
            else: # 如果没有预期输出，只要成功运行就算通过
                results['passed_tests'] += 1
                print(f"  │  ✅ PASSED ({exec_time:.2f}s)")
        else:
            print(f"  │  ❌ FAILED: {result}")
            results['errors'].append(f"{test_name}: {result}")
        
        results['test_results'].append(test_result_details)
            
    if execution_times:
        results['avg_execution_time'] = sum(execution_times) / len(execution_times)
    
    if results['passed_tests'] == results['total_tests']:
        results['status'] = 'healthy'
        print(f"  └─ ✅ ALL TESTS PASSED ({results['passed_tests']}/{results['total_tests']})")
    elif results['passed_tests'] > 0:
        results['status'] = 'partial'
        print(f"  └─ ⚠️  PARTIAL SUCCESS ({results['passed_tests']}/{results['total_tests']})")
    else:
        results['status'] = 'failed'
        print(f"  └─ ❌ ALL TESTS FAILED ({results['passed_tests']}/{results['total_tests']})")
    
    return results


def test_all_endpoints_comprehensive():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    """对所有已配置的沙箱端点运行全面的并发测试套件。"""
    test_cases = [
        {'name': 'Basic Math', 'code': 'print(2 + 2)', 'expected_output': '4'},
        {'name': 'String Operations', 'code': 'print("Hello, " + "World!")', 'expected_output': 'Hello, World!'},
        {'name': 'Loop and Conditionals', 'code': 'evens = [i for i in range(5) if i % 2 == 0]; print(f"Evens: {evens}")', 'expected_output': 'Evens: [0, 2, 4]'},
        {'name': 'Import Standard Library', 'code': 'import math; print(f"Pi: {math.pi:.2f}")', 'expected_output': 'Pi: 3.14'},
        {'name': 'Error Handling', 'code': 'try: undefined_variable except NameError: print("Caught NameError")', 'expected_output': 'Caught NameError'},
    ]
    
    print("🚀 Starting comprehensive endpoint testing...")
    print(f"📊 Testing {len(SANDBOX_FUSION_ENDPOINTS)} endpoints with {len(test_cases)} test cases each.")
    print("=" * 80)
    
    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(SANDBOX_FUSION_ENDPOINTS), 8)) as t_executor:
        future_to_endpoint = {
            t_executor.submit(test_single_endpoint, endpoint, test_cases): endpoint 
            for endpoint in SANDBOX_FUSION_ENDPOINTS
        }
        
        for future in as_completed(future_to_endpoint):
            try:
                all_results.append(future.result())
            except Exception as exc:
                print(f'❌ Endpoint {future_to_endpoint[future]} generated an exception during testing: {exc}')

    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    healthy = [r for r in all_results if r['status'] == 'healthy']
    partial = [r for r in all_results if r['status'] == 'partial']
    failed = [r for r in all_results if r['status'] == 'failed']
    
    print(f"✅ Healthy endpoints: {len(healthy)}/{len(all_results)}")
    for r in healthy: print(f"  - {r['endpoint']} (avg: {r['avg_execution_time']:.2f}s)")
        
    print(f"⚠️  Partial endpoints: {len(partial)}/{len(all_results)}")
    for r in partial: print(f"  - {r['endpoint']} ({r['passed_tests']}/{r['total_tests']} passed)")
        
    print(f"❌ Failed endpoints: {len(failed)}/{len(all_results)}")
    for r in failed: print(f"  - {r['endpoint']}")
    
    total_passed = sum(r['passed_tests'] for r in all_results)
    total_possible = sum(r['total_tests'] for r in all_results)
    health_score = (total_passed / total_possible) * 100 if total_possible > 0 else 0
    print(f"\n🏥 OVERALL SYSTEM HEALTH: {health_score:.1f}% ({total_passed}/{total_possible} tests passed)")


if __name__ == '__main__':
    # 1. 运行全面的并发健康检查
    test_all_endpoints_comprehensive()

    # 使用例子
    test_code = """
import sympy as sp
X=sp.symbols('X')
poly_factor = (X**2 - (sp.sqrt(34)+sp.sqrt(14))*X + 2*sp.sqrt(119))*(X**2 - 2*(sp.sqrt(11)+sp.sqrt(6))*X + 4*sp.sqrt(66))
poly_original = X**4 - sp.sqrt(34)*X**3 - sp.sqrt(14)*X**3 - 2*sp.sqrt(11)*X**3 - 2*sp.sqrt(6)*X**3 + 2*sp.sqrt(374)*X**2 + 2*sp.sqrt(154)*X**2 + 2*sp.sqrt(119)*X**2 + 4*sp.sqrt(66)*X**2 + 4*sp.sqrt(51)*X**2 + 4*sp.sqrt(21)*X**2 - 4*sp.sqrt(1309)*X - 4*sp.sqrt(714)*X - 8*sp.sqrt(561)*X - 8*sp.sqrt(231)*X + 8*sp.sqrt(7854)
# 由于浮点精度问题，直接比较可能失败，使用simplify来验证
is_match = sp.simplify(poly_factor - poly_original) == 0
print(f'Expanded factor matches original polynomial? {is_match}')
"""

#     test_code = """
# print('Hello, World!')
# """
    params = {"code": test_code}
    executor = PythonInterpreter()
    result = executor.call(params)
    
    print("\n--- Single Test Result ---")
    print(result)
    print("--------------------------")
