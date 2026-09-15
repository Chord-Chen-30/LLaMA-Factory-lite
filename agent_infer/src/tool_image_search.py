# WARNING: 目前不支持 download=True
# - 貌似因为搜到的image url都做了加密处理。从zhili image search 代码参考，源码处也没有启用下载图片功能。

import os
import requests
import json
import time
import hashlib
from dotenv import load_dotenv
import uuid
import oss2
import logging

logger = logging.getLogger(__name__)

def string_to_uuid(input_string: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, input_string))

def upload_to_oss(image_path):

    load_dotenv()
    accessKeyId = os.getenv('accessKeyId')
    accessKeySecret = os.getenv('accessKeySecret')
    uuid_ = string_to_uuid(image_path)
        
    image_name = f'{uuid_}.jpeg'
    target_path = f'cz/vl_agent_infer_image/{image_name}'
    if image_path.startswith('file://'):
        image_path = image_path.replace('file://', '')

    try:
        image_size = os.path.getsize(image_path)
    except OSError:
        print("OSError: os.path.getsize(image_path)")
        return None

    if image_size <= 1024:
        return None

    endpoint = os.getenv("OSS_ENDPOINT", "https://oss-cn-hangzhou.aliyuncs.com")
    bucket_name = os.getenv("OSS_BUCKET", "")
    if not bucket_name:
        raise RuntimeError("Set OSS_BUCKET")
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.put_object_from_file(target_path, image_path)
    file_url = bucket.sign_url('GET', target_path, 360000)
    # print(f"Uploaded. Image URL: {file_url}")
    return file_url


class ImageSearcher:
    """
    一个独立的图像搜索引擎，通过指定的API实现以图搜图功能，
    并可选择将搜索到的图片下载到本地。
    """
    def __init__(self):
        """
        初始化ImageSearcher，并从环境变量加载API密钥。
        """
        load_dotenv()
        
        self.api_key = os.getenv('IMG_SEARCH_KEY')
        if not self.api_key:
            raise ValueError("错误：环境变量 'IMG_SEARCH_KEY' 未设置。请在 .env 文件或系统中定义它。")
            
        self.api_url = os.getenv("IMAGE_SEARCH_API_URL", "")

        # Test
        logger.info(f"---- Testing ImageSearcher ----")
        _res = self.call({"image_urls": "file://./agent_infer/downloaded_images/sc.jpg"})
        if _res.startswith("[Error]"):
            logger.error(f"\n--- Testing ImageSearcher Failed: {_res} ---")
            exit(-1)
        else:
            logger.info(f"ImageSearcher 初始化完成. Output:\n{_res[:300]}...")
            logger.info("------------------------------")

    def _send_request(self, image_url: str, retry_attempts: int = 10, timeout: int = 30):
        headers = {
            "X-AK": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "extendParams": {"url": image_url},
            "platformInput": {"model": "google-search"}
        }

        for attempt in range(retry_attempts):
            try:
                # logger.info(f"正在发送图像搜索请求 (尝试 {attempt + 1}/{retry_attempts})... URL: {image_url}")
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()

                response_data = response.json()
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)

                if response_data.get("data") is None:
                    return f"[Error]: data is None. Response: {json.dumps(response_data, ensure_ascii=False)}"

                docs = response_data.get("data", {}).get("originalOutput", {}).get("organic", [])
                
                search_results = [
                    {
                        "image_path": item.get("thumbnailUrl", ""),
                        "snippet": item.get("title", ""),
                        "url": item.get("link", ""),
                    }
                    for item in docs
                ]
                
                return search_results[:5]

            except requests.exceptions.Timeout:
                logger.warning(f"请求超时 (尝试 {attempt + 1}/{retry_attempts}) for URL: {image_url}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{retry_attempts}): {e}")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"解析响应失败: {e}. 响应内容: {response.text if 'response' in locals() else 'N/A'}")
                break
            except Exception as e:
                logger.warning(f"错误: {e}")
                return f"[Error]: {str(e)}"
            
            time.sleep(1)
        
        logger.error(f"所有搜索尝试均失败 for image: {image_url}。请检查log")
        return f"[Error] 所有搜索尝试均失败 for image: {image_url}。"

    def _download_and_save_image(self, image_url: str, save_path: str) -> str | None:
        """
        下载单个图片并保存到指定路径。

        Args:
            image_url (str): 要下载的图片的URL。
            save_path (str): 保存图片的目录。

        Returns:
            str | None: 如果成功，返回保存后的本地文件路径；否则返回 None。
        """
        if not image_url:
            return None
        
        try:
            # 使用URL的MD5哈希生成唯一且安全的文件名，保留.jpg后缀
            filename = f"{hashlib.md5(image_url.encode()).hexdigest()}.jpg"
            local_filepath = os.path.join(save_path, filename)

            # 下载图片
            response = requests.get(image_url, timeout=20, stream=True)
            response.raise_for_status()

            # 以二进制写模式保存文件
            with open(local_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"图片成功下载并保存到: {local_filepath}")
            return local_filepath

        except requests.exceptions.RequestException as e:
            logger.warning(f"下载图片失败: {image_url}，原因: {e}")
            return None
        except IOError as e:
            logger.error(f"保存文件失败: {local_filepath}，原因: {e}")
            return None

    def call(self, params: dict, download_images: bool = False, save_path: str = '/tmp/') -> str:
        """
        执行图像搜索，并可选择下载结果图片到本地。

        Args:
            params (dict): {"image(s)/image_url(s)": ...}。
            download_images (bool): 是否下载搜索到的图片。默认为 False。
            save_path (str): 图片的本地保存目录。默认为 '/tmp/'。

        Returns:
            str: 格式化后的搜索结果字符串。
        """
        image_url = params.get('images', False) or params.get('image', False) or params.get('image_urls', False) or params.get('image_url', False)

        if not image_url:
            return "[Error] [Image Search] 错误: 请将正确图片路径作为参数 image_url(s)/image(s) 传入。"
        
        if isinstance(image_url, list):
            image_url = image_url[0]
        
        if not image_url.startswith('http'):
            image_url = upload_to_oss(image_url)
            if image_url is None:
                return f"[Error] 上传oss失败：{image_url}。导致ImageSearch失败，检查图片路径！"

        # 1. 调用API获取原始搜索结果
        search_results = self._send_request(image_url)
        if not search_results:
            return "[Error] [Image Search] 未找到相关图片或信息。"
        
        # 返回了错误log
        if isinstance(search_results, str):
            return search_results

        # 2. 如果需要下载，则处理图片下载
        if download_images:
            # 确保保存目录存在
            try:
                os.makedirs(save_path, exist_ok=True)
                logger.info(f"图片将保存到目录: {save_path}")
            except OSError as e:
                logger.error(f"无法创建目录 '{save_path}': {e}。图片将不会被下载。")
                # 如果目录创建失败，则退回到不下载的模式
                download_images = False

        processed_results = []
        if download_images:
            for item in search_results:
                web_image_url = item.get('image_path')
                # 尝试下载图片
                local_image_path = self._download_and_save_image(web_image_url, save_path)
                
                # 如果下载成功，更新路径并添加到最终结果列表
                if local_image_path:
                    item['image_path'] = local_image_path
                    processed_results.append(item)
                else:
                    # 如果下载失败，则忽略这条结果
                    logger.warning(f"已移除一条搜索结果，因为其图片无法下载: {web_image_url}")
        else:
            # 如果不下载，直接使用原始结果
            processed_results = search_results
        
        # 3. 检查处理后是否还有结果
        if not processed_results:
            return "[Image Search] 找到了结果，但所有相关图片都无法成功下载。"

        # 4. 格式化输出
        lines = []
        for item in processed_results:
            entry = (
                f"Image: {item.get('image_path')}\n"
                f"Text: {item.get('snippet')}\n"
                f"Webpage Url: {item.get('url')}"
            )
            lines.append(entry)
        
        formatted_output = "```\n" + '\n\n'.join(lines) + "\n```"
        
        return formatted_output

# --- 使用示例 ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        searcher = ImageSearcher()
        # example_image_url = "https://example.com/example.jpg"
        # example_image_url = "https://example.com/example.jpg"
        example_image_url = "file://./agent_infer/data/mmsearch_files/clO2I1ta06.jpeg"
        # example_image_url = "https://example/vn3n24ib3ir.jpg" # 无效 url

        params = {"image_urls": example_image_url}

        # --- 场景1: 不下载图片 (默认行为) ---
        print("--- 场景1: 图像搜索结果 (不下载图片) ---")
        results_no_download = searcher.call(params, download_images=False)
        print(results_no_download)
        print("-" * 50)
        exit()

        # --- 场景2: 下载图片到目录 ---
        # 注意: 请确保您对指定的 `save_path` 目录有写入权限。
        # 如果在Docker中运行，可以映射一个本地卷到容器的这个路径。
        # 如果在本地直接运行，可以改成一个您有权限的路径，如 './image_downloads'
        local_save_directory = './agent_infer/downloaded_images' # 使用相对路径以便于测试
        
        print(f"--- 场景2: 图像搜索结果 (下载图片到 '{local_save_directory}') ---")
        results_with_download = searcher.call(
            params, 
            download_images=True, 
            save_path=local_save_directory
        )
        print(results_with_download)
        print(f"\n请检查 '{local_save_directory}' 目录查看下载的图片。")

    except ValueError as e:
        print(e)
    except Exception as e:
        print(f"发生未知错误: {e}")

