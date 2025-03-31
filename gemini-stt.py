#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import mimetypes
from datetime import datetime
import sys
import requests
import json
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# 预设的system prompt模板
SYSTEM_PROMPT_PRESETS = {
    "standard": """你是一个专业的音频转字幕助手。请将音频内容转录为准确的SRT格式字幕，保持原始语言。
每个字幕条目应包含：序号、时间码（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）和对应的文本内容。
请确保字幕准确反映音频内容，包括语言习惯和表达方式。""",
    
    "meeting": """你是一个专业的会议记录转写专家。请将会议音频转录为精确的SRT格式字幕。
特别注意：
1. 明确标识不同发言人（如有可能）
2. 准确捕捉专业术语和缩写
3. 省略无意义的填充词和重复内容
4. 保持原始语言的表达方式
每个字幕条目应包含：序号、时间码（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）和对应的文本内容。""",
    
    "interview": """你是一个专业的访谈内容转写专家。请将访谈音频转录为SRT格式字幕。
特别注意：
1. 清晰区分采访者和受访者
2. 标记明显的情绪变化（例如笑声、停顿、迟疑）
3. 保留访谈的真实语气和风格
4. 保持原始语言的地道表达
每个字幕条目应包含：序号、时间码（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）和对应的文本内容。""",
    
    "lecture": """你是一个专业的学术演讲转写专家。请将演讲音频转录为SRT格式字幕。
特别注意：
1. 准确捕捉所有学术术语、理论名称和专业概念
2. 保留讲者使用的示例和比喻
3. 标注关键观点和论证
4. 保持原始语言的学术表达方式
每个字幕条目应包含：序号、时间码（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）和对应的文本内容。""",
    
    "condensed": """你是一个专业的音频内容精简转写专家。请将音频内容转录为SRT格式字幕，但需要删除所有口头禅、重复内容和无意义的填充词。
保留核心信息和关键内容，使字幕更加简洁易读。
每个字幕条目应包含：序号、时间码（格式：HH:MM:SS,mmm --> HH:MM:SS,mmm）和对应的经过精简的文本内容。
在保持原意的基础上，尽可能使字幕简洁明了。"""
}

# --- 配置 ---
# 注意：在生产环境中，最好从环境变量或安全存储中获取敏感信息（如API密钥）

# --- 辅助函数 ---

def get_mime_type(file_path):
    """猜测文件的 MIME 类型。"""
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        # 提供通用后备或引发错误
        # 对于音频，常见的后备类型可能包括：
        if file_path.lower().endswith(('.wav', '.wave')):
            return 'audio/wav'
        elif file_path.lower().endswith('.mp3'):
            return 'audio/mp3'
        elif file_path.lower().endswith(('.flac')):
             return 'audio/flac'
        elif file_path.lower().endswith(('.m4a')):
             # M4A 文件通常包含 AAC 编码
             return 'audio/aac' # 或 'audio/mp4'
        elif file_path.lower().endswith(('.ogg')):
             return 'audio/ogg'
        else:
            print(f"警告：无法确定文件 {os.path.basename(file_path)} 的 MIME 类型，将使用通用类型 'application/octet-stream'。", file=sys.stderr)
            return 'application/octet-stream' # 通用二进制
    return mime_type

def get_system_prompt(preset_key=None, system_prompt_file=None):
    """
    获取系统提示。优先使用预设模板，如未指定则尝试从文件读取。
    
    参数:
        preset_key (str, optional): 预设模板的键名
        system_prompt_file (str, optional): 系统提示文件路径
        
    返回:
        str or None: 系统提示内容，如果无法获取则返回None
    """
    # 如果指定了预设模板，返回相应的预设内容
    if preset_key and preset_key in SYSTEM_PROMPT_PRESETS:
        print(f"使用预设系统提示模板: {preset_key}")
        return SYSTEM_PROMPT_PRESETS[preset_key]
    
    # 否则，尝试从文件读取
    if system_prompt_file:
        return read_system_prompt(system_prompt_file)
    
    return None

def read_system_prompt(file_path):
    """从文件中读取系统提示。"""
    if not file_path or not os.path.exists(file_path):
        print(f"警告：系统提示文件未找到或未指定：{file_path}", file=sys.stderr)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            print(f"已从文件读取系统提示: {file_path}")
            return content
    except Exception as e:
        print(f"错误：读取系统提示文件 {file_path} 失败：{e}", file=sys.stderr)
        return None

# --- API 交互函数（非SDK版本）---

def get_session(proxy=None, disable_proxy=False, retries=3):
    """
    创建一个配置了重试和代理的requests会话对象
    
    参数:
        proxy (str, optional): 代理URL，例如 "http://proxy.example.com:8080"
        disable_proxy (bool): 如果为True，则禁用所有代理
        retries (int): 最大重试次数
        
    返回:
        requests.Session: 配置好的会话对象
    """
    session = requests.Session()
    
    # 配置重试策略
    retry_strategy = Retry(
        total=retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 配置代理设置
    if disable_proxy:
        session.proxies = {"http": None, "https": None}
    elif proxy:
        session.proxies = {"http": proxy, "https": proxy}
    
    return session

def upload_file_with_http(api_key, file_path, api_base_url=None, proxy=None, disable_proxy=False, retries=3):
    """
    使用HTTP请求上传文件到Gemini File API
    
    参数:
        api_key (str): API密钥
        file_path (str): 要上传的文件路径
        api_base_url (str, optional): 自定义API基础URL
        proxy (str, optional): 代理URL
        disable_proxy (bool): 如果为True，则禁用所有代理
        retries (int): 最大重试次数
        
    返回:
        tuple: (文件URI, 文件MIME类型)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件未找到：{file_path}")
    
    # 使用自定义API基础URL或默认URL
    base_url = f"{api_base_url.rstrip('/')}/upload/v1beta/files" if api_base_url else "https://generativelanguage.googleapis.com/upload/v1beta/files"
    mime_type = get_mime_type(file_path)
    file_size = os.path.getsize(file_path)
    display_name = os.path.basename(file_path)
    
    print(f"开始上传文件：{file_path}")
    print(f"使用API端点：{base_url}")
    
    try:
        # 创建会话
        session = get_session(proxy, disable_proxy, retries)
        
        # 步骤1：发起可恢复上传会话
        headers = {
            'X-Goog-Upload-Protocol': 'resumable',
            'X-Goog-Upload-Command': 'start',
            'X-Goog-Upload-Header-Content-Length': str(file_size),
            'X-Goog-Upload-Header-Content-Type': mime_type,
            'Content-Type': 'application/json'
        }
        
        data = json.dumps({'file': {'display_name': display_name}})
        
        print("发起上传会话请求...")
        response = session.post(
            f"{base_url}?key={api_key}",
            headers=headers,
            data=data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"初始化上传失败：{response.status_code} {response.text}")
        
        # 从响应头获取上传URL
        upload_url = response.headers.get('X-Goog-Upload-URL')
        if not upload_url:
            raise Exception("响应中未找到上传URL")
            
        # 步骤2：上传文件内容
        with open(file_path, 'rb') as f:
            file_content = f.read()
            
        upload_headers = {
            'Content-Length': str(file_size),
            'X-Goog-Upload-Offset': '0',
            'X-Goog-Upload-Command': 'upload, finalize'
        }
        
        print(f"上传文件内容 ({file_size} 字节)...")
        upload_response = session.post(
            upload_url,
            headers=upload_headers,
            data=file_content,
            timeout=300  # 大文件可能需要更长时间
        )
        
        if upload_response.status_code != 200:
            raise Exception(f"上传文件内容失败：{upload_response.status_code} {upload_response.text}")
            
        # 解析响应以获取文件信息
        file_info = upload_response.json()
        if 'file' not in file_info or 'uri' not in file_info['file']:
            raise Exception(f"文件上传响应格式无效：{file_info}")
            
        file_uri = file_info['file']['uri']
        print(f"文件上传成功。文件URI：{file_uri}")
        
        return file_uri, mime_type
    
    except Exception as e:
        print(f"文件上传期间发生错误：{e}", file=sys.stderr)
        raise

def generate_srt_with_http(api_key, model_name, file_uri, mime_type, system_prompt=None, api_base_url=None, proxy=None, disable_proxy=False, retries=3):
    """
    使用HTTP请求调用Gemini API生成SRT转录文本
    
    参数:
        api_key (str): API密钥
        model_name (str): 模型名称
        file_uri (str): 上传文件的URI
        mime_type (str): 文件的MIME类型
        system_prompt (str, optional): 系统提示
        api_base_url (str, optional): 自定义API基础URL
        proxy (str, optional): 代理URL
        disable_proxy (bool): 如果为True，则禁用所有代理
        retries (int): 最大重试次数
        
    返回:
        str: 生成的SRT内容
    """
    # 使用自定义API基础URL或默认URL
    base_url = f"{api_base_url.rstrip('/')}/v1beta/models/{model_name}:generateContent" if api_base_url else f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
    # 定义用户提示
    user_prompt = "请识别这段音频的源语言，并严格按照 SRT 格式生成该源语言的字幕，包含序号、时间戳 (HH:MM:SS,mmm --> HH:MM:SS,mmm) 和对应的文本。"
    
    # 构建请求体
    request_body = {
        "contents": [{
            "parts": [
                {"text": user_prompt},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }]
    }
    
    # 如果有系统提示，添加到请求中
    if system_prompt:
        request_body["systemInstruction"] = {
            "parts": [
                {"text": system_prompt}
            ]
        }
        print("已应用系统提示。")
    
    print(f"开始生成SRT内容，使用模型：{model_name}")
    print(f"使用API端点：{base_url}")
    
    try:
        # 创建会话
        session = get_session(proxy, disable_proxy, retries)
        
        # 发送请求
        headers = {
            'Content-Type': 'application/json'
        }
        
        print("发送内容生成请求...")
        response = session.post(
            f"{base_url}?key={api_key}",
            headers=headers,
            json=request_body,
            timeout=600  # 音频转录可能需要更长时间
        )
        
        if response.status_code != 200:
            raise Exception(f"生成内容请求失败：{response.status_code} {response.text}")
            
        # 解析响应
        response_data = response.json()
        
        # 提取生成的文本
        if 'candidates' not in response_data or not response_data['candidates']:
            raise Exception(f"响应中未找到候选项：{response_data}")
            
        candidate = response_data['candidates'][0]
        if 'content' not in candidate or 'parts' not in candidate['content']:
            raise Exception(f"候选项中未找到内容：{candidate}")
            
        # 从parts中组合文本
        generated_text = ""
        for part in candidate['content']['parts']:
            if 'text' in part:
                generated_text += part['text']
        
        print("成功接收到生成响应。")
        
        # 移除可能的 ```srt ... ``` 代码块标记
        if generated_text.startswith("```srt\n"):
            generated_text = generated_text[len("```srt\n"):]
        if generated_text.endswith("\n```"):
            generated_text = generated_text[:-len("\n```")]
            
        return generated_text.strip()
    
    except Exception as e:
        print(f"内容生成期间发生错误：{e}", file=sys.stderr)
        raise

# --- 主执行逻辑 ---

def main():
    parser = argparse.ArgumentParser(description="使用 Gemini API 生成 SRT 字幕（非SDK方式）。")
    parser.add_argument("--audio", required=True, help="输入音频文件的路径。")
    parser.add_argument("--output", required=True, help="保存输出 SRT 文件的路径。")
    parser.add_argument("--api-key", required=True, help="您的 Gemini API 密钥。")
    parser.add_argument("--model", default="gemini-1.5-flash-latest", help="要使用的 Gemini 模型名称 (例如 gemini-1.5-pro-latest)。")
    
    # 系统提示相关参数
    prompt_group = parser.add_argument_group('系统提示选项')
    prompt_group.add_argument("--system-prompt-file", help="可选：包含系统提示的文件的路径。")
    prompt_group.add_argument("--use-system-prompt", action="store_true", help="启用系统提示文件。")
    prompt_group.add_argument("--prompt-preset", choices=list(SYSTEM_PROMPT_PRESETS.keys()), 
                           help=f"使用预设的系统提示模板。可选值: {', '.join(SYSTEM_PROMPT_PRESETS.keys())}")
    
    # 网络和代理选项
    network_group = parser.add_argument_group('网络选项')
    network_group.add_argument("--proxy", help="指定代理服务器，格式：http://host:port")
    network_group.add_argument("--no-proxy", action="store_true", help="禁用所有系统代理")
    network_group.add_argument("--retries", type=int, default=3, help="请求失败时的最大重试次数")
    network_group.add_argument("--api-base-url", help="自定义API基础URL，替换默认的https://generativelanguage.googleapis.com")
    
    # 输出选项
    output_group = parser.add_argument_group('输出选项')
    output_group.add_argument("--output-format", choices=["srt", "txt"], default="srt", 
                            help="输出格式，默认为SRT格式，也可选择纯文本格式")

    args = parser.parse_args()

    print(f"--- 开始生成 SRT 字幕 ---")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"音频文件：{args.audio}")
    print(f"输出文件：{args.output}")
    print(f"模型：{args.model}")
    
    # 系统提示信息
    if args.prompt_preset:
        print(f"使用预设系统提示：{args.prompt_preset}")
    elif args.use_system_prompt:
        print(f"使用自定义系统提示文件：{args.system_prompt_file}")
    else:
        print("未使用系统提示")
        
    # 网络设置信息
    print(f"网络设置：{'使用代理: ' + args.proxy if args.proxy else '禁用代理' if args.no_proxy else '使用系统默认设置'}")
    print(f"最大重试次数：{args.retries}")
    if args.api_base_url:
        print(f"使用自定义API基础URL：{args.api_base_url}")
    else:
        print(f"使用默认API基础URL：https://generativelanguage.googleapis.com")
    
    print(f"输出格式：{args.output_format}")

    # 获取系统提示内容
    system_prompt_content = None
    if args.prompt_preset or args.use_system_prompt:
        system_prompt_content = get_system_prompt(
            preset_key=args.prompt_preset,
            system_prompt_file=args.system_prompt_file if args.use_system_prompt else None
        )
        if system_prompt_content is None:
            print(f"警告：无法获取系统提示，将不使用系统提示继续。", file=sys.stderr)

    try:
        # 步骤 1：上传音频文件
        print("\n步骤 1：使用 HTTP 请求上传音频文件...")
        retries_counter = 0
        max_upload_attempts = 3
        upload_success = False
        
        while retries_counter < max_upload_attempts and not upload_success:
            try:
                if retries_counter > 0:
                    print(f"尝试重新上传，第 {retries_counter+1} 次尝试...")
                    time.sleep(2 * retries_counter)  # 指数退避
                    
                file_uri, mime_type = upload_file_with_http(
                    api_key=args.api_key, 
                    file_path=args.audio,
                    api_base_url=args.api_base_url,
                    proxy=args.proxy,
                    disable_proxy=args.no_proxy,
                    retries=args.retries
                )
                upload_success = True
            except requests.exceptions.ProxyError as e:
                retries_counter += 1
                if retries_counter >= max_upload_attempts:
                    print(f"\n错误：代理连接问题，已尝试 {max_upload_attempts} 次。您可以尝试：")
                    print("1. 使用 --no-proxy 参数禁用代理")
                    print("2. 使用 --proxy 参数指定可用的代理")
                    raise Exception(f"代理连接失败：{str(e)}")
                else:
                    print(f"\n警告：代理连接问题，将重试... ({retries_counter}/{max_upload_attempts})")
            except Exception as e:
                raise  # 其他错误直接抛出
        
        # 步骤 2：生成内容 (SRT)
        print("\n步骤 2：生成 SRT 内容...")
        retries_counter = 0
        max_generate_attempts = 3
        generate_success = False
        
        while retries_counter < max_generate_attempts and not generate_success:
            try:
                if retries_counter > 0:
                    print(f"尝试重新生成内容，第 {retries_counter+1} 次尝试...")
                    time.sleep(3 * retries_counter)  # 指数退避
                    
                srt_content = generate_srt_with_http(
                    api_key=args.api_key,
                    model_name=args.model,
                    file_uri=file_uri,
                    mime_type=mime_type,
                    system_prompt=system_prompt_content,
                    api_base_url=args.api_base_url,
                    proxy=args.proxy,
                    disable_proxy=args.no_proxy,
                    retries=args.retries
                )
                generate_success = True
            except requests.exceptions.ProxyError as e:
                retries_counter += 1
                if retries_counter >= max_generate_attempts:
                    print(f"\n错误：代理连接问题，已尝试 {max_generate_attempts} 次。您可以尝试：")
                    print("1. 使用 --no-proxy 参数禁用代理")
                    print("2. 使用 --proxy 参数指定可用的代理")
                    raise Exception(f"代理连接失败：{str(e)}")
                else:
                    print(f"\n警告：代理连接问题，将重试... ({retries_counter}/{max_generate_attempts})")
            except requests.exceptions.Timeout as e:
                retries_counter += 1
                if retries_counter >= max_generate_attempts:
                    print(f"\n错误：请求超时，已尝试 {max_generate_attempts} 次。")
                    raise Exception(f"请求超时：{str(e)}")
                else:
                    print(f"\n警告：请求超时，将重试... ({retries_counter}/{max_generate_attempts})")
            except Exception as e:
                raise  # 其他错误直接抛出

        # 步骤 3：保存结果
        print(f"\n步骤 3：将内容保存到 {args.output}...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 如果请求纯文本格式，尝试从SRT格式中提取纯文本
        if args.output_format == "txt" and "srt" in args.output.lower():
            print("处理为纯文本格式...")
            text_content = ""
            lines = srt_content.split('\n')
            i = 0
            while i < len(lines):
                if i+2 < len(lines) and '-->' in lines[i+1]:
                    # 找到一个SRT条目，添加文本部分
                    text_content += lines[i+2] + "\n"
                    i += 4  # 跳过空行到下一个条目
                else:
                    i += 1
            
            # 如果没有成功提取，就使用原始内容
            if not text_content.strip():
                text_content = srt_content
                print("警告：无法从SRT格式中提取纯文本，将保存原始内容。")
            
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(text_content)
        else:
            # 保存原始SRT内容
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(srt_content)

        print("\n--- 内容生成完成 ---")
        print(f"输出格式：{'纯文本' if args.output_format == 'txt' else 'SRT字幕'}")
        print(f"输出已保存至：{os.path.abspath(args.output)}")

    except FileNotFoundError as e:
         print(f"\n错误：{e}", file=sys.stderr)
         sys.exit(1)
    except requests.exceptions.ProxyError as e:
         print(f"\n代理连接错误：{e}", file=sys.stderr)
         print("\n可能的解决方案：")
         print("1. 检查您的网络连接是否正常")
         print("2. 使用 --no-proxy 参数禁用系统代理：python gemini-stt-http.py --audio ... --no-proxy")
         print("3. 使用 --proxy 参数指定可用的代理：python gemini-stt-http.py --audio ... --proxy http://proxy.example.com:8080")
         sys.exit(1)
    except requests.exceptions.Timeout as e:
         print(f"\n请求超时错误：{e}", file=sys.stderr)
         print("\n可能的解决方案：")
         print("1. 检查您的网络连接是否稳定")
         print("2. 对于大文件，可能需要更长的处理时间，尝试稍后再试")
         sys.exit(1)
    except requests.exceptions.ConnectionError as e:
         print(f"\n连接错误：{e}", file=sys.stderr)
         print("\n可能的解决方案：")
         print("1. 检查您的网络连接")
         print("2. 检查API域名是否在您的网络环境中可访问")
         print("3. 如果API域名被屏蔽，尝试使用 --api-base-url 参数指定可替代的API域名")
         print("4. 如果您使用VPN，尝试切换到不同的服务器")
         sys.exit(1)
    except Exception as e:
        print(f"\n发生错误：{e}", file=sys.stderr)
        # 打印详细的错误追踪信息，有助于调试
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

