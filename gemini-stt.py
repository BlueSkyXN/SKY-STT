#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import mimetypes
from datetime import datetime
import sys
import requests
import json

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

def read_system_prompt(file_path):
    """从文件中读取系统提示。"""
    if not file_path or not os.path.exists(file_path):
        print(f"警告：系统提示文件未找到或未指定：{file_path}", file=sys.stderr)
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"错误：读取系统提示文件 {file_path} 失败：{e}", file=sys.stderr)
        return None

# --- API 交互函数（非SDK版本）---

def upload_file_with_http(api_key, file_path):
    """
    使用HTTP请求上传文件到Gemini File API
    
    参数:
        api_key (str): API密钥
        file_path (str): 要上传的文件路径
        
    返回:
        tuple: (文件URI, 文件MIME类型)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"音频文件未找到：{file_path}")
    
    base_url = "https://generativelanguage.googleapis.com/upload/v1beta/files"
    mime_type = get_mime_type(file_path)
    file_size = os.path.getsize(file_path)
    display_name = os.path.basename(file_path)
    
    print(f"开始上传文件：{file_path}")
    try:
        # 步骤1：发起可恢复上传会话
        headers = {
            'X-Goog-Upload-Protocol': 'resumable',
            'X-Goog-Upload-Command': 'start',
            'X-Goog-Upload-Header-Content-Length': str(file_size),
            'X-Goog-Upload-Header-Content-Type': mime_type,
            'Content-Type': 'application/json'
        }
        
        data = json.dumps({'file': {'display_name': display_name}})
        
        response = requests.post(
            f"{base_url}?key={api_key}",
            headers=headers,
            data=data
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
        
        upload_response = requests.post(
            upload_url,
            headers=upload_headers,
            data=file_content
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

def generate_srt_with_http(api_key, model_name, file_uri, mime_type, system_prompt=None):
    """
    使用HTTP请求调用Gemini API生成SRT转录文本
    
    参数:
        api_key (str): API密钥
        model_name (str): 模型名称
        file_uri (str): 上传文件的URI
        mime_type (str): 文件的MIME类型
        system_prompt (str, optional): 系统提示
        
    返回:
        str: 生成的SRT内容
    """
    base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
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
    try:
        # 发送请求
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            f"{base_url}?key={api_key}",
            headers=headers,
            json=request_body
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
    parser.add_argument("--system-prompt-file", help="可选：包含系统提示的文件的路径。")
    parser.add_argument("--use-system-prompt", action="store_true", help="启用系统提示文件。")

    args = parser.parse_args()

    print(f"--- 开始生成 SRT 字幕 ---")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"音频文件：{args.audio}")
    print(f"输出文件：{args.output}")
    print(f"模型：{args.model}")
    print(f"使用系统提示：{'是' if args.use_system_prompt else '否'}")
    if args.use_system_prompt:
        print(f"系统提示文件：{args.system_prompt_file}")

    system_prompt_content = None
    if args.use_system_prompt:
        system_prompt_content = read_system_prompt(args.system_prompt_file)
        if system_prompt_content is None and args.system_prompt_file:
             print(f"警告：无法从 {args.system_prompt_file} 读取系统提示，将不使用系统提示继续。", file=sys.stderr)

    try:
        # 步骤 1：上传音频文件
        print("\n步骤 1：使用 HTTP 请求上传音频文件...")
        file_uri, mime_type = upload_file_with_http(args.api_key, args.audio)

        # 步骤 2：生成内容 (SRT)
        print("\n步骤 2：生成 SRT 内容...")
        srt_content = generate_srt_with_http(
            api_key=args.api_key,
            model_name=args.model,
            file_uri=file_uri,
            mime_type=mime_type,
            system_prompt=system_prompt_content
        )

        # 步骤 3：保存结果
        print(f"\n步骤 3：将 SRT 内容保存到 {args.output}...")
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        print("\n--- SRT 生成完成 ---")
        print(f"输出已保存至：{os.path.abspath(args.output)}")

    except FileNotFoundError as e:
         print(f"\n错误：{e}", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"\n发生错误：{e}", file=sys.stderr)
        # 打印详细的错误追踪信息，有助于调试
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()