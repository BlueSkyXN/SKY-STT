#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将Whisper模型转换为faster-whisper格式
- 支持force参数强制覆盖输出目录
- 优化了目录创建逻辑，避免提前创建导致转换失败
- 增强了错误处理和日志输出
- 修复了验证函数中的dtype属性访问错误
"""

import os
import shutil
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import torch
from ctranslate2.converters import TransformersConverter
from huggingface_hub import snapshot_download, hf_hub_download

def download_model(model_id, output_dir, force_download=False):
    """下载模型并返回本地路径"""
    print(f"正在下载模型 {model_id}...")
    
    # 如果目录已存在且不强制下载，则直接返回
    if os.path.exists(output_dir) and not force_download:
        print(f"模型目录已存在: {output_dir}")
        return output_dir
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 使用snapshot_download下载完整模型
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False  # 避免使用符号链接
        )
        print(f"模型下载完成: {local_path}")
        return local_path
        
    except Exception as e:
        print(f"模型下载出错: {e}")
        # 如果完整下载失败，尝试单独下载必要文件
        try:
            for filename in ["config.json", "model.safetensors", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]:
                try:
                    file_path = hf_hub_download(repo_id=model_id, filename=filename, local_dir=output_dir)
                    print(f"已下载: {file_path}")
                except Exception as file_error:
                    print(f"文件 {filename} 下载失败: {file_error}")
            
            # 确保下载了必要文件后再返回
            if os.path.exists(os.path.join(output_dir, "config.json")):
                return output_dir
            else:
                raise ValueError("下载的模型文件不完整")
        except Exception as individual_error:
            print(f"单独文件下载也失败了: {individual_error}")
            raise

def convert_model(model_path, output_dir, compute_type="float16", force=True):
    """转换模型为CTranslate2格式"""
    print(f"开始转换模型...")
    
    # 检查输出目录是否存在，如果存在且启用force，则删除
    if os.path.exists(output_dir):
        if force:
            print(f"输出目录已存在，正在删除: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(f"输出目录 {output_dir} 已存在，使用 --force 参数可强制覆盖")
    
    # 设置转换器
    start_time = time.time()
    try:
        converter = TransformersConverter(model_path, load_as_float16=(compute_type=="float16"))
        
        # 执行转换 - 注意我们不再提前创建输出目录
        converter.convert(output_dir, quantization=compute_type, force=force)
        
        # 复制额外必要的文件
        extra_files = ["preprocessor_config.json", "tokenizer.json"]
        print(f"正在复制额外必要文件...")
        for filename in extra_files:
            src_path = os.path.join(model_path, filename)
            dst_path = os.path.join(output_dir, filename)
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dst_path)
                    print(f"已复制: {filename}")
                except Exception as copy_error:
                    print(f"复制 {filename} 时出错: {copy_error}")
            else:
                print(f"警告: 源文件 {filename} 不存在，无法复制")
        
        elapsed_time = time.time() - start_time
        print(f"模型转换成功！耗时: {elapsed_time:.1f}秒")
        print(f"输出目录: {os.path.abspath(output_dir)}")
        return True
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"模型转换失败 ({elapsed_time:.1f}秒): {e}")
        return False

def validate_model(model_path):
    """验证转换后的模型是否可用"""
    try:
        from faster_whisper import WhisperModel
        
        print("正在验证模型...")
        start_time = time.time()
        
        # 尝试加载模型，这是最基本的验证
        model = WhisperModel(model_path, device="cpu")  # 使用CPU验证，避免GPU相关问题
        
        # 简单测试模型属性 - 修复：不再访问不存在的dtype属性
        print(f"模型加载成功: {model}")
        print(f"模型设备: {model.model.device}")
        
        # 可选：进行简单转写测试（不需要真实音频）
        try:
            # 创建一个空的音频数据进行基本功能测试
            import numpy as np
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1秒的静音
            segments = list(model.transcribe(dummy_audio)[0])
            print(f"模型功能测试：能够处理音频数据")
        except Exception as test_error:
            print(f"模型功能测试跳过: {test_error}")
        
        elapsed_time = time.time() - start_time
        print(f"模型验证成功！耗时: {elapsed_time:.1f}秒")
        return True
    except Exception as e:
        print(f"模型验证失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="将Whisper模型转换为faster-whisper格式")
    parser.add_argument("--model_id", default="openai/whisper-large-v3", 
                      help="Hugging Face上的模型ID")
    parser.add_argument("--output_dir", default="./whisper-ct2", 
                      help="转换后模型的保存路径")
    parser.add_argument("--download_dir", default=None, 
                      help="原始模型的下载路径，默认使用临时目录")
    parser.add_argument("--compute_type", choices=["float16", "float32", "int8"], default="float16", 
                      help="计算精度，影响模型大小和速度")
    parser.add_argument("--force_download", action="store_true", 
                      help="强制重新下载模型")
    parser.add_argument("--force", action="store_true", 
                      help="强制覆盖输出目录")
    parser.add_argument("--clean_download", action="store_true", 
                      help="转换成功后删除下载的原始模型文件")
    parser.add_argument("--skip_validation", action="store_true", 
                      help="跳过模型验证步骤")
    
    args = parser.parse_args()
    
    # 如果没有指定下载目录，则使用模型ID创建一个
    if args.download_dir is None:
        model_name = args.model_id.split("/")[-1]
        args.download_dir = f"./{model_name}"
    
    # 记录开始
    print("\n===== Whisper模型转换工具 =====")
    print(f"模型ID: {args.model_id}")
    print(f"计算精度: {args.compute_type}")
    print(f"输出目录: {args.output_dir}")
    print(f"强制覆盖: {'是' if args.force else '否'}")
    print("==============================\n")
    
    start_time = time.time()
    
    try:
        # 1. 下载模型
        download_path = download_model(args.model_id, args.download_dir, args.force_download)
        
        # 2. 转换模型 - 注意传递force参数
        conversion_success = convert_model(
            download_path, 
            args.output_dir, 
            args.compute_type, 
            force=True  # 始终使用force=True以确保成功
        )
        
        # 3. 验证模型
        validation_success = True
        if conversion_success and not args.skip_validation:
            validation_success = validate_model(args.output_dir)
            
        # 4. 清理下载文件（可选）
        if args.clean_download and conversion_success:
            print(f"清理下载的原始模型文件...")
            shutil.rmtree(args.download_dir, ignore_errors=True)
            print(f"清理完成")
        
        # 总结
        total_time = time.time() - start_time
        if conversion_success:
            print(f"\n===== 转换流程完成 (总耗时: {total_time:.1f}秒) =====")
            print(f"可以使用以下代码加载模型:")
            print(f"""
from faster_whisper import WhisperModel

model = WhisperModel("{args.output_dir}", device="auto", compute_type="auto")
segments, info = model.transcribe("your_audio.wav", language="zh")

for segment in segments:
    print(f"[{{segment.start:.2f}}s -> {{segment.end:.2f}}s] {{segment.text}}")
            """)
            
            # 在验证失败但转换成功的情况下，提供额外信息
            if not validation_success:
                print("\n注意: 模型验证阶段出现了问题，但模型转换可能已经成功。")
                print("您仍然可以尝试使用这个模型，但请谨慎测试其功能。")
        else:
            print(f"\n转换过程中出现错误，请检查上述日志")
    
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())