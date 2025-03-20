#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pyannote模型完整下载工具 (优化版)

自动下载Pyannote说话人分割所需的全部模型，包括：
1. speaker-diarization-3.1（主模型）
2. wespeaker-voxceleb-resnet34-LM（依赖模型，用于嵌入）
3. segmentation-3.0（依赖模型，用于分割）

并自动配置模型路径关系，确保与STT.py兼容。
"""

import os
import sys
import argparse
import yaml
import time
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, login, HfApi

# 定义模型依赖关系
MODEL_DEPENDENCIES = {
    "speaker-diarization-3.1": {
        "repo_id": "pyannote/speaker-diarization-3.1",
        "dependencies": [
            "wespeaker-voxceleb-resnet34-LM",
            "segmentation-3.0"
        ]
    },
    "wespeaker-voxceleb-resnet34-LM": {
        "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "dependencies": []
    },
    "segmentation-3.0": {
        "repo_id": "pyannote/segmentation-3.0",
        "dependencies": []
    }
}

# 3.1版本的标准config.yaml模板
CONFIG_YAML_TEMPLATE_V31 = """version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: {embedding_path}
    embedding_batch_size: 32
    embedding_exclude_overlap: true
    segmentation: {segmentation_path}
    segmentation_batch_size: 32

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0
"""

def check_pyannote_version():
    """检查安装的pyannote.audio版本"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "pyannote.audio"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split(':', 1)[1].strip()
                    print(f"已安装的pyannote.audio版本: {version}")
                    
                    # 基于版本给出建议
                    major_ver = int(version.split('.')[0])
                    if major_ver >= 3:
                        print("当前版本兼容speaker-diarization-3.1模型")
                    else:
                        print("警告: 当前版本可能与speaker-diarization-3.1模型不兼容")
                        print("建议: pip install pyannote.audio==3.1.1")
                    
                    return version
        
        print("未检测到已安装的pyannote.audio")
        print("建议: pip install pyannote.audio==3.1.1")
        return None
    except Exception as e:
        print(f"检查pyannote.audio版本时出错: {e}")
        return None

def download_model(model_id, output_dir, use_auth=True, force=False):
    """下载单个模型到指定目录"""
    print(f"准备下载模型: {model_id}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查目标目录是否已有内容
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()) and not force:
        print(f"目录 {output_dir} 已存在并且不为空。如果要重新下载，请使用 --force 参数")
        return output_dir
    
    # 如果需要授权，先登录
    if use_auth:
        try:
            token = os.environ.get('HF_TOKEN')
            if token:
                print("使用环境变量中的Hugging Face token")
                login(token=token)
            else:
                print("请输入您的Hugging Face token:")
                login()  # 交互式登录，会提示输入 token
            print("Hugging Face 登录成功")
        except Exception as e:
            print(f"登录失败: {e}")
            print("请确保你已经在 Hugging Face 上获得了此模型的访问权限")
            return None
    
    try:
        # 下载模型
        print(f"开始下载模型 {model_id} 到: {output_dir}")
        start_time = time.time()
        
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=output_dir,
            local_dir_use_symlinks=False,  # 不使用符号链接，直接复制文件
            ignore_patterns=[".*", "__pycache__"]  # 忽略某些文件
        )
        
        elapsed_time = time.time() - start_time
        print(f"模型成功下载到: {local_path}")
        print(f"下载耗时: {elapsed_time:.2f}秒")
        
        # 验证下载的文件
        path = Path(local_path)
        model_files = list(path.glob('**/pytorch_model.bin'))
        config_files = list(path.glob('**/config.yaml'))
        
        print(f"找到 {len(model_files)} 个模型文件，{len(config_files)} 个配置文件")
        
        # 列出关键文件
        print("关键文件:")
        for file in model_files:
            print(f"- 模型文件: {file}")
        for file in config_files:
            print(f"- 配置文件: {file}")
        
        return local_path
    except Exception as e:
        print(f"下载 {model_id} 时出错: {e}")
        print("如果是权限错误，请确认你已经在网站上申请并获得了模型访问权限")
        return None


def download_all_dependencies(base_output_dir, use_auth=True, force=False, restructure=False):
    """下载主模型及其所有依赖"""
    print("=" * 80)
    print("开始下载Pyannote完整模型环境")
    print("=" * 80)
    
    # 检查pyannote版本
    pyannote_version = check_pyannote_version()
    
    # 用于存储所有模型的下载路径
    model_paths = {}
    
    # 创建基础输出目录
    os.makedirs(base_output_dir, exist_ok=True)
    
    # 首先下载所有模型
    for model_name, model_info in MODEL_DEPENDENCIES.items():
        print("\n" + "=" * 50)
        print(f"下载模型: {model_name}")
        print("=" * 50)
        
        output_dir = os.path.join(base_output_dir, model_name)
        local_path = download_model(model_info["repo_id"], output_dir, use_auth, force)
        
        if local_path:
            model_paths[model_name] = local_path
        else:
            print(f"模型 {model_name} 下载失败，请解决上述错误后重试")
            return False
    
    # 检查模型文件是否存在
    main_model_dir = model_paths.get("speaker-diarization-3.1")
    embedding_model_dir = model_paths.get("wespeaker-voxceleb-resnet34-LM")
    segmentation_model_dir = model_paths.get("segmentation-3.0")
    
    if not all([main_model_dir, embedding_model_dir, segmentation_model_dir]):
        print("缺少必要的模型目录，下载可能不完整")
        return False
    
    # 寻找模型文件
    embedding_model_file = list(Path(embedding_model_dir).glob('**/pytorch_model.bin'))
    segmentation_model_file = list(Path(segmentation_model_dir).glob('**/pytorch_model.bin'))
    
    if not embedding_model_file:
        print(f"警告: 在 {embedding_model_dir} 中找不到 pytorch_model.bin")
        return False
    
    if not segmentation_model_file:
        print(f"警告: 在 {segmentation_model_dir} 中找不到 pytorch_model.bin")
        return False
    
    embedding_model_path = str(embedding_model_file[0])
    segmentation_model_path = str(segmentation_model_file[0])
    
    # 检查并更新/创建config.yaml文件
    config_path = os.path.join(main_model_dir, "config.yaml")
    
    # 如果要重构目录或config.yaml不存在，创建新的config
    if restructure or not os.path.exists(config_path):
        # 如果存在旧config，先备份
        if os.path.exists(config_path) and not os.path.exists(config_path + ".backup"):
            shutil.copy(config_path, config_path + ".backup")
            print(f"已创建config.yaml备份: {config_path}.backup")
        
        # 创建新的config.yaml
        config_content = CONFIG_YAML_TEMPLATE_V31.format(
            embedding_path=embedding_model_path.replace('\\', '/'),
            segmentation_path=segmentation_model_path.replace('\\', '/')
        )
        
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        print(f"已创建新的config.yaml: {config_path}")
    else:
        # 更新现有config
        try:
            update_config_paths(config_path, embedding_model_path, segmentation_model_path)
            print(f"已更新config.yaml中的依赖路径: {config_path}")
        except Exception as e:
            print(f"更新config.yaml时出错: {e}")
            return False
    
    print("\n" + "=" * 80)
    print("Pyannote模型环境下载完成!")
    print("=" * 80)
    print(f"模型保存在: {base_output_dir}")
    print("下载的模型清单:")
    for model_name, path in model_paths.items():
        print(f"- {model_name}: {path}")
    
    # 检查最终配置是否正确
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print("\n配置文件内容验证:")
            
            # 检查pipeline.params.embedding路径
            embedding_path = config.get('pipeline', {}).get('params', {}).get('embedding', "")
            if embedding_path and os.path.exists(str(embedding_path)):
                print(f"✓ 嵌入模型路径有效: {embedding_path}")
            else:
                print(f"⚠ 嵌入模型路径可能无效: {embedding_path}")
            
            # 检查pipeline.params.segmentation路径
            segmentation_path = config.get('pipeline', {}).get('params', {}).get('segmentation', "")
            if segmentation_path and os.path.exists(str(segmentation_path)):
                print(f"✓ 分割模型路径有效: {segmentation_path}")
            else:
                print(f"⚠ 分割模型路径可能无效: {segmentation_path}")
    except Exception as e:
        print(f"验证配置文件时出错: {e}")
    
    # 提供使用说明 - 适配STT.py
    print("\n" + "=" * 80)
    print("在STT.py中使用此模型的说明:")
    print("=" * 80)
    print(f"1. 在命令行中使用以下参数:")
    print(f"   --speakers --speaker-model \"{main_model_dir}\"")
    print(f"2. 完整命令示例:")
    print(f"   python stt.py -a \"your_audio.m4a\" -m \"your_whisper_model\" -o \"output.txt\" -d cuda -v --speakers --speaker-model \"{main_model_dir}\"")
    
    # PyAnnote版本建议
    if pyannote_version:
        major_ver = int(pyannote_version.split('.')[0])
        if major_ver < 3:
            print("\n注意: 您当前安装的pyannote.audio版本可能与下载的模型不兼容")
            print("建议运行: pip install pyannote.audio==3.1.1")
    else:
        print("\n请确保安装了兼容的pyannote.audio版本:")
        print("pip install pyannote.audio==3.1.1")
    
    return True


def update_config_paths(config_path, embedding_model_path, segmentation_model_path):
    """更新config.yaml中的依赖模型路径"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 转换路径格式为YAML兼容格式
        embedding_path = str(embedding_model_path).replace('\\', '/')
        segmentation_path = str(segmentation_model_path).replace('\\', '/')
        
        # 检查并更新配置结构 - 适应3.1版本格式
        if 'pipeline' not in config:
            config['pipeline'] = {
                'name': 'pyannote.audio.pipelines.SpeakerDiarization',
                'params': {}
            }
        
        if 'params' not in config['pipeline']:
            config['pipeline']['params'] = {}
        
        # 更新嵌入模型路径
        config['pipeline']['params']['embedding'] = embedding_path
        config['pipeline']['params']['embedding_batch_size'] = 32
        config['pipeline']['params']['embedding_exclude_overlap'] = True
        
        # 更新分割模型路径
        config['pipeline']['params']['segmentation'] = segmentation_path
        config['pipeline']['params']['segmentation_batch_size'] = 32
        
        # 确保params部分存在
        if 'params' not in config:
            config['params'] = {
                'clustering': {
                    'method': 'centroid',
                    'min_cluster_size': 12,
                    'threshold': 0.7045654963945799
                },
                'segmentation': {
                    'min_duration_off': 0.0
                }
            }
        
        # 保存修改后的配置
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        
        print(f"已更新配置文件: {config_path}")
        
    except Exception as e:
        print(f"更新配置文件时出错: {e}")
        raise


def check_model_versions():
    """检查当前可用的模型版本"""
    try:
        api = HfApi()
        print("正在查询Pyannote模型的最新版本信息...")
        
        for model_name, model_info in MODEL_DEPENDENCIES.items():
            try:
                repo_info = api.repo_info(repo_id=model_info["repo_id"], repo_type="model")
                print(f"{model_info['repo_id']}:")
                print(f"  - 最新更新: {repo_info.lastModified}")
                print(f"  - 下载次数: {repo_info.downloads}")
                print(f"  - 链接: https://huggingface.co/{model_info['repo_id']}")
            except Exception as e:
                print(f"无法获取 {model_info['repo_id']} 的信息: {e}")
        
        return True
    except Exception as e:
        print(f"检查模型版本时出错: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="一键下载Pyannote完整模型环境工具 (优化版)")
    
    parser.add_argument("--output", default="./pyannote-models",
                      help="输出根目录，默认为 ./pyannote-models")
    parser.add_argument("--no-auth", action="store_true",
                      help="不使用身份验证（仅用于公开模型）")
    parser.add_argument("--force", action="store_true",
                      help="强制重新下载，即使目标目录已存在文件")
    parser.add_argument("--restructure", action="store_true",
                      help="重新创建config.yaml文件，覆盖现有文件")
    parser.add_argument("--check-versions", action="store_true",
                      help="仅检查可用的模型版本，不执行下载")
    parser.add_argument("--check-pyannote", action="store_true",
                      help="仅检查pyannote.audio安装状态")
    parser.add_argument("--token", help="Hugging Face API令牌，也可通过HF_TOKEN环境变量设置")
    
    args = parser.parse_args()
    
    # 如果提供了token，设置为环境变量
    if args.token:
        os.environ['HF_TOKEN'] = args.token
    
    # 如果只是检查pyannote版本
    if args.check_pyannote:
        check_pyannote_version()
        return 0
    
    # 如果只是检查模型版本
    if args.check_versions:
        success = check_model_versions()
        return 0 if success else 1
    
    # 下载所有模型
    success = download_all_dependencies(
        args.output,
        use_auth=not args.no_auth,
        force=args.force,
        restructure=args.restructure
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())