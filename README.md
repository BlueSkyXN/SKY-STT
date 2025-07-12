# SKY-STT - 高性能语音转文字工具集


一个强大的多引擎语音转文字(STT)工具集，支持本地处理和云端API，提供高精度的语音识别、说话人分割和多格式输出功能。

## 🌟 主要特性

### 🚀 双引擎支持
- **本地引擎**: 基于 FasterWhisper + Pyannote，完全离线运行
- **云端引擎**: 基于 Google Gemini API，支持音频和视频文件

### 🎯 核心功能
- **高精度语音识别**: 支持多种语言，中文识别准确率高
- **说话人分割**: 自动识别和标记多个说话人
- **多格式支持**: 输入支持 wav、mp3、m4a、flac、ogg、opus 等格式
- **多输出格式**: txt、srt、json 等多种输出格式
- **批处理模式**: 支持目录批量处理
- **GPU 加速**: 自动检测和使用 CUDA GPU
- **实时进度**: 详细的处理进度和日志输出

### 🛠️ 便利工具
- **模型转换器**: 自动下载和转换 Whisper 模型
- **模型管理器**: 一键下载配置 Pyannote 说话人分割模型
- **格式转换**: 内置音频格式自动转换

## 📋 目录

- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [详细使用说明](#-详细使用说明)
- [工具介绍](#-工具介绍)
- [性能优化](#-性能优化)
- [常见问题](#-常见问题)
- [开发说明](#-开发说明)
- [许可证](#-许可证)

## 🚀 安装指南

### 环境要求

- Python 3.8+
- CUDA 11.0+ (可选，用于 GPU 加速)
- FFmpeg (用于音频格式转换)

### 快速安装

```bash
# 克隆项目
git clone https://github.com/yourusername/SKY-STT.git
cd SKY-STT

# 安装依赖 - 基础版本
pip install faster-whisper soundfile numpy

# 安装依赖 - 完整版本 (包含说话人分割)
pip install faster-whisper pyannote.audio==3.1.1 soundfile numpy pyyaml torch torchaudio

# Gemini API 支持需要额外依赖
pip install requests urllib3

# 模型下载和转换工具依赖
pip install huggingface_hub ctranslate2 tqdm

# 安装 FFmpeg (推荐)
# Ubuntu/Debian:
sudo apt install ffmpeg
# macOS:
brew install ffmpeg
# Windows: 下载并添加到 PATH
```

### GPU 支持 (可选但推荐)

如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch：

```bash
# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Hugging Face Token 配置 (说话人分割必需)

说话人分割功能需要从 Hugging Face 下载模型，某些模型需要访问权限：

1. **注册 Hugging Face 账号**: [https://huggingface.co](https://huggingface.co)
2. **申请模型访问权限**: 访问 [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) 并接受许可条款
3. **生成 Token**: 在 [设置页面](https://huggingface.co/settings/tokens) 创建访问 Token
4. **配置环境变量**:
```bash
export HF_TOKEN=your_huggingface_token_here
# 或在 ~/.bashrc 中添加上述行
```

## 🏃‍♂️ 快速开始

### 1. 准备模型

#### 下载 Whisper 模型
```bash
# 使用内置工具下载并转换 Whisper 模型
python tools/convert_whisper.py --model_id openai/whisper-large-v3 --output_dir ./models/whisper-large-v3

# 或者使用中文优化模型
python tools/convert_whisper.py --model_id Belle-2/Belle-whisper-large-v3-zh --output_dir ./models/whisper-zh
```

#### 下载说话人分割模型 (可选)
```bash
# 下载 Pyannote 说话人分割模型 (需要 Hugging Face Token)
python tools/get_Pyannote_model.py --output ./pyannote-models

# 或使用环境变量设置 Token
export HF_TOKEN=your_huggingface_token
python tools/get_Pyannote_model.py --output ./pyannote-models --token $HF_TOKEN
```

### 2. 基础转录

```bash
# 基础语音转文字
python stt.py --audio audio.wav --model ./models/whisper-large-v3 --output result.txt

# 指定语言和设备
python stt.py -a audio.mp3 -m ./models/whisper-zh -o result.txt -l zh -d cuda

# 生成 SRT 字幕
python stt.py -a video.m4a -m ./models/whisper-large-v3 -o subtitles.srt -f srt
```

### 3. 说话人分割

```bash
# 启用说话人分割
python stt.py -a meeting.wav -m ./models/whisper-large-v3 -o meeting.txt \
  --speakers --speaker-model ./pyannote-models/speaker-diarization-3.1

# 指定说话人数量和范围
python stt.py -a discussion.mp3 -m ./models/whisper-large-v3 -o discussion.srt \
  --speakers --speaker-model ./pyannote-models/speaker-diarization-3.1 \
  --num-speakers 3 --min-speakers 2 --max-speakers 5 -f srt
```

### 4. 批量处理

```bash
# 批量处理目录中的所有音频文件
python stt.py --batch -a ./audio_folder/ -m ./models/whisper-large-v3 \
  --output-dir ./results/ -f json
```

### 5. 使用 Gemini API

```bash
# 使用 Gemini API 处理音频
python gemini-stt.py --input audio.mp3 --output subtitles.srt \
  --api-key YOUR_GEMINI_API_KEY --model gemini-1.5-flash-latest

# 使用预设提示模板
python gemini-stt.py --input meeting.wav --output meeting.srt \
  --api-key YOUR_API_KEY --prompt-preset meeting
```

## 📖 详细使用说明

### STT.py - 本地处理引擎

#### 基本参数

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--audio` | `-a` | 音频文件路径 | 必填 |
| `--model` | `-m` | Whisper模型路径 | 必填 |
| `--output` | `-o` | 输出文件路径 | transcription.txt |
| `--format` | `-f` | 输出格式 (txt/srt/json) | txt |
| `--language` | `-l` | 语言代码 | zh |
| `--device` | `-d` | 计算设备 (cuda/cpu/auto) | auto |
| `--task` | `-t` | 任务类型 (transcribe/translate) | transcribe |

#### 高级参数

```bash
# 调整性能参数
python stt.py -a audio.wav -m model_path -o output.txt \
  --beam-size 5 \
  --compute-type float16 \
  --silence 1000 \
  --no-vad

# 说话人分割参数
python stt.py -a audio.wav -m model_path -o output.txt \
  --speakers \
  --speaker-model pyannote_model_path \
  --min-speakers 2 \
  --max-speakers 5 \
  --audio-format auto

# 批处理参数
python stt.py --batch \
  -a ./audio_folder/ \
  --output-dir ./results/ \
  --in-memory \
  --keep-temp-files
```

#### 输出格式示例

**TXT 格式**:
```
# 转录文件: audio.wav
# 转录时间: 2024-01-15 10:30:00
# 检测语言: zh, 置信度: 0.9999

[0.00s -> 3.50s] 大家好，欢迎来到今天的会议。
[3.50s -> 7.20s] 今天我们主要讨论项目进展情况。
```

**SRT 格式**:
```
1
00:00:00,000 --> 00:00:03,500
大家好，欢迎来到今天的会议。

2
00:00:03,500 --> 00:00:07,200
今天我们主要讨论项目进展情况。
```

**带说话人的 SRT 格式**:
```
1
00:00:00,000 --> 00:00:03,500
[SPEAKER_00] 大家好，欢迎来到今天的会议。

2
00:00:03,500 --> 00:00:07,200
[SPEAKER_01] 今天我们主要讨论项目进展情况。
```

### Gemini-STT.py - 云端处理引擎

#### 基本用法

```bash
# 基础音频转录
python gemini-stt.py --input audio.mp3 --output result.srt \
  --api-key YOUR_API_KEY

# 视频文件转录
python gemini-stt.py --input video.mp4 --output subtitles.srt \
  --api-key YOUR_API_KEY --model gemini-1.5-pro-latest

# 使用远程文件
python gemini-stt.py --file-id your_uploaded_file_id --output result.srt \
  --api-key YOUR_API_KEY
```

#### 系统提示模板

| 模板名称 | 用途 | 特点 |
|----------|------|------|
| `standard` | 通用转录 | 保持原始语言，准确转录 |
| `meeting` | 会议记录 | 识别发言人，过滤填充词 |
| `interview` | 访谈内容 | 区分采访者和受访者，保留情绪 |
| `lecture` | 学术演讲 | 准确捕捉术语，保留学术表达 |
| `condensed` | 精简版本 | 删除口头禅，简洁表达 |

```bash
# 使用会议模板
python gemini-stt.py --input meeting.wav --output meeting.srt \
  --api-key YOUR_API_KEY --prompt-preset meeting

# 使用自定义系统提示
python gemini-stt.py --input lecture.mp3 --output lecture.srt \
  --api-key YOUR_API_KEY --use-system-prompt --system-prompt-file custom_prompt.txt
```

#### 高级参数

```bash
# 调整生成参数
python gemini-stt.py --input audio.wav --output result.srt \
  --api-key YOUR_API_KEY \
  --temperature 0.3 \
  --max-output-tokens 8000 \
  --enable-safety

# 文件管理
python gemini-stt.py --input audio.wav --output result.srt \
  --api-key YOUR_API_KEY \
  --auto-delete \
  --check-first

# 网络设置
python gemini-stt.py --input audio.wav --output result.srt \
  --api-key YOUR_API_KEY \
  --proxy http://proxy.example.com:8080 \
  --retries 5
```

## 🛠️ 工具介绍

### convert_whisper.py - Whisper 模型转换器

自动下载并转换 Whisper 模型为 faster-whisper 格式，提升推理速度。

```bash
# 基本使用
python tools/convert_whisper.py --model_id openai/whisper-large-v3

# 自定义输出目录和精度
python tools/convert_whisper.py \
  --model_id Belle-2/Belle-whisper-large-v3-zh \
  --output_dir ./models/whisper-zh \
  --compute_type float16 \
  --force

# 清理下载文件
python tools/convert_whisper.py \
  --model_id openai/whisper-medium \
  --clean_download \
  --skip_validation
```

**参数说明**:
- `--model_id`: Hugging Face 模型 ID
- `--output_dir`: 转换后模型保存路径
- `--compute_type`: 计算精度 (float16/float32/int8)
- `--force`: 强制覆盖已存在的输出目录
- `--clean_download`: 转换完成后删除原始下载文件
- `--skip_validation`: 跳过模型验证步骤

### get_Pyannote_model.py - Pyannote 模型管理器

一键下载和配置 Pyannote 说话人分割模型及其依赖。

```bash
# 基本下载
python tools/get_Pyannote_model.py --output ./pyannote-models

# 强制重新下载和重构配置
python tools/get_Pyannote_model.py --output ./pyannote-models --force --restructure

# 不使用身份验证 (仅公开模型)
python tools/get_Pyannote_model.py --output ./pyannote-models --no-auth

# 检查模型版本
python tools/get_Pyannote_model.py --check-versions

# 检查 pyannote.audio 安装状态
python tools/get_Pyannote_model.py --check-pyannote
```

**功能特点**:
- 自动下载 speaker-diarization-3.1 主模型
- 自动下载依赖模型 (wespeaker、segmentation)
- 自动配置模型路径关系
- 生成兼容的 config.yaml 文件
- 验证模型完整性

## ⚡ 性能优化

### GPU 加速设置

```bash
# 检查 CUDA 环境
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 指定 CUDA 路径 (如果自动检测失败)
python stt.py -a audio.wav -m model_path -o output.txt \
  --cuda-path /usr/local/cuda-11.8 \
  --device cuda
```

### 内存优化

```bash
# 使用内存处理模式 (大文件)
python stt.py -a large_file.wav -m model_path -o output.txt \
  --in-memory

# 调整批处理大小
python stt.py -a audio.wav -m model_path -o output.txt \
  --compute-type int8  # 减少内存使用
```

### 速度优化

```bash
# 禁用 VAD (语音活动检测) 加速处理
python stt.py -a audio.wav -m model_path -o output.txt \
  --no-vad

# 减少 beam size 加速推理
python stt.py -a audio.wav -m model_path -o output.txt \
  --beam-size 1

# 使用较小的模型
python tools/convert_whisper.py --model_id openai/whisper-base
```

## 🎯 推荐模型

### Whisper 模型选择

| 模型大小 | 推荐用途 | 内存需求 | 速度 | 精度 |
|----------|----------|----------|------|------|
| whisper-base | 快速测试 | ~1GB | 很快 | 中等 |
| whisper-medium | 平衡性能 | ~2GB | 快 | 良好 |
| whisper-large-v3 | 高精度英文 | ~3GB | 中等 | 很高 |
| Belle-whisper-large-v3-zh | 中文优化 | ~3GB | 中等 | 很高 |

### 说话人分割模型

| 模型版本 | 推荐用途 | 支持语言 | 最大说话人 |
|----------|----------|----------|------------|
| speaker-diarization-3.1 | 通用场景 | 多语言 | 无限制 |
| speaker-diarization-3.0 | 兼容性好 | 多语言 | 无限制 |

## ❓ 常见问题

### 安装问题

**Q: 安装 pyannote.audio 时出现错误**
```bash
# 解决方案：使用特定版本
pip install pyannote.audio==3.1.1 torch torchaudio
```

**Q: CUDA 检测失败**
```bash
# 检查 CUDA 安装
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# 手动指定 CUDA 路径
python stt.py --cuda-path /usr/local/cuda-11.8 --device cuda
```

### 使用问题

**Q: 音频格式不支持**
```bash
# 安装 FFmpeg 解决大部分格式问题
sudo apt install ffmpeg  # Ubuntu
brew install ffmpeg      # macOS

# 手动指定 FFmpeg 路径
python stt.py --ffmpeg-path /usr/local/bin --audio input.m4a

# 或手动转换格式
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav
```

**Q: 说话人分割效果不佳**
```bash
# 尝试指定说话人数量
python stt.py --speakers --num-speakers 3

# 调整音频质量
python stt.py --speakers --audio-format wav
```

**Q: 内存不足**
```bash
# 使用较小的计算精度
python stt.py --compute-type int8

# 分段处理大文件
python stt.py --silence 2000  # 增加静音检测时长
```

### API 问题

**Q: Gemini API 调用失败**
```bash
# 检查 API Key (仅上传测试)
python gemini-stt.py --input test.wav --api-key YOUR_KEY --upload-only

# 使用代理和禁用代理
python gemini-stt.py --proxy http://proxy:8080 --input audio.wav --output result.srt --api-key YOUR_KEY
python gemini-stt.py --no-proxy --input audio.wav --output result.srt --api-key YOUR_KEY

# 检查现有文件
python gemini-stt.py --file-id your_file_id --api-key YOUR_KEY --output result.srt
```

## 🔧 开发说明

### 项目结构
```
SKY-STT/
├── stt.py                    # 主处理引擎 (FasterWhisper + Pyannote)
├── gemini-stt.py            # Gemini API 引擎
├── tools/
│   ├── convert_whisper.py   # Whisper 模型转换器
│   └── get_Pyannote_model.py # Pyannote 模型管理器
├── LICENSE                   # GPL-3.0 许可证
└── README.md                # 项目文档
```

