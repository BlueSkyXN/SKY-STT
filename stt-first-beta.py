#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FasterWhisper与Pyannote集成的语音转文字工具

一个高效、稳定的语音转文字命令行工具，基于Faster Whisper和Pyannote模型。
支持多种输出格式、批处理、说话人分割和精确的进度报告。
增强版：支持m4a格式自动转换为wav以适配说话人分割
"""

import os
import sys
import time
import logging
import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Generator, Tuple, Iterator, Set
import io
import numpy as np

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("transcribe.log", encoding="utf-8"),
    ]
)
logger = logging.getLogger("whisper_transcriber")

# =========================== 环境配置 ===========================
# 默认使用系统环境变量
CUDA_PATH = None     # CUDA安装路径
CUDNN_PATH = None    # cuDNN库路径
FFMPEG_PATH = None   # FFmpeg可执行文件路径

# 辅助函数：安全添加路径到环境变量
def add_path_to_env(path, name):
    if path and Path(path).exists():
        path_sep = ";" if os.name == "nt" else ":"
        os.environ["PATH"] = path + path_sep + os.environ.get("PATH", "")
        logger.info(f"使用自定义{name}路径: {path}")
        return True
    elif path:
        logger.warning(f"指定的{name}路径不存在: {path}")
    return False

# 防止OpenMP多次加载问题
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# 限制OpenMP线程数，避免过度使用CPU
os.environ["OMP_NUM_THREADS"] = "4"
# =================================================================

# 临时文件列表，用于跟踪需要清理的文件
TEMP_FILES = []

# 计时装饰器
def timer(func):
    """测量函数执行时间的装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"函数 {func.__name__} 执行耗时: {elapsed_time:.2f}s")
        return result
    return wrapper

class GPUInfo:
    """GPU信息检测与管理类"""
    
    @staticmethod
    def check_cuda_available() -> Dict[str, Any]:
        """增强的CUDA可用性检测"""
        status = {
            "cuda_available": False,
            "device": "cpu",
            "compute_type": "float32",
            "gpu_info": None,
            "detection_method": None
        }
        
        # 方法1：使用PyTorch检测
        try:
            import torch
            torch_cuda_available = torch.cuda.is_available()
            
            if torch_cuda_available:
                status["cuda_available"] = True
                status["device"] = "cuda"
                status["compute_type"] = "float16"
                status["detection_method"] = "PyTorch"
                
                # 获取当前设备信息
                current_device = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(current_device)
                
                # 记录GPU详细信息
                status["gpu_info"] = {
                    "name": props.name,
                    "total_memory_GB": round(props.total_memory / (1024**3), 2),
                    "cuda_version": torch.version.cuda,
                    "device_id": current_device,
                    "multi_processor_count": props.multi_processor_count
                }
                
                # 选择最优的compute_type
                # 较新的GPU支持更高效的半精度计算
                if torch.cuda.get_device_capability(current_device)[0] >= 7:
                    status["compute_type"] = "float16"
                else:
                    status["compute_type"] = "int8"
                
                logger.info(f"CUDA检测(PyTorch): 可用, GPU: {props.name}, "
                           f"内存: {status['gpu_info']['total_memory_GB']}GB, "
                           f"计算类型: {status['compute_type']}")
            else:
                logger.debug("CUDA检测(PyTorch): 不可用")
        except Exception as e:
            logger.debug(f"PyTorch CUDA检测出错: {e}")
        
        # 方法2：检查CUDA环境变量和系统路径
        if not status["cuda_available"]:
            try:
                # 检查CUDA环境变量
                cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                if cuda_home and os.path.exists(cuda_home):
                    status["cuda_available"] = True
                    status["device"] = "cuda"
                    status["detection_method"] = "环境变量"
                    logger.info(f"CUDA检测(环境变量): 可用, CUDA_HOME={cuda_home}")
                
                # 检查系统路径中的CUDA
                if os.name == "nt":  # Windows
                    cuda_paths = [
                        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                        r"C:\Program Files\NVIDIA Corporation\CUDA"
                    ]
                    for base_path in cuda_paths:
                        if os.path.exists(base_path):
                            for version_dir in os.listdir(base_path):
                                if version_dir.startswith("v"):
                                    status["cuda_available"] = True
                                    status["device"] = "cuda"
                                    status["detection_method"] = "系统路径"
                                    logger.info(f"CUDA检测(系统路径): 可用, 路径={os.path.join(base_path, version_dir)}")
                                    break
                else:  # Linux/Mac
                    if any(os.path.exists(f"/usr/local/cuda-{i}") for i in range(9, 13)):
                        status["cuda_available"] = True
                        status["device"] = "cuda"
                        status["detection_method"] = "系统路径"
                        logger.info("CUDA检测(系统路径): 可用, 路径=/usr/local/cuda-*")
            except Exception as e:
                logger.debug(f"环境CUDA检测出错: {e}")
        
        # 方法3：检查设备文件 (仅Linux)
        if not status["cuda_available"] and os.name != "nt":
            try:
                if os.path.exists("/dev/nvidia0"):
                    status["cuda_available"] = True
                    status["device"] = "cuda"
                    status["detection_method"] = "设备文件"
                    logger.info("CUDA检测(设备文件): 可用, /dev/nvidia0存在")
            except Exception as e:
                logger.debug(f"设备文件CUDA检测出错: {e}")
        
        # 总结检测结果
        if not status["cuda_available"]:
            logger.info("CUDA检测结果: 所有检测方法均未发现可用CUDA，将使用CPU模式")
        
        return status
    
    @staticmethod
    def get_gpu_memory_usage() -> Dict[int, Dict[str, float]]:
        """获取当前GPU内存使用率"""
        try:
            import torch
            gpu_memory_usage = {}
            
            for i in range(torch.cuda.device_count()):
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                free = torch.cuda.get_device_properties(i).total_memory / 1024**3 - allocated
                
                gpu_memory_usage[i] = {
                    "total_GB": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                    "used_GB": round(allocated, 2),
                    "free_GB": round(free, 2),
                    "usage_percent": round(allocated / (torch.cuda.get_device_properties(i).total_memory / 1024**3) * 100, 2)
                }
            
            return gpu_memory_usage
        except Exception as e:
            logger.debug(f"获取GPU内存使用率失败: {e}")
            return {}


class SpeakerDiarization:
    """说话人分割处理类"""
    
    def __init__(self, device: str = "auto", model_path: str = None):
        """
        初始化说话人分割处理器
        
        Args:
            device: 计算设备 ("cuda", "cpu", "auto")
            model_path: 本地模型路径或HuggingFace模型ID
        """
        # 检测CUDA可用性
        gpu_status = GPUInfo.check_cuda_available()
        self.device = device
        if device == "auto":
            self.device = gpu_status["device"]
        
        # 验证模型路径
        if model_path is None:
            raise ValueError("必须提供有效的模型路径或Hugging Face模型ID")
        
        self.model_path = model_path
        self._pipeline = None
        self._memory_cache = {}  # 用于缓存音频文件数据
        
        logger.info(f"说话人分割将使用设备: {self.device}")
        logger.info(f"将使用模型: {model_path}")
    
    @property
    def pipeline(self):
        """延迟加载分割模型"""
        if self._pipeline is None:
            try:
                from pyannote.audio import Pipeline
                import torch
                import os
                import yaml
                
                logger.info("加载说话人分割模型...")
                load_start = time.time()
                
                # 检查配置文件是否存在
                config_path = os.path.join(self.model_path, "config.yaml")
                
                # 如果配置文件不存在，尝试创建
                if not os.path.exists(config_path):
                    logger.info(f"配置文件不存在，尝试创建: {config_path}")
                    
                    # 检查所需模型文件路径
                    base_dir = os.path.dirname(self.model_path)
                    embedding_model_path = os.path.join(base_dir, "wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin")
                    segmentation_model_path = os.path.join(base_dir, "segmentation-3.0", "pytorch_model.bin")
                    
                    # 如果目录结构不一致，尝试在模型路径内部寻找
                    if not os.path.exists(embedding_model_path):
                        embedding_model_path = os.path.join(self.model_path, "wespeaker-voxceleb-resnet34-LM", "pytorch_model.bin")
                    
                    if not os.path.exists(segmentation_model_path):
                        segmentation_model_path = os.path.join(self.model_path, "segmentation-3.0", "pytorch_model.bin")
                    
                    # 检查模型文件是否存在
                    if not os.path.exists(embedding_model_path):
                        logger.error(f"嵌入模型文件不存在: {embedding_model_path}")
                        logger.info("请确保模型目录结构如下:")
                        logger.info(f"  {base_dir}/")
                        logger.info(f"  ├── speaker-diarization-3.1/")
                        logger.info(f"  ├── wespeaker-voxceleb-resnet34-LM/pytorch_model.bin")
                        logger.info(f"  └── segmentation-3.0/pytorch_model.bin")
                        raise FileNotFoundError(f"嵌入模型文件不存在: {embedding_model_path}")
                    
                    if not os.path.exists(segmentation_model_path):
                        logger.error(f"分割模型文件不存在: {segmentation_model_path}")
                        raise FileNotFoundError(f"分割模型文件不存在: {segmentation_model_path}")
                    
                    # 创建config.yaml内容
                    config = {
                        "version": "3.1.0",
                        "pipeline": {
                            "name": "pyannote.audio.pipelines.SpeakerDiarization",
                            "params": {
                                "clustering": "AgglomerativeClustering",
                                "embedding": embedding_model_path,
                                "embedding_batch_size": 32,
                                "embedding_exclude_overlap": True,
                                "segmentation": segmentation_model_path,
                                "segmentation_batch_size": 32
                            }
                        },
                        "params": {
                            "clustering": {
                                "method": "centroid",
                                "min_cluster_size": 12,
                                "threshold": 0.7045654963945799
                            },
                            "segmentation": {
                                "min_duration_off": 0.0
                            }
                        }
                    }
                    
                    # 确保目录存在
                    os.makedirs(os.path.dirname(config_path), exist_ok=True)
                    
                    # 写入配置文件
                    with open(config_path, 'w', encoding='utf-8') as f:
                        yaml.dump(config, f, default_flow_style=False)
                    
                    logger.info(f"成功创建配置文件: {config_path}")
                
                # 现在加载模型
                logger.info(f"从配置文件加载说话人分割模型: {config_path}")
                self._pipeline = Pipeline.from_pretrained(config_path)
                
                # 移动到适当的设备
                self._pipeline.to(torch.device(self.device))
                
                load_time = time.time() - load_start
                logger.info(f"说话人分割模型加载完成，耗时: {load_time:.2f}s")
                
            except ImportError as e:
                logger.error(f"导入 pyannote.audio 失败: {e}")
                logger.error("请确保已安装 pyannote.audio: pip install pyannote.audio")
                logger.error("可能还需要 PyYAML: pip install pyyaml")
                raise
            except Exception as e:
                logger.error(f"说话人分割模型加载失败: {e}")
                logger.error(f"错误详情: {str(e)}")
                
                # 诊断信息
                if "not installed" in str(e) or "No module named" in str(e):
                    logger.error("缺少依赖库。请确保已安装: pip install pyannote.audio torch pyyaml")
                elif "yaml" in str(e).lower():
                    logger.error("YAML相关错误。请安装PyYAML: pip install pyyaml")
                
                raise
        
        return self._pipeline
    
    def _preload_audio(self, audio_file: str) -> None:
        """预加载音频到内存缓存"""
        if audio_file not in self._memory_cache:
            try:
                import soundfile as sf
                logger.debug(f"预加载音频到内存: {audio_file}")
                # 使用soundfile加载音频数据
                data, sample_rate = sf.read(audio_file)
                self._memory_cache[audio_file] = {
                    "data": data,
                    "sample_rate": sample_rate
                }
                logger.debug(f"音频预加载完成，样本率: {sample_rate}Hz, 数据大小: {data.shape}")
            except Exception as e:
                logger.warning(f"音频预加载失败，将使用直接文件加载: {e}")
    
    def _clear_cache(self, audio_file: Optional[str] = None):
        """清理内存缓存"""
        if audio_file:
            if audio_file in self._memory_cache:
                del self._memory_cache[audio_file]
                logger.debug(f"已清理音频缓存: {audio_file}")
        else:
            self._memory_cache.clear()
            logger.debug("已清理所有音频缓存")
    
    @timer
    def process(
        self, 
        audio_file: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        preload_audio: bool = True
    ) -> Dict[str, Any]:
        """
        处理音频文件并进行说话人分割
        
        Args:
            audio_file: 音频文件路径
            num_speakers: 指定说话人数量（如果已知）
            min_speakers: 最小说话人数量
            max_speakers: 最大说话人数量
            preload_audio: 是否预加载音频到内存
            
        Returns:
            说话人分割结果
        """
        logger.info(f"开始说话人分割: {audio_file}")
        
        # 可选：预加载音频到内存
        if preload_audio:
            self._preload_audio(audio_file)
        
        # 构建分割参数
        diarization_options = {}
        if num_speakers is not None:
            diarization_options["num_speakers"] = num_speakers
        else:
            if min_speakers is not None:
                diarization_options["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_options["max_speakers"] = max_speakers
        
        try:
            # 执行分割
            diarization = self.pipeline(audio_file, **diarization_options)
            
            # 转换结果为易于使用的格式
            result = {"speakers": {}}
            
            # 将PyAnnote时间轴结果转换为时间片段列表
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time = turn.start
                end_time = turn.end
                
                if speaker not in result["speakers"]:
                    result["speakers"][speaker] = []
                
                result["speakers"][speaker].append({
                    "start": start_time,
                    "end": end_time
                })
            
            # 添加一个展平的时间轴视图（按时间排序的所有片段）
            timeline = []
            for speaker, segments in result["speakers"].items():
                for segment in segments:
                    timeline.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "speaker": speaker
                    })
            
            # 按开始时间排序
            timeline.sort(key=lambda x: x["start"])
            result["timeline"] = timeline
            
            # 计算每个说话人的说话时长和百分比
            total_duration = sum(seg["end"] - seg["start"] for seg in timeline)
            speaker_stats = {}
            
            for speaker, segments in result["speakers"].items():
                speaker_duration = sum(seg["end"] - seg["start"] for seg in segments)
                speaker_stats[speaker] = {
                    "duration": speaker_duration,
                    "percentage": (speaker_duration / total_duration * 100) if total_duration > 0 else 0
                }
            
            result["speaker_stats"] = speaker_stats
            result["total_duration"] = total_duration
            result["num_speakers"] = len(result["speakers"])
            
            # 清理缓存
            if preload_audio:
                self._clear_cache(audio_file)
            
            return result
            
        except Exception as e:
            logger.error(f"说话人分割处理失败: {e}")
            # 清理缓存
            if preload_audio:
                self._clear_cache(audio_file)
            raise
    
    def close(self):
        """释放模型资源"""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        self._clear_cache()

class TranscriptionFormatter:
    """转录结果格式化工具类"""
    
    @staticmethod
    def format_srt(segments: List[Any], output_file: str) -> None:
        """将转录结果保存为SRT字幕格式"""
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, 1):
                # 转换为SRT时间格式 (HH:MM:SS,mmm)
                start_time = TranscriptionFormatter._format_timestamp(segment.start)
                end_time = TranscriptionFormatter._format_timestamp(segment.end)
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment.text.strip()}\n\n")
                f.flush()
    
    @staticmethod
    def format_json(segments: List[Any], info: Any, output_file: str) -> None:
        """将转录结果保存为JSON格式"""
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": []
        }
        
        for segment in segments:
            result["segments"].append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip()
            })
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """将秒转换为SRT格式时间戳 (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    
    @staticmethod
    def format_srt_with_speakers(segments: List[Any], diarization_result: Dict[str, Any], output_file: str) -> None:
        """将转录结果和说话人分割结果保存为带说话人标签的SRT字幕格式"""
        # 将转录段落与说话人时间轴对齐
        segments_with_speaker = []
        
        for segment in segments:
            # 找出与当前片段重叠的说话人
            segment_start = segment.start
            segment_end = segment.end
            
            # 寻找重叠最多的说话人
            best_speaker = None
            max_overlap = 0
            
            for timeline_item in diarization_result["timeline"]:
                speaker_start = timeline_item["start"]
                speaker_end = timeline_item["end"]
                
                # 计算重叠
                overlap_start = max(segment_start, speaker_start)
                overlap_end = min(segment_end, speaker_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = timeline_item["speaker"]
            
            # 添加说话人信息
            segments_with_speaker.append({
                "start": segment_start,
                "end": segment_end,
                "text": segment.text.strip(),
                "speaker": best_speaker if max_overlap > 0 else "未知"
            })
        
        # 写入SRT文件
        with open(output_file, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments_with_speaker, 1):
                # 转换为SRT时间格式
                start_time = TranscriptionFormatter._format_timestamp(segment["start"])
                end_time = TranscriptionFormatter._format_timestamp(segment["end"])
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{segment['speaker']}] {segment['text']}\n\n")
                f.flush()

    @staticmethod
    def format_json_with_speakers(segments: List[Any], diarization_result: Dict[str, Any], info: Any, output_file: str) -> None:
        """将转录结果和说话人分割结果保存为带说话人标签的JSON格式"""
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": []
        }
        
        for segment in segments:
            segment_start = segment.start
            segment_end = segment.end
            
            # 寻找重叠最多的说话人
            best_speaker = None
            max_overlap = 0
            
            for timeline_item in diarization_result["timeline"]:
                speaker_start = timeline_item["start"]
                speaker_end = timeline_item["end"]
                
                # 计算重叠
                overlap_start = max(segment_start, speaker_start)
                overlap_end = min(segment_end, speaker_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = timeline_item["speaker"]
            
            result["segments"].append({
                "start": segment_start,
                "end": segment_end,
                "text": segment.text.strip(),
                "speaker": best_speaker if max_overlap > 0 else "未知"
            })
        
        # 添加说话人信息
        result["speakers"] = {}
        for speaker, segments in diarization_result["speakers"].items():
            result["speakers"][speaker] = {
                "segments": segments,
                "total_time": sum(s["end"] - s["start"] for s in segments)
            }
        
        # 添加说话人统计信息
        result["speaker_stats"] = diarization_result.get("speaker_stats", {})
        result["total_duration"] = diarization_result.get("total_duration", 0)
        result["num_speakers"] = diarization_result.get("num_speakers", 0)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    @staticmethod
    def format_txt_with_speakers(segments: List[Any], diarization_result: Dict[str, Any], info: Any, output_file: str) -> None:
        """将转录结果和说话人分割结果保存为带说话人标签的TXT格式"""
        with open(output_file, "w", encoding="utf-8") as f:
            # 写入文件头信息
            f.write(f"# 转录文件: {Path(output_file).stem}\n")
            f.write(f"# 转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 检测语言: {info.language}, 置信度: {info.language_probability:.4f}\n")
            f.write(f"# 检测说话人数量: {len(diarization_result['speakers'])}\n\n")
            
            # 添加说话人统计信息
            f.write("# 说话人统计:\n")
            for speaker, stats in diarization_result.get("speaker_stats", {}).items():
                duration_mins = stats["duration"] / 60
                f.write(f"# - {speaker}: {duration_mins:.2f}分钟, {stats['percentage']:.2f}%\n")
            f.write("\n")
            
            # 计算总段数以显示进度
            segment_count = 0
            for segment in segments:
                segment_count += 1
                
                # 寻找重叠最多的说话人
                segment_start = segment.start
                segment_end = segment.end
                
                best_speaker = None
                max_overlap = 0
                
                for timeline_item in diarization_result["timeline"]:
                    speaker_start = timeline_item["start"]
                    speaker_end = timeline_item["end"]
                    
                    # 计算重叠
                    overlap_start = max(segment_start, speaker_start)
                    overlap_end = min(segment_end, speaker_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = timeline_item["speaker"]
                
                speaker_tag = f"[{best_speaker}]" if max_overlap > 0 else "[未知]"
                line = f"{speaker_tag} [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                f.write(line)
                f.flush()
                
                # 每10段显示进度
                if segment_count % 10 == 0:
                    logger.info(f"已处理 {segment_count} 个片段...")
    
    @staticmethod
    def get_formatted_output(segments: List[Any], diarization_result: Optional[Dict[str, Any]] = None, info: Optional[Any] = None, output_format: str = "txt") -> str:
        """
        将转录结果格式化为字符串，避免写入磁盘
        
        Args:
            segments: 转录片段
            diarization_result: 说话人分割结果（可选）
            info: 转录信息（可选）
            output_format: 输出格式 ("txt", "srt", "json")
            
        Returns:
            格式化后的字符串
        """
        output_buffer = io.StringIO()
        
        if diarization_result and output_format == "srt":
            # 格式化带说话人的SRT
            segments_with_speaker = []
            
            for segment in segments:
                # 找出与当前片段重叠的说话人
                segment_start = segment.start
                segment_end = segment.end
                
                # 寻找重叠最多的说话人
                best_speaker = None
                max_overlap = 0
                
                for timeline_item in diarization_result["timeline"]:
                    speaker_start = timeline_item["start"]
                    speaker_end = timeline_item["end"]
                    
                    # 计算重叠
                    overlap_start = max(segment_start, speaker_start)
                    overlap_end = min(segment_end, speaker_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = timeline_item["speaker"]
                
                # 添加说话人信息
                segments_with_speaker.append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment.text.strip(),
                    "speaker": best_speaker if max_overlap > 0 else "未知"
                })
            
            # 写入SRT格式
            for i, segment in enumerate(segments_with_speaker, 1):
                # 转换为SRT时间格式
                start_time = TranscriptionFormatter._format_timestamp(segment["start"])
                end_time = TranscriptionFormatter._format_timestamp(segment["end"])
                
                output_buffer.write(f"{i}\n")
                output_buffer.write(f"{start_time} --> {end_time}\n")
                output_buffer.write(f"[{segment['speaker']}] {segment['text']}\n\n")
        
        elif diarization_result and output_format == "json":
            # 格式化带说话人的JSON
            result = {
                "language": info.language if info else "unknown",
                "language_probability": info.language_probability if info else 0.0,
                "segments": []
            }
            
            for segment in segments:
                segment_start = segment.start
                segment_end = segment.end
                
                # 寻找重叠最多的说话人
                best_speaker = None
                max_overlap = 0
                
                for timeline_item in diarization_result["timeline"]:
                    speaker_start = timeline_item["start"]
                    speaker_end = timeline_item["end"]
                    
                    # 计算重叠
                    overlap_start = max(segment_start, speaker_start)
                    overlap_end = min(segment_end, speaker_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = timeline_item["speaker"]
                
                result["segments"].append({
                    "start": segment_start,
                    "end": segment_end,
                    "text": segment.text.strip(),
                    "speaker": best_speaker if max_overlap > 0 else "未知"
                })
            
            # 添加说话人信息
            result["speakers"] = {}
            for speaker, segments in diarization_result["speakers"].items():
                result["speakers"][speaker] = {
                    "segments": segments,
                    "total_time": sum(s["end"] - s["start"] for s in segments)
                }
            
            # 添加说话人统计信息
            result["speaker_stats"] = diarization_result.get("speaker_stats", {})
            result["total_duration"] = diarization_result.get("total_duration", 0)
            result["num_speakers"] = diarization_result.get("num_speakers", 0)
            
            # 写入JSON格式
            json.dump(result, output_buffer, ensure_ascii=False, indent=2)
        
        elif diarization_result and output_format == "txt":
            # 格式化带说话人的TXT
            # 写入文件头信息
            output_buffer.write(f"# 转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if info:
                output_buffer.write(f"# 检测语言: {info.language}, 置信度: {info.language_probability:.4f}\n")
            output_buffer.write(f"# 检测说话人数量: {len(diarization_result['speakers'])}\n\n")
            
            # 添加说话人统计信息
            output_buffer.write("# 说话人统计:\n")
            for speaker, stats in diarization_result.get("speaker_stats", {}).items():
                duration_mins = stats["duration"] / 60
                output_buffer.write(f"# - {speaker}: {duration_mins:.2f}分钟, {stats['percentage']:.2f}%\n")
            output_buffer.write("\n")
            
            # 计算总段数以显示进度
            for segment in segments:
                # 寻找重叠最多的说话人
                segment_start = segment.start
                segment_end = segment.end
                
                best_speaker = None
                max_overlap = 0
                
                for timeline_item in diarization_result["timeline"]:
                    speaker_start = timeline_item["start"]
                    speaker_end = timeline_item["end"]
                    
                    # 计算重叠
                    overlap_start = max(segment_start, speaker_start)
                    overlap_end = min(segment_end, speaker_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = timeline_item["speaker"]
                
                speaker_tag = f"[{best_speaker}]" if max_overlap > 0 else "[未知]"
                line = f"{speaker_tag} [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                output_buffer.write(line)
        
        elif output_format == "srt":
            # 普通SRT格式
            for i, segment in enumerate(segments, 1):
                # 转换为SRT时间格式 (HH:MM:SS,mmm)
                start_time = TranscriptionFormatter._format_timestamp(segment.start)
                end_time = TranscriptionFormatter._format_timestamp(segment.end)
                
                output_buffer.write(f"{i}\n")
                output_buffer.write(f"{start_time} --> {end_time}\n")
                output_buffer.write(f"{segment.text.strip()}\n\n")
        
        elif output_format == "json":
            # 普通JSON格式
            result = {
                "language": info.language if info else "unknown",
                "language_probability": info.language_probability if info else 0.0,
                "segments": []
            }
            
            for segment in segments:
                result["segments"].append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip()
                })
            
            # 写入JSON格式
            json.dump(result, output_buffer, ensure_ascii=False, indent=2)
        
        else:  # 默认txt格式
            # 普通TXT格式
            if info:
                output_buffer.write(f"# 转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                output_buffer.write(f"# 检测语言: {info.language}, 置信度: {info.language_probability:.4f}\n\n")
            
            for segment in segments:
                line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                output_buffer.write(line)
        
        return output_buffer.getvalue()


class AudioProcessor:
    """音频处理工具类"""
    
    # 定义支持的格式列表
    COMPATIBLE_FORMATS = [".wav"]  # 默认兼容的格式
    SUPPORTED_FORMATS = [".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"]  # 支持转换的格式
    
    @staticmethod
    def check_ffmpeg_available() -> bool:
        """
        检查ffmpeg是否可用
        
        Returns:
            是否可用
        """
        try:
            import subprocess
            # 尝试运行ffmpeg命令 - 使用二进制模式避免编码问题
            result = subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=False,  # 使用二进制模式
                check=False
            )
            if result.returncode == 0:
                # 安全解码stdout
                stdout_text = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                ffmpeg_version = stdout_text.split('\n')[0] if stdout_text else "未知版本"
                logger.debug(f"检测到FFmpeg: {ffmpeg_version}")
                return True
            else:
                logger.debug("FFmpeg命令执行失败")
                return False
        except Exception as e:
            logger.debug(f"FFmpeg检测失败: {e}")
            return False
    
    @staticmethod
    def convert_audio_with_ffmpeg(audio_file: str, output_path: str, output_format: str = "wav") -> bool:
        """
        使用FFmpeg将音频文件转换为指定格式，保留原始音频质量特性
        
        Args:
            audio_file: 原始音频文件路径
            output_path: 输出文件路径
            output_format: 输出格式(wav, flac等)
            
        Returns:
            是否成功
        """
        try:
            import subprocess
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 根据输出格式设置编码参数，保留输入音频特性
            codec_params = []
            if output_format.lower() == "wav":
                # PCM格式，不做特殊指定让FFmpeg自动匹配原始采样率和通道数
                codec_params = ["-acodec", "pcm_s24le"]  # 仅指定位深度为24位
            elif output_format.lower() == "flac":
                # FLAC无损压缩，保留原始音频特性
                codec_params = [
                    "-acodec", "flac",
                    "-compression_level", "12"  # 最高压缩级别，不影响质量
                ]
            else:
                # 其他格式尽量使用复制模式
                codec_params = ["-acodec", "copy"]
                logger.info(f"使用复制模式转换为{output_format}，保持原始音频流")
            
            # 构建ffmpeg命令
            cmd = ["ffmpeg", "-i", audio_file, "-y"] + codec_params + [output_path]
            
            logger.debug(f"运行FFmpeg命令: {' '.join(cmd)}")
            
            # 执行命令 - 使用二进制模式避免编码问题
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # 使用二进制模式
                check=False
            )
            
            if process.returncode == 0:
                logger.info(f"FFmpeg转换成功: {output_path}")
                return True
            else:
                # 安全解码stderr，使用replace策略处理无法解码的字符
                stderr_text = process.stderr.decode('utf-8', errors='replace') if process.stderr else ""
                logger.warning(f"FFmpeg转换失败: {stderr_text}")
                
                # 尝试使用最简单的转换参数
                logger.info("尝试使用基本设置进行转换...")
                basic_cmd = ["ffmpeg", "-i", audio_file, "-y", output_path]
                
                basic_process = subprocess.run(
                    basic_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                    check=False
                )
                
                if basic_process.returncode == 0:
                    logger.info(f"使用基本设置转换成功: {output_path}")
                    return True
                else:
                    basic_stderr_text = basic_process.stderr.decode('utf-8', errors='replace') if basic_process.stderr else ""
                    logger.error(f"基本设置转换也失败: {basic_stderr_text}")
                    return False
                
        except Exception as e:
            logger.error(f"FFmpeg转换错误: {e}")
            return False
    
    @staticmethod
    def convert_to_format(
        audio_file: str, 
        target_format: str = "wav", 
        output_dir: Optional[str] = None, 
        auto_cleanup: bool = True
    ) -> str:
        """
        将音频文件转换为指定格式，使用FFmpeg进行无损转换
        
        Args:
            audio_file: 原始音频文件路径
            target_format: 目标格式('wav', 'flac'等)，不包含点
            output_dir: 输出目录（如果为None则使用原文件目录）
            auto_cleanup: 是否自动将转换后的文件添加到清理列表
            
        Returns:
            转换后的音频文件路径
        """
        audio_path = Path(audio_file)
        current_format = audio_path.suffix.lower()
        target_format = target_format.lower()
        target_ext = f".{target_format}" if not target_format.startswith(".") else target_format
        
        # 如果已经是目标格式，则直接返回
        if current_format == target_ext:
            return str(audio_path)
        
        logger.info(f"转换音频格式: {audio_file} -> {target_format.upper()}")
        
        # 确定输出路径
        if output_dir:
            output_path = Path(output_dir) / f"{audio_path.stem}{target_ext}"
            os.makedirs(output_dir, exist_ok=True)
        else:
            # 如果未指定输出目录，使用与原始文件相同的目录
            output_path = audio_path.parent / f"{audio_path.stem}{target_ext}"
        
        # 检查FFmpeg是否可用
        ffmpeg_available = AudioProcessor.check_ffmpeg_available()
        
        if not ffmpeg_available:
            logger.error("FFmpeg不可用，无法转换音频格式")
            logger.error("请安装FFmpeg并确保其在系统PATH中，或使用--ffmpeg-path参数指定其路径")
            return audio_file
        
        # 使用FFmpeg进行转换
        logger.info(f"使用FFmpeg进行无损音频格式转换为{target_format}")
        conversion_success = AudioProcessor.convert_audio_with_ffmpeg(
            audio_file, 
            str(output_path),
            target_format
        )
        
        if conversion_success:
            logger.info(f"音频格式转换完成: {output_path}")
            
            # 将生成的文件添加到清理列表
            if auto_cleanup:
                global TEMP_FILES
                TEMP_FILES.append(str(output_path))
                logger.debug(f"添加临时{target_format}文件到清理列表: {output_path}")
            
            return str(output_path)
        else:
            logger.error(f"无法转换到{target_format}格式，返回原始文件")
            return audio_file
    
    @staticmethod
    def is_format_compatible(audio_file: str, for_diarization: bool = False) -> bool:
        """
        检查音频格式是否与处理兼容
        
        Args:
            audio_file: 音频文件路径
            for_diarization: 是否用于说话人分割（更严格的兼容性要求）
            
        Returns:
            是否兼容
        """
        audio_path = Path(audio_file)
        audio_ext = audio_path.suffix.lower()
        
        # 说话人分割对格式要求更严格
        if for_diarization:
            return audio_ext in AudioProcessor.COMPATIBLE_FORMATS
        
        # 一般转录支持更多格式
        return audio_ext in AudioProcessor.SUPPORTED_FORMATS
    
    @staticmethod
    def convert_to_wav(audio_file: str, output_dir: Optional[str] = None, auto_cleanup: bool = True) -> str:
        """
        将音频文件转换为WAV格式（为了向后兼容）
        
        Args:
            audio_file: 原始音频文件路径
            output_dir: 输出目录
            auto_cleanup: 是否自动将转换后的文件添加到清理列表
            
        Returns:
            转换后的WAV文件路径
        """
        return AudioProcessor.convert_to_format(audio_file, "wav", output_dir, auto_cleanup)
    
    @staticmethod
    def convert_to_wav_with_ffmpeg(audio_file: str, output_path: str) -> bool:
        """
        使用FFmpeg将音频文件转换为WAV格式（为了向后兼容）
        
        Args:
            audio_file: 原始音频文件路径
            output_path: 输出文件路径
            
        Returns:
            是否成功
        """
        return AudioProcessor.convert_audio_with_ffmpeg(audio_file, output_path, "wav")
    
    @staticmethod
    def cleanup_temp_files():
        """清理所有临时生成的文件"""
        global TEMP_FILES
        success_count = 0
        
        for temp_file in TEMP_FILES:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"已删除临时文件: {temp_file}")
                    success_count += 1
            except Exception as e:
                logger.warning(f"删除临时文件失败: {temp_file}, 错误: {e}")
        
        logger.info(f"临时文件清理完成: 成功删除 {success_count}/{len(TEMP_FILES)} 个文件")
        TEMP_FILES.clear()


class WhisperTranscriber:
    """语音转录器核心类"""
    
    def __init__(self, model_path: str, device: str = "auto", compute_type: str = "auto"):
        """
        初始化转录器
        
        Args:
            model_path: 模型路径
            device: 设备类型 ("cuda", "cpu", "auto")
            compute_type: 计算精度类型 ("float16", "float32", "int8", "auto")
        """
        self.model_path = model_path
        
        # 用户明确指定设备时的处理
        if device == "cuda":
            logger.info("用户明确指定使用CUDA设备")
            self.device = "cuda"
            self.compute_type = compute_type if compute_type != "auto" else "float16"
        elif device == "cpu":
            logger.info("用户明确指定使用CPU设备")
            self.device = "cpu"
            self.compute_type = compute_type if compute_type != "auto" else "float32"
        else:  # auto模式
            # 获取自动检测结果
            gpu_status = GPUInfo.check_cuda_available()
            self.device = gpu_status["device"]
            
            if compute_type != "auto":
                self.compute_type = compute_type
            else:
                self.compute_type = gpu_status["compute_type"]
            
            logger.info(f"自动选择设备: {self.device}, 计算类型: {self.compute_type}")
        
        # 延迟加载模型，避免不必要的资源消耗
        self._model = None
        
        # 内存缓存
        self._audio_cache = {}
    
    @property
    def model(self):
        """延迟加载模型"""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                
                logger.info(f"加载模型: {self.model_path}")
                logger.info(f"设备: {self.device}, 计算类型: {self.compute_type}")
                
                # 记录加载开始时间
                load_start = time.time()
                
                self._model = WhisperModel(
                    model_size_or_path=self.model_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=True,
                    download_root=None
                )
                
                load_time = time.time() - load_start
                logger.info(f"模型加载完成，耗时: {load_time:.2f}s")
                
            except ImportError as e:
                logger.error(f"导入faster_whisper失败: {e}")
                logger.error("请确保已安装 faster_whisper: pip install faster-whisper")
                raise
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                raise
        
        return self._model
    
    def _preload_audio(self, audio_file: str) -> None:
        """预加载音频到内存缓存"""
        if audio_file not in self._audio_cache:
            try:
                import soundfile as sf
                logger.debug(f"预加载音频到内存: {audio_file}")
                # 使用soundfile加载音频数据
                data, sample_rate = sf.read(audio_file)
                self._audio_cache[audio_file] = {
                    "data": data,
                    "sample_rate": sample_rate,
                    "temp_file": None
                }
                logger.debug(f"音频预加载完成，样本率: {sample_rate}Hz, 数据大小: {data.shape}")
            except Exception as e:
                logger.warning(f"音频预加载失败，将使用直接文件加载: {e}")
    
    def _clear_cache(self, audio_file: Optional[str] = None):
        """清理内存缓存"""
        if audio_file:
            if audio_file in self._audio_cache:
                # 如果有临时文件，清理它
                if self._audio_cache[audio_file]["temp_file"] and os.path.exists(self._audio_cache[audio_file]["temp_file"]):
                    try:
                        os.remove(self._audio_cache[audio_file]["temp_file"])
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
                
                del self._audio_cache[audio_file]
                logger.debug(f"已清理音频缓存: {audio_file}")
        else:
            # 清理所有临时文件
            for cache_info in self._audio_cache.values():
                if cache_info["temp_file"] and os.path.exists(cache_info["temp_file"]):
                    try:
                        os.remove(cache_info["temp_file"])
                    except Exception as e:
                        logger.warning(f"清理临时文件失败: {e}")
            
            self._audio_cache.clear()
            logger.debug("已清理所有音频缓存")
    
    @timer
    def transcribe(
        self,
        audio_file: str,
        beam_size: int = 5,
        language: str = "zh",
        task: str = "transcribe",
        vad_filter: bool = True,
        min_silence_ms: int = 1000,
        preload_audio: bool = True
    ) -> Tuple[List[Any], Any]:
        """
        转录音频文件
        
        Args:
            audio_file: 音频文件路径
            beam_size: Beam search大小
            language: 语言代码
            task: 任务类型 ("transcribe", "translate")
            vad_filter: 是否使用语音活动检测过滤
            min_silence_ms: 最小静音时长(毫秒)
            preload_audio: 是否预加载音频到内存
            
        Returns:
            (segments列表, 转录信息)
        """
        # 检查文件是否存在
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        
        # 检查文件扩展名
        audio_ext = audio_path.suffix.lower()
        supported_formats = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"]
        if audio_ext not in supported_formats:
            logger.warning(f"音频格式 {audio_ext} 可能不受支持，支持的格式: {', '.join(supported_formats)}")
        
        logger.info(f"开始转录: {audio_file}")
        
        # 预加载音频到内存
        audio_input = audio_file
        if preload_audio:
            self._preload_audio(audio_file)
            if audio_file in self._audio_cache and not self._audio_cache[audio_file]["temp_file"]:
                # 创建临时文件
                try:
                    import soundfile as sf
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext)
                    temp_filename = temp_file.name
                    temp_file.close()
                    
                    # 写入临时文件
                    sf.write(temp_filename, self._audio_cache[audio_file]["data"], 
                             self._audio_cache[audio_file]["sample_rate"])
                    
                    # 更新缓存
                    self._audio_cache[audio_file]["temp_file"] = temp_filename
                    audio_input = temp_filename
                    
                    logger.debug(f"已创建音频临时文件: {temp_filename}")
                except Exception as e:
                    logger.warning(f"创建临时文件失败，将使用原始文件: {e}")
        
        try:
            # 转录配置
            vad_parameters = dict(min_silence_duration_ms=min_silence_ms) if vad_filter else None
            
            segments_iter, info = self.model.transcribe(
                audio_input,
                beam_size=beam_size,
                language=language,
                task=task,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
            )
            
            # 转换迭代器为列表，避免迭代器只能使用一次的问题
            segments = list(segments_iter)
            
            return segments, info
        
        finally:
            # 清理临时文件
            if preload_audio and audio_file in self._audio_cache and audio_input != audio_file:
                try:
                    # 临时文件会在_clear_cache中删除
                    pass
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {e}")
    
    @timer
    def transcribe_and_save(
        self,
        audio_file: str,
        output_file: str = "transcription.txt",
        output_format: str = "txt",
        in_memory: bool = False,
        **kwargs
    ) -> str:
        """
        转录音频并保存结果
        
        Args:
            audio_file: 音频文件路径
            output_file: 输出文件路径
            output_format: 输出格式 ("txt", "srt", "json")
            in_memory: 是否在内存中处理（不写入磁盘）
            **kwargs: 其他转录参数
            
        Returns:
            输出文件的完整路径或内存中的结果字符串
        """
        # 获取转录结果
        segments, info = self.transcribe(audio_file, **kwargs)
        
        # 如果in_memory为True，则不写入磁盘，直接返回格式化的字符串
        if in_memory:
            return TranscriptionFormatter.get_formatted_output(
                segments, 
                None, 
                info, 
                output_format
            )
        
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据输出格式保存结果
        if output_format == "srt":
            TranscriptionFormatter.format_srt(segments, output_file)
        elif output_format == "json":
            TranscriptionFormatter.format_json(segments, info, output_file)
        else:  # 默认txt格式
            with open(output_file, "w", encoding="utf-8") as f:
                # 写入文件头信息
                f.write(f"# 转录文件: {Path(audio_file).name}\n")
                f.write(f"# 转录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 检测语言: {info.language}, 置信度: {info.language_probability:.4f}\n\n")
                
                # 计算总段数以显示进度
                segment_count = 0
                for segment in segments:
                    segment_count += 1
                    line = f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n"
                    f.write(line)
                    f.flush()
                    
                    # 每10段显示进度
                    if segment_count % 10 == 0:
                        logger.info(f"已处理 {segment_count} 个片段...")
        
        # 清理音频缓存
        self._clear_cache(audio_file)
        
        logger.info(f"转录完成，结果已保存至: {output_path.absolute()}")
        return str(output_path.absolute())
    
    @timer
    def transcribe_with_diarization(
        self,
        audio_file: str,
        diarization: 'SpeakerDiarization',
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_format: str = "auto",
        **kwargs
    ) -> Tuple[List[Any], Any, Dict[str, Any]]:
        """
        转录音频并进行说话人分割
        
        Args:
            audio_file: 音频文件路径
            diarization: 说话人分割处理器
            num_speakers: 指定说话人数量（如果已知）
            min_speakers: 最小说话人数量
            max_speakers: 最大说话人数量
            audio_format: 音频格式处理方式 ("auto", "wav", "flac"等)
            **kwargs: 其他转录参数
            
        Returns:
            (segments列表, 转录信息, 说话人分割结果)
        """
        # 检查音频文件格式
        audio_path = Path(audio_file)
        
        # 处理音频格式
        processed_audio = audio_file
        if audio_format == "auto":
            # 自动模式：如果格式不兼容说话人分割，转换为wav
            if not AudioProcessor.is_format_compatible(audio_file, for_diarization=True):
                logger.info(f"检测到{audio_path.suffix}格式音频文件，Pyannote可能无法直接处理")
                # 使用与输出文件相同的目录保存转换后的wav文件
                output_dir = audio_path.parent
                processed_audio = AudioProcessor.convert_to_wav(audio_file, output_dir=output_dir)
                logger.info(f"已将{audio_path.suffix}格式转换为wav: {processed_audio}")
        elif audio_format.lower() != "none":
            # 指定格式模式：强制转换为指定格式
            logger.info(f"根据用户设置，将音频转换为{audio_format}格式")
            output_dir = audio_path.parent
            processed_audio = AudioProcessor.convert_to_format(audio_file, audio_format, output_dir=output_dir)
            logger.info(f"音频转换完成: {processed_audio}")
        else:
            # none模式：不进行任何转换
            logger.info("用户选择不进行音频格式转换，保持原格式")
        
        # 首先进行语音转录
        segments, info = self.transcribe(audio_file, **kwargs)
        
        # 然后执行说话人分割
        diarization_result = diarization.process(
            processed_audio,  # 使用处理后的音频文件
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        return segments, info, diarization_result

    @timer
    def transcribe_and_save_with_speakers(
        self,
        audio_file: str,
        output_file: str = "transcription_with_speakers.txt",
        output_format: str = "txt",
        in_memory: bool = False,
        diarization: Optional[SpeakerDiarization] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        audio_format: str = "auto",
        **kwargs
    ) -> str:
        """
        转录音频、进行说话人分割并保存结果
        
        Args:
            audio_file: 音频文件路径
            output_file: 输出文件路径
            output_format: 输出格式 ("txt", "srt", "json")
            in_memory: 是否在内存中处理（不写入磁盘）
            diarization: 说话人分割处理器（必须提供）
            num_speakers: 指定说话人数量（如果已知）
            min_speakers: 最小说话人数量
            max_speakers: 最大说话人数量
            audio_format: 音频格式处理方式 ("auto", "wav", "flac"等)
            **kwargs: 其他转录参数
            
        Returns:
            输出文件的完整路径或内存中的结果字符串
        """
        # 确保提供了说话人分割处理器
        if diarization is None:
            raise ValueError("必须提供说话人分割处理器(diarization)参数")
        
        try:
            # 获取转录和说话人分割结果
            segments, info, diarization_result = self.transcribe_with_diarization(
                audio_file,
                diarization=diarization,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                audio_format=audio_format,
                **kwargs
            )
            
            # 如果in_memory为True，则不写入磁盘，直接返回格式化的字符串
            if in_memory:
                return TranscriptionFormatter.get_formatted_output(
                    segments, 
                    diarization_result, 
                    info, 
                    output_format
                )
            
            # 确保输出目录存在
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 根据输出格式保存结果
            if output_format == "srt":
                TranscriptionFormatter.format_srt_with_speakers(segments, diarization_result, output_file)
            elif output_format == "json":
                TranscriptionFormatter.format_json_with_speakers(segments, diarization_result, info, output_file)
            else:  # 默认txt格式
                TranscriptionFormatter.format_txt_with_speakers(segments, diarization_result, info, output_file)
            
            logger.info(f"转录与说话人分割完成，结果已保存至: {output_path.absolute()}")
            return str(output_path.absolute())
            
        finally:
            # 清理音频缓存
            self._clear_cache(audio_file)
    
    def close(self):
        """释放模型资源"""
        if self._model is not None:
            del self._model
            self._model = None
        self._clear_cache()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="语音转文字工具 - 基于FasterWhisper与Pyannote",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必选参数
    parser.add_argument("--audio", "-a", required=True, help="音频文件路径")
    parser.add_argument("--model", "-m", required=True, help="Whisper模型路径")
    
    # 输出参数
    parser.add_argument("--output", "-o", default="transcription.txt", help="输出文件路径")
    parser.add_argument("--format", "-f", choices=["txt", "srt", "json"], default="txt",
                      help="输出格式")
    
    # 转录参数
    parser.add_argument("--beam-size", "-b", type=int, default=5, help="Beam search大小")
    parser.add_argument("--language", "-l", default="zh", help="语言代码")
    parser.add_argument("--task", "-t", choices=["transcribe", "translate"], default="transcribe",
                      help="任务类型")
    parser.add_argument("--no-vad", action="store_false", dest="vad_filter",
                      help="禁用语音活动检测")
    parser.add_argument("--silence", "-s", type=int, default=1000,
                      help="最小静音时长(毫秒)")
    
    # 设备参数
    parser.add_argument("--device", "-d", choices=["cuda", "cpu", "auto"], default="auto",
                      help="计算设备")
    parser.add_argument("--compute-type", "-c", 
                      choices=["float16", "float32", "int8", "auto"], default="auto",
                      help="计算精度类型")
    
    # 批处理参数
    parser.add_argument("--batch", action="store_true", help="批处理模式")
    parser.add_argument("--output-dir", default="transcriptions", 
                      help="批处理模式下的输出目录")
    parser.add_argument("--in-memory", action="store_true", 
                      help="在内存中处理而不是写入临时文件")
    
    # 说话人分割参数
    parser.add_argument("--speakers", action="store_true", help="启用说话人分割")
    parser.add_argument("--speaker-model", required=False, help="说话人分割本地模型路径或Hugging Face模型ID")
    parser.add_argument("--num-speakers", type=int, help="指定说话人数量（如果已知）")
    parser.add_argument("--min-speakers", type=int, default=1, help="最小说话人数量")
    parser.add_argument("--max-speakers", type=int, default=5, help="最大说话人数量")
    
    # 音频格式参数（新增）
    parser.add_argument("--audio-format", choices=["auto", "wav", "flac", "none"], default="auto",
                      help="音频格式处理方式: auto(自动检测并转换不兼容格式), wav(强制转换为WAV), "
                           "flac(强制转换为FLAC), none(不进行转换)")
    
    # 环境与路径参数
    env_group = parser.add_argument_group('环境与路径配置')
    env_group.add_argument("--cuda-path", help="自定义CUDA安装路径")
    env_group.add_argument("--cudnn-path", help="自定义cuDNN库路径")
    env_group.add_argument("--ffmpeg-path", help="自定义FFmpeg可执行文件目录")
    
    # 保留旧参数 --cudnn 以保持向后兼容
    env_group.add_argument("--cudnn", help="覆盖cuDNN库路径 (已弃用，请使用--cudnn-path)")
    
    # 高级参数
    parser.add_argument("--verbose", "-v", action="store_true", help="启用详细日志")
    parser.add_argument("--keep-temp-files", action="store_true", help="保留临时文件（不自动删除）")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 处理环境变量配置
    global CUDA_PATH, CUDNN_PATH, FFMPEG_PATH
    
    # 处理CUDA路径
    if args.cuda_path:
        CUDA_PATH = args.cuda_path
        add_path_to_env(CUDA_PATH, "CUDA")
        # 设置CUDA_HOME环境变量
        os.environ["CUDA_HOME"] = CUDA_PATH
        os.environ["CUDA_PATH"] = CUDA_PATH
    
    # 处理cuDNN路径 (优先使用新参数)
    if args.cudnn_path:
        CUDNN_PATH = args.cudnn_path
        add_path_to_env(CUDNN_PATH, "cuDNN")
    elif args.cudnn:  # 向后兼容旧参数
        CUDNN_PATH = args.cudnn
        add_path_to_env(CUDNN_PATH, "cuDNN")
        logger.warning("--cudnn参数已弃用，请使用--cudnn-path")
    
    # 处理FFmpeg路径
    if args.ffmpeg_path:
        FFMPEG_PATH = args.ffmpeg_path
        add_path_to_env(FFMPEG_PATH, "FFmpeg")
    
    # 验证说话人分割参数
    if args.speakers and not args.speaker_model:
        parser.error("启用说话人分割时(--speakers)必须通过--speaker-model提供模型路径或ID")
    
    return args


def process_single_file(args):
    """处理单个音频文件"""
    try:
        # 创建转录器实例
        transcriber = WhisperTranscriber(
            model_path=args.model,
            device=args.device,
            compute_type=args.compute_type
        )
        
        # 如果启用说话人分割
        if args.speakers:
            # 验证模型路径已提供
            if not args.speaker_model:
                raise ValueError("使用说话人分割功能时必须提供--speaker-model参数")
                
            # 创建说话人分割处理器
            diarization = SpeakerDiarization(
                device=args.device,
                model_path=args.speaker_model
            )
            
            # 执行转录与说话人分割
            output_file = transcriber.transcribe_and_save_with_speakers(
                audio_file=args.audio,
                output_file=args.output,
                output_format=args.format,
                in_memory=args.in_memory,
                diarization=diarization,
                num_speakers=args.num_speakers,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                audio_format=args.audio_format,
                beam_size=args.beam_size,
                language=args.language,
                task=args.task,
                vad_filter=args.vad_filter,
                min_silence_ms=args.silence,
                preload_audio=args.in_memory
            )
            
            # 释放资源
            diarization.close()
        else:
            # 常规转录
            output_file = transcriber.transcribe_and_save(
                audio_file=args.audio,
                output_file=args.output,
                output_format=args.format,
                in_memory=args.in_memory,
                beam_size=args.beam_size,
                language=args.language,
                task=args.task,
                vad_filter=args.vad_filter,
                min_silence_ms=args.silence,
                preload_audio=args.in_memory
            )
        
        # 如果是内存模式，需要写入文件
        if args.in_memory and isinstance(output_file, str) and not os.path.isfile(output_file):
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_file)
            
            logger.info(f"内存处理结果已保存至: {output_path.absolute()}")
            output_file = str(output_path.absolute())
        
    except Exception as e:
        logger.error(f"转录失败: {e}", exc_info=True)
        return False
    finally:
        # 确保资源被释放
        if 'transcriber' in locals():
            transcriber.close()
    
    return True


def process_batch(args):
    """批处理多个音频文件"""
    audio_path = Path(args.audio)
    
    # 检查是否为目录
    if not audio_path.is_dir():
        logger.error(f"批处理模式需要指定目录: {args.audio}")
        return False
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有音频文件
    audio_files = []
    for ext in AudioProcessor.SUPPORTED_FORMATS:
        audio_files.extend(audio_path.glob(f"*{ext}"))
        # 同时检查大写扩展名
        audio_files.extend(audio_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.error(f"目录中未找到支持的音频文件: {args.audio}")
        return False
    
    logger.info(f"找到 {len(audio_files)} 个音频文件")
    
    # 创建转录器实例
    transcriber = WhisperTranscriber(
        model_path=args.model,
        device=args.device,
        compute_type=args.compute_type
    )
    
    # 创建说话人分割处理器（如果启用）
    diarization = None
    if args.speakers:
        # 验证模型路径已提供
        if not args.speaker_model:
            logger.error("使用说话人分割功能时必须提供--speaker-model参数")
            return False
            
        try:
            diarization = SpeakerDiarization(
                device=args.device,
                model_path=args.speaker_model
            )
        except Exception as e:
            logger.error(f"初始化说话人分割失败: {e}")
            return False
    
    # 批量处理
    success_count = 0
    try:
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"处理 [{i}/{len(audio_files)}]: {audio_file.name}")
            
            # 构建输出文件路径
            output_file = output_dir / f"{audio_file.stem}.{args.format}"
            
            try:
                if args.speakers and diarization:
                    transcriber.transcribe_and_save_with_speakers(
                        audio_file=str(audio_file),
                        output_file=str(output_file),
                        output_format=args.format,
                        in_memory=args.in_memory,
                        diarization=diarization,
                        num_speakers=args.num_speakers,
                        min_speakers=args.min_speakers,
                        max_speakers=args.max_speakers,
                        audio_format=args.audio_format,
                        beam_size=args.beam_size,
                        language=args.language,
                        task=args.task,
                        vad_filter=args.vad_filter,
                        min_silence_ms=args.silence,
                        preload_audio=args.in_memory
                    )
                else:
                    transcriber.transcribe_and_save(
                        audio_file=str(audio_file),
                        output_file=str(output_file),
                        output_format=args.format,
                        in_memory=args.in_memory,
                        beam_size=args.beam_size,
                        language=args.language,
                        task=args.task,
                        vad_filter=args.vad_filter,
                        min_silence_ms=args.silence,
                        preload_audio=args.in_memory
                    )
                
                success_count += 1
                
            except Exception as e:
                logger.error(f"处理 {audio_file.name} 失败: {e}")
                continue
    
    finally:
        # 释放资源
        transcriber.close()
        if diarization:
            diarization.close()
    
    logger.info(f"批处理完成: 成功 {success_count}/{len(audio_files)}")
    return success_count > 0


def main():
    """主函数"""
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示环境信息
    logger.info(f"Python 版本: {sys.version}")
    logger.info(f"系统平台: {sys.platform}")
    
    # 显示主要组件路径
    if CUDA_PATH:
        logger.info(f"CUDA 路径: {CUDA_PATH}")
    if CUDNN_PATH:
        logger.info(f"cuDNN 路径: {CUDNN_PATH}")
    if FFMPEG_PATH:
        logger.info(f"FFmpeg 路径: {FFMPEG_PATH}")
    
    # 检查FFmpeg是否可用
    ffmpeg_available = AudioProcessor.check_ffmpeg_available()
    if ffmpeg_available:
        logger.info("FFmpeg 可用: 将使用FFmpeg进行音频格式转换")
    else:
        logger.info("FFmpeg 不可用: 将使用soundfile进行音频格式转换")
    
    # 判断处理模式
    success = False
    if args.batch:
        success = process_batch(args)
    else:
        success = process_single_file(args)
    
    # 清理临时文件
    if not args.keep_temp_files:
        AudioProcessor.cleanup_temp_files()
    else:
        logger.info(f"已保留临时文件，共 {len(TEMP_FILES)} 个文件未删除")
    
    # 执行统计
    total_time = time.time() - start_time
    logger.info(f"程序运行总耗时: {total_time:.2f}s")
    
    # 获取GPU状态
    gpu_info = GPUInfo.check_cuda_available()
    if gpu_info["cuda_available"]:
        # 显示GPU内存使用情况
        memory_usage = GPUInfo.get_gpu_memory_usage()
        for gpu_id, info in memory_usage.items():
            logger.info(f"GPU {gpu_id} 内存占用: {info['used_GB']}/{info['total_GB']}GB "
                       f"({info['usage_percent']}%)")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())