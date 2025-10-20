import os
import warnings
import time
from typing import List, Union, Optional, NamedTuple
from pathlib import Path

import ctranslate2
import faster_whisper
import numpy as np
import torch
from transformers import Pipeline
from transformers.pipelines.pt_utils import PipelineIterator
from huggingface_hub import snapshot_download

from .audio import N_SAMPLES, SAMPLE_RATE, load_audio, log_mel_spectrogram
from .types import TranscriptionResult, SingleSegment


def find_silence_points(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, 
                        silence_threshold: float = 0.02, min_silence_duration: float = 0.3):
    """
    检测音频中的静音点
    
    Args:
        audio: 音频数据 (numpy array)
        sample_rate: 采样率
        silence_threshold: 静音阈值（相对于最大音量）
        min_silence_duration: 最小静音持续时间（秒）
    
    Returns:
        List of (start, end) tuples 表示静音区间
    """
    # 计算音频能量（使用滑动窗口）
    window_size = int(0.02 * sample_rate)  # 20ms 窗口
    hop_size = window_size // 2
    
    # 计算每个窗口的 RMS 能量
    energy = []
    for i in range(0, len(audio) - window_size, hop_size):
        window = audio[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        energy.append(rms)
    
    energy = np.array(energy)
    
    # 动态阈值：基于音频的最大能量
    max_energy = np.max(energy)
    threshold = max_energy * silence_threshold
    
    # 检测静音区间
    is_silence = energy < threshold
    silence_regions = []
    
    start = None
    for i, silent in enumerate(is_silence):
        if silent and start is None:
            start = i
        elif not silent and start is not None:
            # 静音区间结束
            duration = (i - start) * hop_size / sample_rate
            if duration >= min_silence_duration:
                # 转换为秒
                start_time = start * hop_size / sample_rate
                end_time = i * hop_size / sample_rate
                silence_regions.append((start_time, end_time))
            start = None
    
    return silence_regions


def find_best_split_point(audio: np.ndarray, target_time: float, 
                          sample_rate: int = SAMPLE_RATE, search_window: float = 5.0):
    """
    在目标时间之前寻找最佳分割点（静音点）
    只向前搜索，确保不超过目标时间（避免超过 30 秒限制）
    
    Args:
        audio: 音频数据
        target_time: 目标分割时间（秒）
        sample_rate: 采样率
        search_window: 向前搜索窗口（秒）
    
    Returns:
        最佳分割点时间（秒），保证 ≤ target_time
    """
    # 只向前搜索：[target_time - search_window, target_time]
    search_start = max(0, target_time - search_window)
    search_end = target_time  # 不超过目标时间
    
    # 提取搜索区域的音频
    start_sample = int(search_start * sample_rate)
    end_sample = int(search_end * sample_rate)
    search_audio = audio[start_sample:end_sample]
    
    # 在搜索区域内查找静音点
    silence_regions = find_silence_points(search_audio, sample_rate)
    
    if not silence_regions:
        # 如果没有找到静音点，返回能量最低的点
        window_size = int(0.1 * sample_rate)  # 100ms 窗口
        min_energy = float('inf')
        best_point = target_time
        
        for i in range(0, len(search_audio) - window_size, window_size // 2):
            window = search_audio[i:i + window_size]
            energy = np.sqrt(np.mean(window ** 2))
            if energy < min_energy:
                min_energy = energy
                best_point = search_start + i / sample_rate
        
        return min(best_point, target_time)  # 确保不超过目标时间
    
    # 找到最接近目标时间但不超过的静音区间中点
    best_silence = None
    min_distance = float('inf')
    
    for start, end in silence_regions:
        # 静音区间的实际时间（相对于整个音频）
        actual_start = search_start + start
        actual_end = search_start + end
        mid_point = (actual_start + actual_end) / 2
        
        # 只考虑不超过目标时间的静音点
        if mid_point <= target_time:
            distance = target_time - mid_point
            if distance < min_distance:
                min_distance = distance
                best_silence = mid_point
    
    return best_silence if best_silence is not None else target_time

def download_model_from_hf(model_name: str, cache_dir: Optional[str] = None) -> str:
    """
    Download model from HuggingFace using huggingface_hub with retry mechanism.
    
    Args:
        model_name: Model name or repo_id (e.g., "large-v3" or "Systran/faster-whisper-large-v3")
        cache_dir: Optional cache directory. If None, uses default HF cache.
    
    Returns:
        Path to the downloaded model directory
    
    Raises:
        Exception: If download fails after retries
    """
    # Map model names to HuggingFace repo IDs
    model_repo_mapping = {
        "large-v3": "Systran/faster-whisper-large-v3",
        "large-v2": "Systran/faster-whisper-large-v2",
        "large-v1": "Systran/faster-whisper-large",
        "large": "Systran/faster-whisper-large",
        "medium": "Systran/faster-whisper-medium",
        "medium.en": "Systran/faster-whisper-medium.en",
        "small": "Systran/faster-whisper-small",
        "small.en": "Systran/faster-whisper-small.en",
        "base": "Systran/faster-whisper-base",
        "base.en": "Systran/faster-whisper-base.en",
        "tiny": "Systran/faster-whisper-tiny",
        "tiny.en": "Systran/faster-whisper-tiny.en",
        "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
        "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
        "distil-small.en": "Systran/faster-distil-whisper-small.en",
    }
    
    # Check if it's a custom repo_id or a standard model name
    if "/" in model_name:
        repo_id = model_name
    else:
        repo_id = model_repo_mapping.get(model_name, f"Systran/faster-whisper-{model_name}")
    
    print(f"[WhisperX] Checking model: {repo_id}")
    
    # Check for HF_ENDPOINT environment variable (for mirrors)
    hf_endpoint = os.environ.get("HF_ENDPOINT", None)
    if hf_endpoint:
        print(f"[WhisperX] Using HuggingFace mirror: {hf_endpoint}")
    
    # 先尝试使用本地已有的模型
    try:
        print(f"[WhisperX] Checking for local model...")
        local_kwargs = {
            "repo_id": repo_id,
            "cache_dir": cache_dir,
            "local_files_only": True,  # 只使用本地文件
        }
        model_path = snapshot_download(**local_kwargs)
        
        # 验证关键文件
        required_files = {
            "config.json": 0,
            "model.bin": 100 * 1024 * 1024,  # Min 100 MB
        }
        
        all_files_ok = True
        for filename, min_size in required_files.items():
            file_path = os.path.join(model_path, filename)
            if not os.path.exists(file_path):
                all_files_ok = False
                break
            if os.path.getsize(file_path) < min_size:
                all_files_ok = False
                break
        
        if all_files_ok:
            print(f"[WhisperX] ✓ Using cached model at: {model_path}")
            model_size_gb = os.path.getsize(os.path.join(model_path, "model.bin")) / (1024**3)
            print(f"[WhisperX] ✓ model.bin: {model_size_gb:.2f} GB")
            return model_path
        else:
            print(f"[WhisperX] Local model incomplete, will download")
    except Exception as e:
        print(f"[WhisperX] Local model not found: {e}")
    
    # 本地没有或不完整，开始下载
    download_kwargs = {
        "repo_id": repo_id,
        "cache_dir": cache_dir,
        "resume_download": True,
        "local_files_only": False,
    }
    
    if hf_endpoint:
        download_kwargs["endpoint"] = hf_endpoint
    
    # Retry mechanism for download
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            print(f"[WhisperX] Downloading model (attempt {attempt + 1}/{max_retries})...")
            model_path = snapshot_download(**download_kwargs)
            
            # Verify critical files exist and are complete
            required_files = {
                "config.json": 0,  # Min size 0 (just check existence)
                "model.bin": 100 * 1024 * 1024,  # Min 100 MB
            }
            
            all_files_ok = True
            for filename, min_size in required_files.items():
                file_path = os.path.join(model_path, filename)
                if not os.path.exists(file_path):
                    print(f"[WhisperX] ⚠ Missing file: {filename}")
                    all_files_ok = False
                else:
                    file_size = os.path.getsize(file_path)
                    if file_size < min_size:
                        print(f"[WhisperX] ⚠ File too small: {filename} ({file_size} bytes < {min_size} bytes)")
                        all_files_ok = False
                    else:
                        if filename == "model.bin":
                            size_gb = file_size / (1024 * 1024 * 1024)
                            print(f"[WhisperX] ✓ {filename}: {size_gb:.2f} GB ({file_size:,} bytes)")
            
            if not all_files_ok:
                if attempt < max_retries - 1:
                    print(f"[WhisperX] Model files incomplete, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise ValueError("Model files incomplete after download")
            
            print(f"[WhisperX] ✓ Model ready at: {model_path}")
            return model_path
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[WhisperX] Download failed: {str(e)}")
                print(f"[WhisperX] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"[WhisperX] ❌ Failed to download model after {max_retries} attempts")
                print(f"[WhisperX] Error: {str(e)}")
                print(f"[WhisperX] Troubleshooting:")
                print(f"[WhisperX]   1. Check internet connection")
                print(f"[WhisperX]   2. Set mirror: export HF_ENDPOINT=https://hf-mirror.com")
                print(f"[WhisperX]   3. Check disk space")
                print(f"[WhisperX]   4. Try: pip install -U huggingface_hub")
                raise

def find_numeral_symbol_tokens(tokenizer):
    numeral_symbol_tokens = []
    for i in range(tokenizer.eot):
        token = tokenizer.decode([i]).removeprefix(" ")
        has_numeral_symbol = any(c in "0123456789%$£" for c in token)
        if has_numeral_symbol:
            numeral_symbol_tokens.append(i)
    return numeral_symbol_tokens

class WhisperModel(faster_whisper.WhisperModel):
    '''
    FasterWhisperModel provides batched inference for faster-whisper.
    Currently only works in non-timestamp mode and fixed prompt for all samples in batch.
    '''

    def generate_segment_batched(self, features: np.ndarray, tokenizer: faster_whisper.tokenizer.Tokenizer, options: faster_whisper.transcribe.TranscriptionOptions, encoder_output = None):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0
        if options.initial_prompt is not None:
            initial_prompt = " " + options.initial_prompt.strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options.without_timestamps,
            prefix=options.prefix,
        )

        encoder_output = self.encode(features)

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )

        result = self.model.generate(
                encoder_output,
                [prompt] * batch_size,
                beam_size=options.beam_size,
                patience=options.patience,
                length_penalty=options.length_penalty,
                max_length=self.max_length,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
            )

        tokens_batch = [x.sequences_ids[0] for x in result]

        def decode_batch(tokens: List[List[int]]) -> str:
            res = []
            for tk in tokens:
                res.append([token for token in tk if token < tokenizer.eot])
            # text_tokens = [token for token in tokens if token < self.eot]
            return tokenizer.tokenizer.decode_batch(res)

        text = decode_batch(tokens_batch)

        return text

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1
        # unsqueeze if batch size = 1
        if len(features.shape) == 2:
            features = np.expand_dims(features, 0)
        features = faster_whisper.transcribe.get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

class FasterWhisperPipeline(Pipeline):
    """
    Huggingface Pipeline wrapper for FasterWhisperModel.
    """
    # TODO:
    # - add support for timestamp mode
    # - add support for custom inference kwargs

    def __init__(
            self,
            model,
            options : NamedTuple,
            tokenizer=None,
            device: Union[int, str, "torch.device"] = -1,
            framework = "pt",
            language : Optional[str] = None,
            suppress_numerals: bool = False,
            **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.suppress_numerals = suppress_numerals
        self._batch_size = kwargs.pop("batch_size", None)
        self._num_workers = 1
        self._preprocess_params, self._forward_params, self._postprocess_params = self._sanitize_parameters(**kwargs)
        self.call_count = 0
        self.framework = framework
        if self.framework == "pt":
            if isinstance(device, torch.device):
                self.device = device
            elif isinstance(device, str):
                self.device = torch.device(device)
            elif device < 0:
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(f"cuda:{device}")
        else:
            self.device = device

        super(Pipeline, self).__init__()

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "tokenizer" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, audio):
        audio = audio['inputs']
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        features = log_mel_spectrogram(
            audio,
            n_mels=model_n_mels if model_n_mels is not None else 80,
            padding=N_SAMPLES - audio.shape[0],
        )
        return {'inputs': features}

    def _forward(self, model_inputs):
        outputs = self.model.generate_segment_batched(model_inputs['inputs'], self.tokenizer, self.options)
        return {'text': outputs}

    def postprocess(self, model_outputs):
        return model_outputs

    def get_iterator(
        self, inputs, num_workers: int, batch_size: int, preprocess_params, forward_params, postprocess_params
    ):
        dataset = PipelineIterator(inputs, self.preprocess, preprocess_params)
        if "TOKENIZERS_PARALLELISM" not in os.environ:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # TODO hack by collating feature_extractor and image_processor

        def stack(items):
            # 处理不同长度的音频块（静音点分割可能产生不同长度）
            # Whisper 要求特征帧数 = 3000（对应 30 秒音频）
            target_frames = 3000
            
            # Pad 或 trim 所有特征到 3000 帧
            padded_inputs = []
            for x in items:
                features = x['inputs']
                current_frames = features.shape[-1]
                
                if current_frames < target_frames:
                    # Pad 到 3000 帧
                    pad_size = target_frames - current_frames
                    features = torch.nn.functional.pad(features, (0, pad_size), mode='constant', value=0)
                elif current_frames > target_frames:
                    # Trim 到 3000 帧（理论上不应该发生，但做个保险）
                    features = features[..., :target_frames]
                    print(f"[WhisperX] Warning: Trimmed features from {current_frames} to {target_frames} frames")
                
                padded_inputs.append(features)
            
            return {'inputs': torch.stack(padded_inputs)}
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, collate_fn=stack)
        model_iterator = PipelineIterator(dataloader, self.forward, forward_params, loader_batch_size=batch_size)
        final_iterator = PipelineIterator(model_iterator, self.postprocess, postprocess_params)
        return final_iterator

    def transcribe(
        self, audio: Union[str, np.ndarray], batch_size=None, num_workers=0, language=None, task=None, chunk_size=30, print_progress = False, combined_progress=False
    ) -> TranscriptionResult:
        if isinstance(audio, str):
            audio = load_audio(audio)

        def data(audio, segments):
            for seg in segments:
                f1 = int(seg['start'] * SAMPLE_RATE)
                f2 = int(seg['end'] * SAMPLE_RATE)
                # print(f2-f1)
                yield {'inputs': audio[f1:f2]}

        # VAD has been disabled - split long audio at silence points
        audio_duration = len(audio) / SAMPLE_RATE
        chunk_duration = 30.0  # 严格的块长度限制（Whisper 要求 ≤ 30s）
        
        vad_segments = []
        if audio_duration <= chunk_duration:
            # 短音频，作为单个片段处理
            vad_segments = [{"start": 0, "end": audio_duration}]
        else:
            # 长音频，在静音点智能分割（只向前搜索，确保不超过 30s）
            print(f"[WhisperX] Audio duration: {audio_duration:.1f}s, finding optimal split points...")
            
            split_points = [0]  # 开始点
            current_pos = 0
            
            while current_pos < audio_duration:
                # 目标：当前位置 + 30 秒
                target_time = current_pos + chunk_duration
                
                if target_time >= audio_duration:
                    # 剩余部分不足 30 秒，直接到结尾
                    break
                
                # 只向前搜索静音点：[target-5s, target]，确保 ≤ 30s
                best_split = find_best_split_point(
                    audio, 
                    target_time, 
                    SAMPLE_RATE, 
                    search_window=5.0  # 向前搜索 5 秒
                )
                
                # 确保块长度 ≤ 30 秒
                if best_split - current_pos > chunk_duration:
                    best_split = current_pos + chunk_duration
                    print(f"[WhisperX] Warning: Forced split at {best_split:.1f}s (exceeds 30s limit)")
                
                split_points.append(best_split)
                current_pos = best_split
            
            # 添加结束点
            split_points.append(audio_duration)
            
            # 生成 segments
            for i in range(len(split_points) - 1):
                start = split_points[i]
                end = split_points[i + 1]
                duration = end - start
                
                # 如果是最后一个块且太短（< 5秒），合并到前一个块
                if i == len(split_points) - 2 and duration < 5.0 and len(vad_segments) > 0:
                    vad_segments[-1]["end"] = end
                else:
                    # 确保每个块严格 ≤ 30 秒
                    if duration > chunk_duration:
                        print(f"[WhisperX] Error: Chunk {i+1} duration {duration:.1f}s exceeds 30s!")
                    vad_segments.append({"start": start, "end": end})
            
            print(f"[WhisperX] Split into {len(vad_segments)} chunks at silence points (all ≤ 30s)")
            # 显示每个块的长度
            for i, seg in enumerate(vad_segments[:5]):  # 显示前 5 个
                duration = seg['end'] - seg['start']
                print(f"[WhisperX]   Chunk {i+1}: {seg['start']:.1f}s - {seg['end']:.1f}s (duration: {duration:.1f}s)")
        if self.tokenizer is None:
            language = language or self.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                self.model.model.is_multilingual, task=task,
                                                                language=language)
        else:
            language = language or self.tokenizer.language_code
            task = task or self.tokenizer.task
            if task != self.tokenizer.task or language != self.tokenizer.language_code:
                self.tokenizer = faster_whisper.tokenizer.Tokenizer(self.model.hf_tokenizer,
                                                                    self.model.model.is_multilingual, task=task,
                                                                    language=language)
                
        if self.suppress_numerals:
            previous_suppress_tokens = self.options.suppress_tokens
            numeral_symbol_tokens = find_numeral_symbol_tokens(self.tokenizer)
            print(f"Suppressing numeral and symbol tokens")
            new_suppressed_tokens = numeral_symbol_tokens + self.options.suppress_tokens
            new_suppressed_tokens = list(set(new_suppressed_tokens))
            self.options = self.options._replace(suppress_tokens=new_suppressed_tokens)

        segments: List[SingleSegment] = []
        batch_size = batch_size or self._batch_size
        total_segments = len(vad_segments)
        for idx, out in enumerate(self.__call__(data(audio, vad_segments), batch_size=batch_size, num_workers=num_workers)):
            if print_progress:
                base_progress = ((idx + 1) / total_segments) * 100
                percent_complete = base_progress / 2 if combined_progress else base_progress
                print(f"Progress: {percent_complete:.2f}%...")
            text = out['text']
            if batch_size in [0, 1, None]:
                text = text[0]
            
            # 调试：显示每个块的识别结果
            chunk_start = round(vad_segments[idx]['start'], 1)
            chunk_end = round(vad_segments[idx]['end'], 1)
            text_preview = text[:60] if text else "(empty)"
            print(f"[WhisperX] Chunk {idx+1}/{total_segments} ({chunk_start}s-{chunk_end}s): {text_preview}...")
            
            segments.append(
                {
                    "text": text,
                    "start": round(vad_segments[idx]['start'], 3),
                    "end": round(vad_segments[idx]['end'], 3)
                }
            )

        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None

        # revert suppressed tokens if suppress_numerals is enabled
        if self.suppress_numerals:
            self.options = self.options._replace(suppress_tokens=previous_suppress_tokens)

        return {"segments": segments, "language": language}


    def detect_language(self, audio: np.ndarray):
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        model_n_mels = self.model.feat_kwargs.get("feature_size")
        segment = log_mel_spectrogram(audio[: N_SAMPLES],
                                      n_mels=model_n_mels if model_n_mels is not None else 80,
                                      padding=0 if audio.shape[0] >= N_SAMPLES else N_SAMPLES - audio.shape[0])
        encoder_output = self.model.encode(segment)
        results = self.model.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        print(f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio...")
        return language

def load_model(whisper_arch,
               device,
               device_index=0,
               compute_type="float16",
               asr_options=None,
               language : Optional[str] = None,
               vad_model=None,
               vad_options=None,
               model : Optional[WhisperModel] = None,
               task="transcribe",
               download_root=None,
               threads=4):
    '''Load a Whisper model for inference.
    Args:
        whisper_arch: str - The name of the Whisper model to load.
        device: str - The device to load the model on.
        compute_type: str - The compute type to use for the model.
        options: dict - A dictionary of options to use for the model.
        language: str - The language of the model. (use English for now)
        model: Optional[WhisperModel] - The WhisperModel instance to use.
        download_root: Optional[str] - The root directory to download the model to.
        threads: int - The number of cpu threads to use per worker, e.g. will be multiplied by num workers.
    Returns:
        A Whisper pipeline.
    '''

    if whisper_arch.endswith(".en"):
        language = "en"

    # Download and load model
    if model is None:
        try:
            # Step 1: Download model using huggingface_hub (with retry and verification)
            print(f"[WhisperX] Loading model: {whisper_arch}")
            downloaded_path = download_model_from_hf(whisper_arch, cache_dir=download_root)
            
            # Step 2: Load the downloaded model with faster-whisper
            print(f"[WhisperX] Initializing model on device: {device}")
            model = WhisperModel(
                downloaded_path,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                local_files_only=True,
                cpu_threads=threads
            )
            print(f"[WhisperX] ✓ Model loaded successfully")
            
        except Exception as e:
            print(f"[WhisperX] ❌ Failed to load model: {whisper_arch}")
            print(f"[WhisperX] Error: {str(e)}")
            print(f"[WhisperX] ")
            print(f"[WhisperX] If model download failed, try:")
            print(f"[WhisperX]   export HF_ENDPOINT=https://hf-mirror.com")
            print(f"[WhisperX]   Then restart ComfyUI")
            raise
    if language is not None:
        tokenizer = faster_whisper.tokenizer.Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task=task, language=language)
    else:
        print("No language specified, language will be first be detected for each audio file (increases inference time).")
        tokenizer = None

    default_asr_options =  {
        "beam_size": 5,
        "best_of": 5,
        "patience": 1,
        "length_penalty": 1,
        "repetition_penalty": 1,
        "no_repeat_ngram_size": 0,
        "temperatures": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": False,
        "prompt_reset_on_temperature": 0.5,
        "initial_prompt": None,
        "prefix": None,
        "suppress_blank": True,
        "suppress_tokens": [-1],
        "without_timestamps": True,
        "max_initial_timestamp": 0.0,
        "word_timestamps": False,
        "prepend_punctuations": "\"'¿([{-",
        "append_punctuations": "\"'.。,，!！?？:：\")]}、",
        "suppress_numerals": False,
        "max_new_tokens": None,
        "clip_timestamps": None,
        "hallucination_silence_threshold": None,
        "hotwords": "",
        "multilingual": False
    }

    if asr_options is not None:
        default_asr_options.update(asr_options)

    suppress_numerals = default_asr_options["suppress_numerals"]
    del default_asr_options["suppress_numerals"]

    default_asr_options = faster_whisper.transcribe.TranscriptionOptions(**default_asr_options)

    # VAD has been disabled - vad_model and vad_options are ignored

    return FasterWhisperPipeline(
        model=model,
        options=default_asr_options,
        tokenizer=tokenizer,
        language=language,
        suppress_numerals=suppress_numerals,
    )
