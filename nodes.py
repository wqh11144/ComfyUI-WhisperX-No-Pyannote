import os
import srt
import torch
import time
import whisperx
import folder_paths
import cuda_malloc
import translators as ts
from tqdm import tqdm
from datetime import timedelta
import torchaudio
import tempfile

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class SRTToString:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "read"

    CATEGORY = "WhisperX"

    def read(self,srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r', encoding="utf-8") as f:
            srt_content = f.read()
        return (srt_content,)


class WhisperX:
    @classmethod
    def INPUT_TYPES(s):
        # 所有支持的 Whisper 模型（与 asr.py 中的 model_repo_mapping 保持一致）
        model_list = [
            "large-v3",           # Systran/faster-whisper-large-v3
            "large-v3-turbo",     # deepdml/faster-whisper-large-v3-turbo-ct2
            "large-v2",           # Systran/faster-whisper-large-v2
            "large-v1",           # Systran/faster-whisper-large
            "large",              # Systran/faster-whisper-large
            "distil-large-v3",    # Systran/faster-distil-whisper-large-v3
            "medium",             # Systran/faster-whisper-medium
            "medium.en",          # Systran/faster-whisper-medium.en
            "distil-medium.en",   # Systran/faster-distil-whisper-medium.en
            "small",              # Systran/faster-whisper-small
            "small.en",           # Systran/faster-whisper-small.en
            "distil-small.en",    # Systran/faster-distil-whisper-small.en
            "base",               # Systran/faster-whisper-base
            "base.en",            # Systran/faster-whisper-base.en
            "tiny",               # Systran/faster-whisper-tiny
            "tiny.en",            # Systran/faster-whisper-tiny.en
        ]
        translator_list = ['alibaba', 'apertium', 'argos', 'baidu', 'bing',
        'caiyun', 'cloudTranslation', 'deepl', 'elia', 'google',
        'hujiang', 'iciba', 'iflytek', 'iflyrec', 'itranslate',
        'judic', 'languageWire', 'lingvanex', 'mglip', 'mirai',
        'modernMt', 'myMemory', 'niutrans', 'papago', 'qqFanyi',
        'qqTranSmart', 'reverso', 'sogou', 'sysTran', 'tilde',
        'translateCom', 'translateMe', 'utibet', 'volcEngine', 'yandex',
        'yeekit', 'youdao']
        lang_list = ["zh","en","ja","ko","ru","fr","de","es","pt","it","ar"]
        level_list = ["segment", "sentence", "word", "char"]
        return {"required":
                    {"audio": ("AUDIO,AUDIOPATH",),  # 支持 AUDIO 对象和文件路径
                     "model_type":(model_list,{
                         "default": "large-v3"
                     }),
                     "language":(lang_list,{  # 新增：语言选择，默认中文
                         "default": "zh"
                     }),
                     "batch_size":("INT",{
                         "default": 4
                     }),
                     "srt_level":(level_list,{
                         "default": "segment"
                     }),
                     "if_translate":("BOOLEAN",{
                         "default": False
                     }),
                     "translator":(translator_list,{
                         "default": "alibaba"
                     }),
                     "to_language":(lang_list,{
                         "default": "en"
                     }),
                     "temperature":("FLOAT",{
                         "default": 0.0,
                         "min": 0.0,
                         "max": 1.0,
                         "step": 0.1
                     }),
                     "condition_on_previous_text":("BOOLEAN",{
                         "default": False
                     })
                     },
                "optional":
                    {"filename_prefix": ("STRING", {"default": "subtitle/ComfyUI"})},
                }

    CATEGORY = "WhisperX"

    RETURN_TYPES = ("SRT","SRT","STRING","STRING")
    RETURN_NAMES = ("ori_srt_file","trans_srt_file","ori_srt_string","trans_srt_string")
    OUTPUT_NODE = True
    FUNCTION = "get_srt"

    def get_srt(self, audio, model_type, language, batch_size, srt_level, if_translate, translator, to_language, temperature, condition_on_previous_text, filename_prefix="subtitle/ComfyUI"):
        # 处理输入：支持 AUDIO 对象或文件路径
        temp_path = None
        
        if isinstance(audio, dict) and 'waveform' in audio:
            # AUDIO 对象：来自 TTS 等节点
            print("[WhisperX] Received AUDIO object from TTS/other node")
            waveform = audio['waveform']  # [batch, channels, samples]
            sample_rate = audio.get('sample_rate', 16000)
            
            # 创建临时音频文件
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # 保存为 WAV 文件
            # waveform 可能是 [1, channels, samples]，需要处理
            if waveform.dim() == 3:
                waveform = waveform.squeeze(0)  # [channels, samples]
            
            torchaudio.save(temp_path, waveform.cpu(), sample_rate)
            audio_path = temp_path
            base_name = f"tts_audio_{int(time.time())}"
            print(f"[WhisperX] Saved AUDIO to temp file: {temp_path}")
            
        elif isinstance(audio, str):
            # 文件路径：传统方式
            print(f"[WhisperX] Received file path: {audio}")
            audio_path = audio
            base_name = os.path.basename(audio)[:-4]
        else:
            raise ValueError(f"[WhisperX] Unsupported audio input type: {type(audio)}")
        
        try:
            compute_type = "float16"
            device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
            
            # 1. Transcribe with original whisper (batched)
            if model_type == "large-v3-turbo":
                model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"
            
            # 配置 ASR 选项：引导 Whisper 生成标点符号
            # initial_prompt 包含多语言标点示例，引导模型正确输出标点
            # 使用更长、更自然的示例，帮助模型学习标点模式
            asr_options = {
                "initial_prompt": (
                    # 中文示例（丰富标点符号，包括感叹号、问号、逗号、顿号、省略号、引号、书名号等）
                    "你好，今天天气真好！外面阳光明媚，温度适宜。你想去哪里玩呢？我们可以去公园散步，也可以去咖啡馆聊天。"
                    "请注意：天气预报说下午可能下雨……要不要带把伞？“当然要！”小明说。——来自《天气物语》。"
                    "哈哈，你听说了吗？昨天的比赛，简直太精彩了！"
                    "唉，事情为什么会变成这样呢？"
                    "快点儿，我们要迟到了！（妈妈喊道。）"
                    "……其实，我还有很多话没说出口。"

                    # 英文示例（包含丰富标点：!?,.:;"'—()…）
                    "Hello, how are you today? I'm doing great, thank you! The weather is beautiful, isn't it? Yes, it's perfect for a walk."
                    "Well... what do you think about the movie—did you like it? \"Absolutely amazing!\" Sarah exclaimed."
                    "Don't forget: tomorrow's meeting is at 10 a.m. -- see you there!"
                    "'Wait,' he whispered, 'are you sure about this?'"

                    # 日文示例（包含多种日语标点：。！？「」、…）
                    "こんにちは、今日はいい天気ですね！本当に素晴らしいです。どこへ行きますか？公園に散歩しましょう。"
                    "「ねえ、知ってる？」…昨日のテスト、めっちゃ難しかった！"
                    "えっ！？うそでしょ？"
                    "……静かなる夜に、星が輝く。"

                    # 韩文示例（丰富的标点：!?, . " ” … ）
                    "안녕하세요, 오늘 날씨가 정말 좋네요! 네, 정말 아름답습니다."
                    "“정말요?” 그는 물었다. 믿을 수 없어…"
                    "뭐라고?! 그게 무슨 뜻이죠?"

                    # 法文示例（丰富标点：!?,.:;«»…）
                    "Bonjour, comment allez-vous? Très bien, merci! Quelle belle journée, n'est-ce pas ?"
                    "« Incroyable ! » s'est-elle exclamée. Eh bien… on verra."
                    "Qu'en penses-tu : c'est une bonne idée ?"

                    # 西班牙文示例（丰富标点：¿¡!?.,:;"«»…）
                    "Hola, ¿cómo estás? ¡Muy bien, gracias! ¿Has visto la película “El Viaje”? ¡Es fantástica!"
                    "—¿En serio? ¡No lo puedo creer!"
                    "Vamos a la cafetería… ¿te apetece?"

                    # 俄文示例（丰富标点：!?, . „“ … —）
                    "Здравствуйте, как дела? Отлично, спасибо! Сегодня замечательная погода — не правда ли?"
                    "«Да ну?!» — удивился он. …Тишина повисла в комнате."
                    "Что это было?.."
                ),
                # 使用用户设置的上下文条件（有助于生成连贯的标点，但可能导致重复）
                "condition_on_previous_text": condition_on_previous_text,
                # 使用用户设置的温度（0.0=最确定，有利于标点生成；值越高越随机）
                "temperatures": [temperature],
                # 降低 no_speech_threshold 以提高识别率（默认 0.6 太高）
                # 值越低，越容易识别出语音内容（但可能增加误识别）
                "no_speech_threshold": 0.4,
                # 降低 log_prob_threshold 以接受更多识别结果
                "log_prob_threshold": -1.5,
            }
            
            print(f"[WhisperX] Using language: {language} (skip auto-detection)")
            print(f"[WhisperX] Temperature: {temperature} | Context: {condition_on_previous_text}")
            model = whisperx.load_model(model_type, device, compute_type=compute_type, asr_options=asr_options, language=language)
            audio_data = whisperx.load_audio(audio_path)  # 使用 audio_path 而不是 audio
            result = model.transcribe(audio_data, batch_size=batch_size)
            
            language_code = language  # 使用用户选择的语言
            
            # 保存原始 segments（用于 segment 级别）
            # 由于使用了基于静音点的智能分割，不会产生重复，无需去重
            original_segments = result["segments"]
            
            # 2. Align whisper output
            # return_char_alignments: char 级别需要
            # merge_sentences: segment/word/char 需要合并，sentence 不合并
            return_char = (srt_level == "char")
            merge_sentences = (srt_level != "sentence")
            
            model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio_data, device, 
                                   return_char_alignments=return_char, 
                                   merge_sentences=merge_sentences)
            
            # 基于静音点分割后，不会产生重复内容，无需额外去重处理
            
            # delete model if low on GPU resources
            import gc; gc.collect(); torch.cuda.empty_cache(); del model_a,model
            
            # 生成文件名（根据 filename_prefix 保存到指定目录）
            # 解析子目录和文件名前缀
            if "/" in filename_prefix or "\\" in filename_prefix:
                # 包含子目录
                parts = filename_prefix.replace("\\", "/").split("/")
                subdir = "/".join(parts[:-1])
                prefix = parts[-1]
                
                # 创建完整的输出目录
                save_dir = os.path.join(out_path, subdir)
                os.makedirs(save_dir, exist_ok=True)
            else:
                # 没有子目录
                save_dir = out_path
                subdir = ""
                prefix = filename_prefix
            
            # 生成带时间戳的文件名
            timestamp = int(time.time() * 1000)  # 毫秒级时间戳
            level_suffix = f"_{srt_level}" if srt_level != "segment" else ""
            srt_filename = f"{prefix}_{timestamp}{level_suffix}.srt"
            trans_srt_filename = f"{prefix}_{timestamp}{level_suffix}_{to_language}.srt"
            
            srt_path = os.path.join(save_dir, srt_filename)
            trans_srt_path = os.path.join(save_dir, trans_srt_filename)
            srt_line = []
            trans_srt_line = []
            
            # 根据不同级别生成字幕
            if srt_level == "segment":
                # Segment 级别：原始大段落（未对齐前的）
                for i, seg in enumerate(tqdm(original_segments, desc="Generating segment-level SRT...", total=len(original_segments))):
                    start = timedelta(seconds=seg['start'])
                    end = timedelta(seconds=seg['end'])
                    content = seg['text']
                    srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))
                    
                    if if_translate:
                        translated = ts.translate_text(query_text=content, translator=translator, to_language=to_language)
                        trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=translated))
            
            elif srt_level == "sentence":
                # Sentence 级别：使用内置的 PunktSentenceTokenizer 分割
                # align() 函数已经按句子分割好了（merge_sentences=False）
                for i, seg in enumerate(tqdm(result["segments"], desc="Generating sentence-level SRT...", total=len(result["segments"]))):
                    start = timedelta(seconds=seg['start'])
                    end = timedelta(seconds=seg['end'])
                    content = seg['text']
                    srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))
                    
                    if if_translate:
                        translated = ts.translate_text(query_text=content, translator=translator, to_language=to_language)
                        trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=translated))
                        
            elif srt_level == "word":
                # Word 级别：每个词一个字幕
                word_index = 1
                for seg in tqdm(result["segments"], desc="Generating word-level SRT...", total=len(result["segments"])):
                    if "words" in seg:
                        for word in seg["words"]:
                            if "start" in word and "end" in word:
                                start = timedelta(seconds=word['start'])
                                end = timedelta(seconds=word['end'])
                                content = word['word']
                                srt_line.append(srt.Subtitle(index=word_index, start=start, end=end, content=content))
                                
                                # 词级别不翻译（效果不好）
                                if if_translate:
                                    trans_srt_line.append(srt.Subtitle(index=word_index, start=start, end=end, content=content))
                                
                                word_index += 1
                                
            elif srt_level == "char":
                # Char 级别：每个字符一个字幕
                char_index = 1
                for seg in tqdm(result["segments"], desc="Generating char-level SRT...", total=len(result["segments"])):
                    if "chars" in seg:
                        for char in seg["chars"]:
                            if "start" in char and "end" in char and char['start'] is not None:
                                start = timedelta(seconds=char['start'])
                                end = timedelta(seconds=char['end'])
                                content = char['char']
                                # 跳过空格字符（可选）
                                if content.strip():
                                    srt_line.append(srt.Subtitle(index=char_index, start=start, end=end, content=content))
                                    
                                    # 字符级别不翻译
                                    if if_translate:
                                        trans_srt_line.append(srt.Subtitle(index=char_index, start=start, end=end, content=content))
                                    
                                    char_index += 1
            
            # 生成 SRT 字符串内容
            srt_string = srt.compose(srt_line)
            trans_srt_string = srt.compose(trans_srt_line) if if_translate else ""
            
            # 写入字幕文件
            with open(srt_path, 'w', encoding="utf-8") as f:
                f.write(srt_string)
            print(f"[WhisperX] Subtitle saved to: {srt_path}")
            
            # 只在翻译时才写入翻译文件
            if if_translate:
                with open(trans_srt_path, 'w', encoding="utf-8") as f:
                    f.write(trans_srt_string)
                
                # 返回结果和下载信息
                return {
                    "result": (srt_path, trans_srt_path, srt_string, trans_srt_string),
                    "ui": {
                        "subtitle": [
                            {
                                "filename": srt_filename,
                                "subfolder": subdir,
                                "type": "output"
                            },
                            {
                                "filename": trans_srt_filename,
                                "subfolder": subdir,
                                "type": "output"
                            }
                        ]
                    }
                }
            else:
                # 返回结果和下载信息
                return {
                    "result": (srt_path, srt_path, srt_string, srt_string),
                    "ui": {
                        "subtitle": [{
                            "filename": srt_filename,
                            "subfolder": subdir,
                            "type": "output"
                        }]
                    }
                }
        
        finally:
            # 清理临时文件
            if temp_path is not None and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    print(f"[WhisperX] Cleaned up temp file: {temp_path}")
                except Exception as e:
                    print(f"[WhisperX] Failed to clean up temp file: {e}")

class LoadAudioPath:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["wav", "mp3","WAV","flac","m4a", "mp4"]]
        return {"required":
                    {"audio": (sorted(files),)},
                }

    CATEGORY = "WhisperX"

    RETURN_TYPES = ("AUDIOPATH",)
    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_path = folder_paths.get_annotated_filepath(audio)
        return (audio_path,)
