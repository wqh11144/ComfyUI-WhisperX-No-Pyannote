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
input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class PreviewSRT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"srt": ("SRT",)},
                }

    CATEGORY = "WhisperX"

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    
    FUNCTION = "show_srt"

    def show_srt(self, srt):
        srt_name = os.path.basename(srt)
        dir_name = os.path.dirname(srt)
        dir_name = os.path.basename(dir_name)
        with open(srt, 'r') as f:
            srt_content = f.read()
        return {"ui": {"srt":[srt_content,srt_name,dir_name]}}


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
        model_list = ["large-v3","distil-large-v3","large-v2", "large-v3-turbo"]
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
                    {"audio": ("AUDIOPATH",),
                     "model_type":(model_list,{
                         "default": "large-v3"
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
                     })
                     },
                }

    CATEGORY = "WhisperX"

    RETURN_TYPES = ("SRT","SRT")
    RETURN_NAMES = ("ori_SRT","trans_SRT")
    FUNCTION = "get_srt"

    def get_srt(self, audio,model_type,batch_size,srt_level,if_translate,translator,to_language):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        
        # 1. Transcribe with original whisper (batched)
        if model_type == "large-v3-turbo":
            model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"
        
        # 配置 ASR 选项：引导 Whisper 生成标点符号
        # initial_prompt 包含多语言标点示例，引导模型正确输出标点
        asr_options = {
            "initial_prompt": (
                "这是一段普通话对话。包含逗号、句号、感叹号！还有问号？"
                "This is an English sentence. It has commas, periods, exclamation marks! And question marks?"
            ),
        }
        
        model = whisperx.load_model(model_type, device, compute_type=compute_type, asr_options=asr_options)
        audio_data = whisperx.load_audio(audio)
        result = model.transcribe(audio_data, batch_size=batch_size)
        
        language_code=result["language"]
        
        # 保存原始 segments（用于 segment 级别）
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
        
        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a,model
        
        # 生成文件名
        level_suffix = f"_{srt_level}" if srt_level != "segment" else ""
        srt_path = os.path.join(out_path,f"{time.time()}_{base_name}{level_suffix}.srt")
        trans_srt_path = os.path.join(out_path,f"{time.time()}_{base_name}{level_suffix}_{to_language}.srt")
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
        
        # 写入文件
        with open(srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(srt_line))
        
        # 只在翻译时才写入翻译文件
        if if_translate:
            with open(trans_srt_path, 'w', encoding="utf-8") as f:
                f.write(srt.compose(trans_srt_line))
            return (srt_path,trans_srt_path)
        else:
            return (srt_path,srt_path)

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
