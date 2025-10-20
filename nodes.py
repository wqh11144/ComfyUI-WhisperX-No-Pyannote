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
        return {"required":
                    {"audio": ("AUDIOPATH",),
                     "model_type":(model_list,{
                         "default": "large-v3"
                     }),
                     "batch_size":("INT",{
                         "default": 4
                     }),
                     "word_level":("BOOLEAN",{
                         "default": False
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

    def get_srt(self, audio,model_type,batch_size,word_level,if_translate,translator,to_language):
        compute_type = "float16"

        base_name = os.path.basename(audio)[:-4]
        device = "cuda" if cuda_malloc.cuda_malloc_supported() else "cpu"
        # 1. Transcribe with original whisper (batched)
        if model_type == "large-v3-turbo":
            model_type = "deepdml/faster-whisper-large-v3-turbo-ct2"
        model = whisperx.load_model(model_type, device, compute_type=compute_type)
        audio = whisperx.load_audio(audio)
        result = model.transcribe(audio, batch_size=batch_size)
        # print(result["segments"]) # before alignment
        language_code=result["language"]
        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # print(result["segments"]) # after alignment
        
        # delete model if low on GPU resources
        import gc; gc.collect(); torch.cuda.empty_cache(); del model_a,model
        
        level_suffix = "_word" if word_level else ""
        srt_path = os.path.join(out_path,f"{time.time()}_{base_name}{level_suffix}.srt")
        trans_srt_path = os.path.join(out_path,f"{time.time()}_{base_name}{level_suffix}_{to_language}.srt")
        srt_line = []
        trans_srt_line = []
        
        if word_level:
            # 词级别 SRT：为每个词生成一个字幕条目
            word_index = 1
            for seg in tqdm(result["segments"], desc="Generating word-level SRT...", total=len(result["segments"])):
                if "words" in seg:
                    for word in seg["words"]:
                        if "start" in word and "end" in word:
                            start = timedelta(seconds=word['start'])
                            end = timedelta(seconds=word['end'])
                            content = word['word']
                            srt_line.append(srt.Subtitle(index=word_index, start=start, end=end, content=content))
                            
                            if if_translate:
                                # 对于词级别，不翻译单个词（翻译单词效果不好）
                                # 保留原文
                                trans_srt_line.append(srt.Subtitle(index=word_index, start=start, end=end, content=content))
                            
                            word_index += 1
        else:
            # 句子级别 SRT（原有逻辑）
            for i, res in enumerate(tqdm(result["segments"],desc="Generating sentence-level SRT...", total=len(result["segments"]))):
                start = timedelta(seconds=res['start'])
                end = timedelta(seconds=res['end'])
                content = res['text']
                srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))
                if if_translate:
                    #if i== 0:
                       # _ = ts.preaccelerate_and_speedtest() 
                    content = ts.translate_text(query_text=content, translator=translator,to_language=to_language)
                    trans_srt_line.append(srt.Subtitle(index=i+1, start=start, end=end, content=content))
                
        with open(srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(srt_line))
        with open(trans_srt_path, 'w', encoding="utf-8") as f:
            f.write(srt.compose(trans_srt_line))

        if if_translate:
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
