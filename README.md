# ComfyUI-WhisperX
a comfyui cuatom node for audio subtitling based on [whisperX](https://github.com/m-bain/whisperX.git) and [translators](https://github.com/UlionTse/translators)
<div>
  <figure>
  <img alt='webpage' src="web.png?raw=true" width="600px"/>
  <figure>
</div>

## Features
- export `srt` file for subtitle was supported
- translate was supported by [translators](https://github.com/UlionTse/translators) with huge number engine
- huge comfyui custom nodes can merge in whisperx
- **🆕 Automatic timestamp overlap fixing** - Detects and fixes overlapping subtitle timestamps (default enabled)
- **🆕 Smart subtitle length control** - Intelligent splitting based on punctuation, conjunctions, and max length (optional)

## How to use
make sure `ffmpeg` is worked in your commandline
for Linux
```
apt update
apt install ffmpeg
```
for Windows,you can install `ffmpeg` by [WingetUI](https://github.com/marticliment/WingetUI) automatically

then!
```
git clone https://github.com/AIFSH/ComfyUI-WhisperX.git
cd ComfyUI-WhisperX
pip install -r requirements.txt
```
`weights` will be downloaded from huggingface automatically! if you in china,make sure your internet attach the huggingface
or if you still struggle with huggingface, you may try follow [hf-mirror](https://hf-mirror.com/) to config your env.

## Advanced Features

### 1. Timestamp Overlap Fixer

WhisperX may generate overlapping timestamps in SRT files. This feature automatically detects and fixes them:

**Parameters:**
- `fix_overlap` (boolean, default: `true`) - Enable/disable automatic fixing
- `gap_ms` (integer, default: `50`) - Gap between subtitles in milliseconds (range: 0-500ms)

**Example:**
```
Before: 00:00:00,200 --> 00:00:01,740  ⚠️ Overlaps with next
After:  00:00:00,200 --> 00:00:01,530  ✅ Fixed with 50ms gap
```

### 2. Smart Subtitle Length Control

Control subtitle length with intelligent splitting for better readability (sentence level only):

**Parameters:**
- `enable_smart_split` (boolean, default: `false`) - Enable/disable smart splitting
- `split_strategy` (dropdown, default: `auto`) - Splitting strategy
  - **`auto`** - Intelligent mode: Auto-adjusts based on language (30 for CJK, 45 for others)
  - **`custom`** - Custom mode: Use your own max/min length
- `custom_max_length` (integer, default: `30`) - Max chars per line (custom mode only)
- `custom_min_length` (integer, default: `20`) - Min chars before split (custom mode only)

**Features:**
- **🤖 Auto mode (recommended)**: Language-aware intelligent splitting
  - Chinese/Japanese/Korean: 30 chars
  - English/European: 45 chars
  - Auto-detects based on `language` parameter
- **⚙️ Custom mode**: Full control with manual length setting
- **Intelligent splitting**: Prioritizes punctuation (`,`), conjunctions (`和`, `but`), then length
- **Preserves timing**: Maintains accurate word-level timestamps

**Example (auto mode, Chinese):**
```
Before (68 chars, too long):
各位听众朋友，今日我要说一个发生在江南某府城的故事，这故事讲的是人心真假。

After (auto split: zh → 30 chars):
1) 各位听众朋友，今日我要说
2) 一个发生在江南某府城的故事，
3) 这故事讲的是人心真假。
```

## Tutorial
[Demo](https://www.bilibili.com/video/BV19i421y7jb/)


## Thanks
- [whisperX](https://github.com/m-bain/whisperX.git)
- [translators](https://github.com/UlionTse/translators)
