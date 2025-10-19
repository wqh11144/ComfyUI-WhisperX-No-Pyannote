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

## Tutorial
[Demo](https://www.bilibili.com/video/BV19i421y7jb/)

## WeChat Group && Donate
<div>
  <figure>
  <img alt='Wechat' src="wechat.jpg?raw=true" width="300px"/>
  <img alt='donate' src="donate.jpg?raw=true" width="300px"/>
  <figure>
</div>

## Thanks
- [whisperX](https://github.com/m-bain/whisperX.git)
- [translators](https://github.com/UlionTse/translators)
