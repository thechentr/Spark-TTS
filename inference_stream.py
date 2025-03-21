import torch
import soundfile as sf
import numpy as np
from cli.SparkTTS import SparkTTS
import re

def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model

model = initialize_model()

def make_prompt(model, text, prompt_audio_path, prompt_text=None):

    # Perform inference and save the output audio
    with torch.no_grad():
        model_prompt, global_token_ids = model.process_prompt(
                text, prompt_audio_path, prompt_text
            )
    

    start_idx = model_prompt.find("<|start_global_token|>")
    voice_params = model_prompt[start_idx:]
    model_prompt = model_prompt[:start_idx]

    print(voice_params)
    print(model_prompt)

    pattern = r"<\|start_global_token\|>(.*?)<\|end_global_token\|>"
    match = re.search(pattern, voice_params, re.DOTALL)
    content = match.group(1)
    ids = re.findall(r"<\|bicodec_global_(\d+)\|>", content)
    int_ids = [int(i) for i in ids]
    print(int_ids)

    return model_prompt, int_ids


text = "Hi~, I am Kai from FlashIntel. I am reaching out because we found that you have visited our website recently. Are you available for a quick chat?"
prompt_audio_path = "assets\\Ben_promptvn.wav"
model_prompt, int_ids = make_prompt(model, text, prompt_audio_path)

def voice_clone_with_model_input(model, model_prompt: str, *voice_params):
    if isinstance(voice_params[0], list):
        voice_params = voice_params[0]
    else:
        voice_params = list(voice_params)
    print('voice_params: ', voice_params)
    # voice_params = voice_params.values.flatten().astype(int).tolist()  # <class 'pandas.core.frame.DataFrame'>
    tokens = "".join(f"<|bicodec_global_{i}|>" for i in voice_params)
    new_bicodec_global = f"<|start_global_token|>{tokens}<|end_global_token|>"
    model_prompt = model_prompt + new_bicodec_global

    print(model_prompt)
    

    wav = model.inference_with_prompt_orig(model_prompt)
    return  wav


# 拼接所有生成的音频片段
full_audio = voice_clone_with_model_input(model, model_prompt, int_ids)

# 保存输出音频到文件
output_path = "output.wav"
sf.write(output_path, full_audio, 16000)
print("Audio saved to:", output_path)