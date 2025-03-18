# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import soundfile as sf
import logging
import argparse
import gradio as gr
from datetime import datetime
from cli.SparkTTS import SparkTTS
from sparktts.utils.token_parser import LEVELS_MAP_UI

import re
import ast


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model




def build_ui(model_dir, device=0):
    
    # Initialize model
    model = initialize_model(model_dir, device=device)

    

    def make_prompt(text, prompt_text, prompt_wav_upload, voice_ids):
        
        print('voice_ids: ', voice_ids)
        if voice_ids:
            voice_ids = ast.literal_eval(voice_ids)
        
        prompt_speech = prompt_wav_upload
        prompt_text = None if len(prompt_text) < 2 else prompt_text

        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text


        # Perform inference and save the output audio
        with torch.no_grad():
            model_prompt, global_token_ids = model.process_prompt(
                    text, prompt_speech, prompt_text
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
        if voice_ids:
            int_ids = voice_ids
        print(int_ids)

        return model_prompt, int_ids

    # Define callback function for voice cloning
    def voice_clone_with_model_input(model_prompt: str, *voice_params):
        voice_params = list(voice_params)
        print(voice_params)
        # voice_params = voice_params.values.flatten().astype(int).tolist()  # <class 'pandas.core.frame.DataFrame'>
        tokens = "".join(f"<|bicodec_global_{i}|>" for i in voice_params)
        new_bicodec_global = f"<|start_global_token|>{tokens}<|end_global_token|>"
        model_prompt = model_prompt + new_bicodec_global

        print(model_prompt)
        

        with torch.no_grad():
            generater = model.inference_with_prompt(model_prompt)
            for chunk_wav in generater:
                # <class 'numpy.ndarray'>
                yield (16000, chunk_wav)


    # Define callback function for voice cloning
    def voice_clone(text, prompt_text, prompt_wav_upload, prompt_wav_record):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text = None if len(prompt_text) < 2 else prompt_text

        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text

        with torch.no_grad():
            generater = model.inference(
                text,
                prompt_speech_path=prompt_speech,
                prompt_text=prompt_text
            )
            for chunk_wav in generater:
                # <class 'numpy.ndarray'>
                yield (16000, chunk_wav)


    with gr.Blocks() as demo:
        # Use HTML for centered title
        gr.HTML('<h1 style="text-align: center;">Spark-TTS by SparkAudio</h1>')
        with gr.Tabs():
            # Voice Clone Tab
            with gr.TabItem("Voice Clone"):
                gr.Markdown(
                    "### Upload reference audio or recording （上传参考音频或者录音）"
                )

                with gr.Row():
                    prompt_wav_upload = gr.Audio(
                        sources="upload",
                        type="filepath",
                        value="./assets/trump_en.wav",
                        label="Choose the prompt audio file, ensuring the sampling rate is no lower than 16kHz.",
                    )
                    voice_ids = gr.Textbox(
                        label="voice ids", lines=3, placeholder="[id1, id2, ..., id32]"
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here", value="Hi, I am Kai from FlashIntel. I am reaching out because we found that you have visited our website recently. Are you available for a quick chat?"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )
                

                model_input = gr.Textbox(
                    label="model_input", lines=3, placeholder="Enter text here"
                )


                # 创建32个 
                int_numbers = []
                for i in range(5):
                    with gr.Row():
                        nrow = 7 if i < 4 else 4
                        for j in range(nrow):
                            int_numbers.append(gr.Number(value=0, label=f"P{i*8+j}"))

                generate_model_input = gr.Button("Generate model input")

                # 当点击按钮时调用 make_prompt，返回 model_input 和一个包含32个整数的列表，
                # 我们使用 * 运算符将列表中的每个数字拆分为独立的输出，更新32个 Number
                generate_model_input.click(
                    fn=lambda text, prompt_text, wav_upload, voice_ids: (
                        make_prompt(text, prompt_text, wav_upload, voice_ids)[0],
                        *make_prompt(text, prompt_text, wav_upload, voice_ids)[1]
                    ),
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, voice_ids],
                    outputs=[model_input] + int_numbers,
                )

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone_with_model_input = gr.Button("Generate with model input")

                # 此处将 model_input 和32个 Slider 的值作为输入传入 voice_clone_with_model_input
                generate_buttom_clone_with_model_input.click(
                    voice_clone_with_model_input,
                    inputs=[model_input] + int_numbers,
                    outputs=[audio_output],
                )


                generate_buttom_clone = gr.Button("Generate voice in one time")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        voice_ids,
                    ],
                    outputs=[audio_output],
                )

    return demo


def parse_arguments():
    """
    Parse command-line arguments such as model directory and device ID.
    """
    parser = argparse.ArgumentParser(description="Spark TTS Gradio server.")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="ID of the GPU device to use (e.g., 0 for cuda:0)."
    )
    parser.add_argument(
        "--server_name",
        type=str,
        default="0.0.0.0",
        help="Server host/IP for Gradio app."
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=7860,
        help="Server port for Gradio app."
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()

    # Build the Gradio demo by specifying the model directory and GPU device
    demo = build_ui(
        model_dir=args.model_dir,
        device=args.device
    )

    # Launch Gradio with the specified server name and port
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port
    )