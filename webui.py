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


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0):
    """Load the model once at the beginning."""
    logging.info(f"Loading model from: {model_dir}")
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    return model




def build_ui(model_dir, device=0):
    
    # Initialize model
    model = initialize_model(model_dir, device=device)

    

    def make_prompt(text, prompt_text, prompt_wav_upload, prompt_wav_record):

        prompt_speech = prompt_wav_upload if prompt_wav_upload else prompt_wav_record
        prompt_text = None if len(prompt_text) < 2 else prompt_text

        if prompt_text is not None:
            prompt_text = None if len(prompt_text) <= 1 else prompt_text


        # Perform inference and save the output audio
        with torch.no_grad():
            model_prompt, global_token_ids = model.process_prompt(
                    text, prompt_speech, prompt_text
                )
        
        model_prompt = model_prompt.replace("<|start_content|>", "\n<|start_content|>\n")
        model_prompt = model_prompt.replace("<|end_content|>", "\n<|end_content|>\n")
        model_prompt = model_prompt.replace("<|task_tts|>", "\n<|task_tts|>\n")
        model_prompt = model_prompt.replace("<|start_global_token|>", "\n<|start_global_token|>\n")
        model_prompt = model_prompt.replace("<|end_global_token|>", "\n<|end_global_token|>\n")
        model_prompt = model_prompt.replace("|bicodec_global_", "@")

        count = model_prompt.count("@")
        print("bicodec global: ", count)
        return model_prompt

    # Define callback function for voice cloning
    def voice_clone_with_model_input(model_prompt: str):
        """
        Gradio callback to clone voice using text and optional prompt speech.
        - text: The input text to be synthesised.
        - prompt_text: Additional textual info for the prompt (optional).
        - prompt_wav_upload/prompt_wav_record: Audio files used as reference.
        """
        model_prompt = model_prompt.replace("\n<|start_content|>\n", "<|start_content|>")
        model_prompt = model_prompt.replace("\n<|end_content|>\n", "<|end_content|>")
        model_prompt = model_prompt.replace("\n<|task_tts|>\n", "<|task_tts|>")
        model_prompt = model_prompt.replace("\n<|start_global_token|>\n", "<|start_global_token|>")
        model_prompt = model_prompt.replace("\n<|end_global_token|>\n", "<|end_global_token|>")
        model_prompt = model_prompt.replace("@", "|bicodec_global_")
        

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

    # Define callback function for creating new voices
    def voice_creation(text, gender, pitch, speed):
        """
        Gradio callback to create a synthetic voice with adjustable parameters.
        - text: The input text for synthesis.
        - gender: 'male' or 'female'.
        - pitch/speed: Ranges mapped by LEVELS_MAP_UI.
        """
        pitch_val = LEVELS_MAP_UI[int(pitch)]
        speed_val = LEVELS_MAP_UI[int(speed)]
        audio_output_path = run_tts(
            text,
            model,
            gender=gender,
            pitch=pitch_val,
            speed=speed_val
        )
        return audio_output_path

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
                    prompt_wav_record = gr.Audio(
                        sources="microphone",
                        type="filepath",
                        label="Record the prompt audio file.",
                    )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Text", lines=3, placeholder="Enter text here", value="The combinations of different textures and flavors create a perfect harmony. The succulence of the steak, the tartness of the cranberries, the crunch of pine nuts, and creaminess of blue cheese make it a truly delectable delight. Enjoy your culinary adventure!"
                    )
                    prompt_text_input = gr.Textbox(
                        label="Text of prompt speech (Optional; recommended for cloning in the same language.)",
                        lines=3,
                        placeholder="Enter text of the prompt speech.",
                    )
                

                model_input = gr.Textbox(
                    label="model_input", lines=10, placeholder="Enter text here"
                )

                gr.Markdown(
                    "### <|start_style_label|><|gender_0|><|pitch_label_4|><|speed_label_4|><|end_style_label|>")

                generate_model_input = gr.Button("Generate model input")

                generate_model_input.click(
                    make_prompt,
                    inputs=[text_input, prompt_text_input, prompt_wav_upload, prompt_wav_record],
                    outputs=[model_input],
                )


                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )

                generate_buttom_clone_with_model_input = gr.Button("Generate with model input")

                generate_buttom_clone_with_model_input.click(
                    voice_clone_with_model_input,
                    inputs=[model_input],
                    outputs=[audio_output],
                )


                generate_buttom_clone = gr.Button("Generate voice in one time")

                generate_buttom_clone.click(
                    voice_clone,
                    inputs=[
                        text_input,
                        prompt_text_input,
                        prompt_wav_upload,
                        prompt_wav_record,
                    ],
                    outputs=[audio_output],
                )

            # Voice Creation Tab
            with gr.TabItem("Voice Creation"):
                gr.Markdown(
                    "### Create your own voice based on the following parameters"
                )

                with gr.Row():
                    with gr.Column():
                        gender = gr.Radio(
                            choices=["male", "female"], value="male", label="Gender"
                        )
                        pitch = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Pitch"
                        )
                        speed = gr.Slider(
                            minimum=1, maximum=5, step=1, value=3, label="Speed"
                        )
                    with gr.Column():
                        text_input_creation = gr.Textbox(
                            label="Input Text",
                            lines=3,
                            placeholder="Enter text here",
                            value="You can generate a customized voice by adjusting parameters such as pitch and speed.",
                        )
                        create_button = gr.Button("Create Voice")

                audio_output = gr.Audio(
                    label="Generated Audio", autoplay=True, streaming=True
                )
                create_button.click(
                    voice_creation,
                    inputs=[text_input_creation, gender, pitch, speed],
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