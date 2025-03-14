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

import re
import torch
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from sparktts.utils.file import load_config
from sparktts.models.audio_tokenizer import BiCodecTokenizer
from sparktts.utils.token_parser import LEVELS_MAP, GENDER_MAP, TASK_TOKEN_MAP


import threading
import time
from transformers.generation.streamers import BaseStreamer
import numpy as np

class GeneratedIDsStreamer(BaseStreamer):
    """
    自定义 streamer，用于流式提取生成的 token ids，并将其扁平化累计，
    当累计的 token 数大于等于 200 时 yield 出这一批 token id（作为一维列表）。
    """
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True):
        # 仅调用父类的默认初始化
        super().__init__()
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self.generated_ids = []  # 用于存放生成的 token id 批次，每个元素通常是 tensor 或列表
        self.lock = threading.Lock()
        self._finished = False
        self._first = True

    def put(self, output_ids):
        """
        在生成过程中，每当生成器输出一批 token ids 时调用，
        将其添加到内部缓冲区中。
        """
        with self.lock:
            if self.skip_prompt and self._first:
                self._first = False
                return
            self.generated_ids.append(output_ids)

    def end(self):
        """
        生成结束时调用，标记内部状态为完成。
        """
        with self.lock:
            self._finished = True

    def __iter__(self):
        accumulated_ids = []
        while True:
            with self.lock:
                if self.generated_ids:
                    accumulated_ids.extend(self.generated_ids)
                    self.generated_ids = []
                elif self._finished:
                    if accumulated_ids:
                        yield accumulated_ids
                    break

            if len(accumulated_ids) >= 100:
                yield accumulated_ids
                accumulated_ids = []
            time.sleep(0.2)


class SparkTTS:
    """
    Spark-TTS for text-to-speech generation.
    """

    def __init__(self, model_dir: Path, device: torch.device = torch.device("cuda:0")):
        """
        Initializes the SparkTTS model with the provided configurations and device.

        Args:
            model_dir (Path): Directory containing the model and config files.
            device (torch.device): The device (CPU/GPU) to run the model on.
        """
        self.device = device
        self.model_dir = model_dir
        self.configs = load_config(f"{model_dir}/config.yaml")
        self.sample_rate = self.configs["sample_rate"]
        self._initialize_inference()

    def _initialize_inference(self):
        """Initializes the tokenizer, model, and audio tokenizer for inference."""
        self.tokenizer = AutoTokenizer.from_pretrained(f"{self.model_dir}/LLM")
        self.model = AutoModelForCausalLM.from_pretrained(f"{self.model_dir}/LLM")
        self.audio_tokenizer = BiCodecTokenizer(self.model_dir, device=self.device)
        self.model.to(self.device)

    def process_prompt(
        self,
        text: str,
        prompt_speech_path: Path,
        prompt_text: str = None,
    ) -> Tuple[str, torch.Tensor]:
        """
        Process input for voice cloning.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.

        Return:
            Tuple[str, torch.Tensor]: Input prompt; global tokens
        """

        global_token_ids, semantic_token_ids = self.audio_tokenizer.tokenize(
            prompt_speech_path
        )
        global_tokens = "".join(
            [f"<|bicodec_global_{i}|>" for i in global_token_ids.squeeze()]
        )

        # Prepare the input tokens for the model
        if prompt_text is not None:
            semantic_tokens = "".join(
                [f"<|bicodec_semantic_{i}|>" for i in semantic_token_ids.squeeze()]
            )
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                prompt_text,
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
                "<|start_semantic_token|>",
                semantic_tokens,
            ]
        else:
            inputs = [
                TASK_TOKEN_MAP["tts"],
                "<|start_content|>",
                text,
                "<|end_content|>",
                "<|start_global_token|>",
                global_tokens,
                "<|end_global_token|>",
            ]

        inputs = "".join(inputs)

        return inputs, global_token_ids

    def process_prompt_control(
        self,
        gender: str,
        pitch: str,
        speed: str,
        text: str,
    ):
        """
        Process input for voice creation.

        Args:
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            text (str): The text input to be converted to speech.

        Return:
            str: Input prompt
        """
        assert gender in GENDER_MAP.keys()
        assert pitch in LEVELS_MAP.keys()
        assert speed in LEVELS_MAP.keys()

        gender_id = GENDER_MAP[gender]
        pitch_level_id = LEVELS_MAP[pitch]
        speed_level_id = LEVELS_MAP[speed]

        pitch_label_tokens = f"<|pitch_label_{pitch_level_id}|>"
        speed_label_tokens = f"<|speed_label_{speed_level_id}|>"
        gender_tokens = f"<|gender_{gender_id}|>"

        attribte_tokens = "".join(
            [gender_tokens, pitch_label_tokens, speed_label_tokens]
        )

        control_tts_inputs = [
            TASK_TOKEN_MAP["controllable_tts"],
            "<|start_content|>",
            text,
            "<|end_content|>",
            "<|start_style_label|>",
            attribte_tokens,
            "<|end_style_label|>",
        ]

        return "".join(control_tts_inputs)
    


    @torch.no_grad()
    def inference(
        self,
        text: str,
        prompt_speech_path: Path = None,
        prompt_text: str = None,
        gender: str = None,
        pitch: str = None,
        speed: str = None,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ):
        """
        Performs inference to generate speech from text, incorporating prompt audio and/or text.

        Args:
            text (str): The text input to be converted to speech.
            prompt_speech_path (Path): Path to the audio file used as a prompt.
            prompt_text (str, optional): Transcript of the prompt audio.
            gender (str): female | male.
            pitch (str): very_low | low | moderate | high | very_high
            speed (str): very_low | low | moderate | high | very_high
            temperature (float, optional): Sampling temperature for controlling randomness. Default is 0.8.
            top_k (float, optional): Top-k sampling parameter. Default is 50.
            top_p (float, optional): Top-p (nucleus) sampling parameter. Default is 0.95.

        Returns:
            torch.Tensor: Generated waveform as a tensor.
        """
        torch.manual_seed(789)
        start = time.time()
        if gender is not None:
            prompt = self.process_prompt_control(gender, pitch, speed, text)

        else:
            prompt, global_token_ids = self.process_prompt(
                text, prompt_speech_path, prompt_text
            )
        print('prompt: ', prompt)
        print('global_token_ids: ', global_token_ids)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)


        streamer = GeneratedIDsStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": 3000,
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        is_first = True
        for generated_ids in streamer:
            # print('get generated_ids: ', generated_ids)
            generated_ids = [torch.cat(generated_ids, dim=0).to(self.device).squeeze(0)]
            # print('converted generated_ids: ', generated_ids)

            # Decode the generated tokens into text
            predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract semantic token IDs from the generated text
            pred_semantic_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
            )

            if gender is not None:
                global_token_ids = (
                    torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", predicts)])
                    .long()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )

            # Convert semantic tokens back to waveform
            wav = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.device).squeeze(0),
                pred_semantic_ids.to(self.device),
            )
            wav_int16 = (wav * 32767).astype(np.int16)

            if is_first:
                print('首音延时: ', time.time() - start)

            is_first = False
            yield wav_int16

    @torch.no_grad()
    def inference_with_prompt(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_k: float = 50,
        top_p: float = 0.95,
    ):
        torch.manual_seed(789)
        start = time.time()
        
        global_token_ids = (
                    torch.tensor([int(token) for token in re.findall(r"bicodec_global_(\d+)", prompt)])
                    .long()
                    .unsqueeze(0)
                    .unsqueeze(0)
                )
        
        print('prompt: ', prompt)
        print('global_token_ids: ', global_token_ids)
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.device)


        streamer = GeneratedIDsStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        generate_kwargs = {
            **model_inputs,
            "max_new_tokens": 3000,
            "do_sample": True,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self.model.generate, kwargs=generate_kwargs)
        thread.start()

        is_first = True
        for generated_ids in streamer:
            # print('get generated_ids: ', generated_ids)
            generated_ids = [torch.cat(generated_ids, dim=0).to(self.device).squeeze(0)]
            # print('converted generated_ids: ', generated_ids)

            # Decode the generated tokens into text
            predicts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # Extract semantic token IDs from the generated text
            pred_semantic_ids = (
                torch.tensor([int(token) for token in re.findall(r"bicodec_semantic_(\d+)", predicts)])
                .long()
                .unsqueeze(0)
            )

            # Convert semantic tokens back to waveform
            wav = self.audio_tokenizer.detokenize(
                global_token_ids.to(self.device).squeeze(0),
                pred_semantic_ids.to(self.device),
            )
            wav_int16 = (wav * 32767).astype(np.int16)

            if is_first:
                print('首音延时: ', time.time() - start)

            is_first = False
            yield wav_int16