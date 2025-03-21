import io
import torch
import soundfile as sf
import numpy as np
import time
import logging
from flask import Flask, request, send_file, jsonify
from cli.SparkTTS import SparkTTS
import re

app = Flask(__name__)

# 配置日志输出
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

VOICE_GLOBAL_TOKEN = {
    'Mike': [3347, 3134, 4091, 4089, 2407, 1177, 1636, 1373, 3162, 2361, 3065,
             3186, 938, 3376, 3193, 1785, 3327, 2793, 3429, 1470, 3416, 493,
             3570, 3255, 2298, 1549, 3543, 2662, 3238, 2691, 1960, 3222],
    'Alex': [1239, 3377, 1464, 3601, 1315, 1435, 2486, 371, 3149, 2070, 1269, 1326,
             531, 3624, 3196, 1303, 1137, 1320, 2948, 315, 194, 890, 252, 201,
             1461, 794, 3633, 2196, 99, 2624, 2091, 3163],
    'John': [2099, 3090, 1989, 3540, 539, 262, 999, 471, 238, 1131, 3350,
             127, 114, 3376, 3317, 2756, 4091, 2018, 3070, 796, 1739, 333,
             501, 1587, 881, 2313, 4042, 3824, 183, 1307, 3560, 705],
    'Susan': [2338, 2530, 2043, 3521, 2729, 265, 997, 695, 586, 2056, 3616,
              252, 746, 3640, 2222, 3554, 3703, 1384, 3818, 2517, 2184, 1820,
              435, 2866, 2930, 3009, 4046, 2464, 1470, 1390, 1496, 390]
}


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0) -> SparkTTS:
    """加载模型，程序启动时只加载一次。"""
    device = torch.device(f"cuda:{device}")
    model = SparkTTS(model_dir, device)
    logging.info("模型加载成功")
    return model


def generate_wav(model: SparkTTS, voice_name: str, text: str):
    """根据给定文本和声音名称生成语音数据。"""
    assert voice_name in VOICE_GLOBAL_TOKEN.keys(), f'Do not support {voice_name}'
    global_token_ids = VOICE_GLOBAL_TOKEN[voice_name]
    wav = model.inference_prompt(text, global_token_ids)
    return wav


def speed_test(model: SparkTTS, voice_name: str, text: str, iterations: int = 10):
    """对生成函数进行测速，记录每次生成所需时间并计算平均时间。"""
    times = []
    for i in range(iterations):
        start_time = time.time()
        _ = generate_wav(model, voice_name, text)
        elapsed = time.time() - start_time
        times.append(elapsed)
        logging.info(f"第 {i+1} 次生成耗时：{elapsed:.4f} 秒")
    avg_time = sum(times) / len(times)
    logging.info(f"共 {iterations} 次生成的平均耗时：{avg_time:.4f} 秒")
    return times


def main():
    model = initialize_model()
    voice_name = 'Mike'
    text = f"Hi, I am {voice_name} from FlashIntel. What can I help?"
    
    logging.info("开始进行测速测试（10次）...")
    speed_test(model, voice_name, text, iterations=10)
    
    # 生成一次语音并保存为文件
    wav = generate_wav(model, voice_name, text)
    sf.write('output.wav', wav, 16000)
    logging.info("语音生成完成，保存为 output.wav")


if __name__ == '__main__':
    main()
