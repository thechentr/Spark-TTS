import io
import torch
import soundfile as sf
import numpy as np
import time
import logging
from cli.SparkTTS_stream import SparkTTS_stream
import re


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


def initialize_model(model_dir="pretrained_models/Spark-TTS-0.5B", device=0) -> SparkTTS_stream:
    """加载模型，程序启动时只加载一次。"""
    device = torch.device(f"cuda:{device}")
    model = SparkTTS_stream(model_dir, device, True)
    logging.info("模型加载成功")
    return model


def generate_wav(model: SparkTTS_stream, voice_name: str, text: str):
    """根据给定文本和声音名称生成语音数据。"""
    assert voice_name in VOICE_GLOBAL_TOKEN.keys(), f'Do not support {voice_name}'
    global_token_ids = VOICE_GLOBAL_TOKEN[voice_name]
    wav = model.inference_prompt(text, global_token_ids)
    return wav


def speed_test(model: SparkTTS_stream, voice_name: str, text: str, iterations: int = 10):
    """对流式生成函数进行测速，记录首音延迟和总体生成时间"""
    overall_times = []
    first_chunk_times = []
    
    for i in range(iterations):
        start = time.time()
        first_chunk_time = None
        
        generator = generate_wav(model, voice_name, text)
        for idx, chunk in enumerate(generator):
            # 第一个数据块产生时记录首音延迟
            if idx == 0:
                first_chunk_time = time.time() - start
            # 可在此处理chunk，例如缓存或直接播放
        
        overall_time = time.time() - start
        overall_times.append(overall_time)
        # 如果没有生成任何chunk，则置为None或0，根据需求处理
        first_chunk_times.append(first_chunk_time if first_chunk_time is not None else 0)
        
        logging.info(f"第 {i+1} 次生成：首音延迟 {first_chunk_time:.4f} 秒, 总体生成时间 {overall_time:.4f} 秒")
    
    avg_first = sum(first_chunk_times) / len(first_chunk_times)
    avg_overall = sum(overall_times) / len(overall_times)
    logging.info(f"平均首音延迟：{avg_first:.4f} 秒")
    logging.info(f"平均总体生成时间：{avg_overall:.4f} 秒")
    
    return first_chunk_times, overall_times



def main():
    model = initialize_model()
    voice_name = 'Mike'
    text = f"Hi, I am {voice_name} from FlashIntel. What can I help?"
    
    logging.info("开始进行测速测试（10次）...")
    speed_test(model, voice_name, text, iterations=10)
    
    # 收集流式生成的所有音频块
    wav_chunks = []
    for chunk in generate_wav(model, voice_name, text):
        wav_chunks.append(chunk)
    
    # 将所有块拼接成一个完整的数组
    # 注意这里假设每个块都是一维的 int16 数组，代表单声道
    full_wav = np.concatenate(wav_chunks, axis=0)
    
    # 保存到 wav 文件
    sf.write('output.wav', full_wav, 16000)
    logging.info("语音生成完成，保存为 output.wav")


if __name__ == '__main__':
    main()
