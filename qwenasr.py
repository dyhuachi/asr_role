import requests
import numpy as np
import soundfile as sf
import time
import io

# --- 配置区 ---
# qwen-asr-demo-streaming --asr-model-path ./Qwen3-ASR-1.7B --gpu-memory-utilization 0.8 --host 0.0.0.0 --port 8888
# 请确保此地址与你启动 qwen-asr-demo-streaming 时的地址一致
API_BASE = "http://172.16.20.52:8888"
AUDIO_FILE = "/home/dieu/asr_三合一/session-927875fb-b043-42cc-91f4-69efda6f8d25.wav"  # 你本地的录音文件路径
STEP_MS = 1000  # 模拟推流频率（毫秒），每秒发送一次数据


def resample_to_16k(wav, sr):
    """确保音频为 16kHz，这是 Qwen3-ASR 的强制要求"""
    if sr == 16000:
        return wav.astype(np.float32, copy=False)
    duration = wav.shape[0] / float(sr)
    new_samples = int(round(duration * 16000))
    x_old = np.linspace(0.0, duration, num=wav.shape[0], endpoint=False)
    x_new = np.linspace(0.0, duration, num=new_samples, endpoint=False)
    return np.interp(x_new, x_old, wav).astype(np.float32)


def run_simulation():
    # 1. 加载并预处理音频
    try:
        wav, sr = sf.read(AUDIO_FILE)
        wav16k = resample_to_16k(wav, sr)
        print(f"[系统] 成功加载音频: {AUDIO_FILE}, 采样率: {sr}Hz -> 16000Hz")
    except Exception as e:
        print(f"[错误] 无法读取音频文件: {e}")
        return

    # 2. 开启 ASR 会话
    try:
        start_resp = requests.post(f"{API_BASE}/api/start")
        session_id = start_resp.json()["session_id"]
        print(f"[系统] 会话已开启，ID: {session_id}")
    except Exception as e:
        print(f"[错误] 无法连接到 ASR 服务: {e}")
        return

    # 3. 循环分块推流
    samples_per_step = int(16000 * (STEP_MS / 1000.0))
    pos = 0
    print("[系统] 开始模拟实时推流...")

    try:
        while pos < wav16k.shape[0]:
            # 取出一小段音频
            chunk = wav16k[pos: pos + samples_per_step]
            pos += chunk.shape[0]

            # 发送到 /api/chunk 接口
            # 注意：必须以 application/octet-stream 格式发送 float32 原始字节
            response = requests.post(
                f"{API_BASE}/api/chunk?session_id={session_id}",
                data=chunk.tobytes(),
                headers={"Content-Type": "application/octet-stream"}
            )

            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "")
                lang = result.get("language", "")
                if text:
                    print(f"\r[{lang}] 识别中: {text}", end="", flush=True)
            else:
                print(f"\n[错误] Chunk 发送失败: {response.text}")

            # 模拟真实的流速，避免瞬间发完
            time.sleep(STEP_MS / 1000.0)

        # 4. 结束并获取最终校准文本
        print("\n[系统] 音频文件播放完毕，进行最终收尾...")
        finish_resp = requests.post(f"{API_BASE}/api/finish?session_id={session_id}")
        if finish_resp.status_code == 200:
            final_result = finish_resp.json()
            print(f"[最终结果] {final_result.get('text', '')}")

    except Exception as e:
        print(f"\n[错误] 推流过程发生异常: {e}")


if __name__ == "__main__":
    run_simulation()