import asyncio
import websockets
import librosa
import numpy as np
import json
import sys
import io
import soundfile as sf

# WS_URL = "ws://182.148.54.70:9001/ws/asr"
WS_URL = "wss://asr.goodsop.cn:30001/ws/asr"
# WS_URL = "ws://172.16.20.59:28888/ws/asr"
# WS_URL = "ws://localhost:8003/ws/asr"
AUDIO_FILE = "/home/dieu/下载/药名_川音.wav"
CHUNK_DURATION = 0.6
SAMPLE_RATE = 16000
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# 用于同步：当收到 final 时触发
final_received = asyncio.Event()

async def receive_messages(websocket):
    try:
        async for message in websocket:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    if "partial" in data:
                        print(f"[Partial] {data['partial']}")
                    elif "corrected" in data:
                        seg_info = f"[{data.get('segment_start', '?')}-{data.get('segment_end', '?')}s]"
                        print(f"\033[94m[Corrected {seg_info}] {data['corrected']}\033[0m")
                    elif "final" in data or "corrected_final" in data:
                        # ✅ 关键修复：明确优先 corrected_final
                        final_text = data.get("corrected_final") or data.get("final", "")
                        if isinstance(final_text, str) and final_text.strip():
                            print(f"\033[92m[Final] {final_text}\033[0m")
                        else:
                            print("\033[93m[Final] (no text)\033[0m")
                        final_received.set()
                    elif "error" in data:
                        print(f"\033[91m[Error] {data['error']}\033[0m")
                except json.JSONDecodeError:
                    print(f"[Raw] {message}")
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        final_received.set()
async def stream_audio_to_asr():
    global final_received
    final_received.clear()

    # 加载音频
    try:
        audio_data, _ = librosa.load(AUDIO_FILE, sr=SAMPLE_RATE, mono=True)
        print(f"Loaded audio: {len(audio_data)/SAMPLE_RATE:.2f}s")
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return

    async with websockets.connect(WS_URL) as websocket:
        print("Connected to ASR server.")
        recv_task = asyncio.create_task(receive_messages(websocket))

        # 发送音频 chunks
        total = len(audio_data)
        for i in range(0, total, CHUNK_SAMPLES):
            chunk = audio_data[i:i+CHUNK_SAMPLES]
            # 转为 int16 PCM（推荐）
            audio_int16 = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
            buf = io.BytesIO()
            sf.write(buf, audio_int16, SAMPLE_RATE, format='WAV', subtype='PCM_16')
            await websocket.send(buf.getvalue())
            # 不需要 sleep，除非客户端太快（一般不用）

        print("Finished sending audio.")

        # ✅ 发送结束信号
        await websocket.send(json.dumps({"is_final": True}))
        # await websocket.send(json.dumps({"is_final": True, "enable_full_final": True}))
        # ✅ 等待最终结果（最多 10 秒）
        try:
            await asyncio.wait_for(final_received.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            print("\033[93m[Warning] Timeout: No final result received.\033[0m")

        # 清理
        recv_task.cancel()
        try:
            await recv_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    if len(sys.argv) > 1:
        AUDIO_FILE = sys.argv[1]
    else:
        print(f"Using default: {AUDIO_FILE}")
    asyncio.run(stream_audio_to_asr())