#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FastAPI 服务：说话人日志 → ASR → 标点 → 角色映射（医生/顾客）
同时支持：
  - /transcribe: 角色分离（需主音频 + 医生注册音频）
  - /asr: 非流式 ASR + 热词（单音频）
  - /ws/asr: 滑动窗口 WebSocket ASR + 热词（实时）
"""

import os
import json
import tempfile
import time
import torch
import numpy as np
import logging
import asyncio
import io
import traceback
import uuid
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import soundfile

# ==============================
# 🔧 日志 & 配置
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ASRService")
# 输出目录
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.7

# 模型路径（支持热词的 Paraformer）
VAD_MODEL_PATH = "/home/dieu/.cache/modelscope/hub/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_PATH = "/home/dieu/.cache/modelscope/hub/models/damo/speech_campplus_speaker-diarization_common"
ASR_MODEL_PATH = "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"  # 支持 hotword
PUNC_MODEL_PATH = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
XVECTOR_MODEL_PATH = "iic/speech_campplus_sv_zh-cn_16k-common"

# 全局模型 & 热词
vad_model = None
sd_model = None
asr_model = None
punc_model = None
xvector_model = None
global_hotword_str = ""

app = FastAPI(title="Speaker Diarization + ASR + Hotword Service", version="1.0")

# ==============================
# 🧰 工具函数
# ==============================
def load_hotwords(file_path: str = "hotwords.txt") -> str:
    """从文本文件加载热词，每行一个，返回空格拼接的字符串"""
    if not os.path.exists(file_path):
        logger.warning(f"Hotword file {file_path} not found. Using empty hotword.")
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            hotwords = [line.strip() for line in f if line.strip()]
        hotword_str = " ".join(hotwords)
        logger.info(f"Loaded {len(hotwords)} hotwords from {file_path}.")
        return hotword_str
    except Exception as e:
        logger.error(f"Failed to load hotwords from {file_path}: {e}")
        return ""

def check_keywords(text: str):
    keywords = ["紧急", "报警", "危险"]
    for kw in keywords:
        if kw in text:
            logger.warning(f"⚠️ Keyword detected: '{kw}' in text: '{text}'")

def get_embedding(audio_path):
    result = xvector_model.generate(input=audio_path)
    if result and isinstance(result[0], dict):
        emb = result[0].get("spk_embedding")
        if emb is not None:
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            return emb.flatten()
    raise ValueError(f"无法提取 embedding: {audio_path}")

def format_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    ms = int(round(seconds * 1000))
    h, m, s = ms // 3600000, (ms % 3600000) // 60000, (ms % 60000) // 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms % 1000:03d}"

def save_upload_file(upload_file: UploadFile, suffix: str = ".wav") -> str:
    try:
        contents = upload_file.file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            return tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {str(e)}")
    finally:
        upload_file.file.close()

# ==============================
# 🚀 启动时加载模型
# ==============================
@app.on_event("startup")
def load_models():
    global vad_model, sd_model, asr_model, punc_model, xvector_model, global_hotword_str
    logger.info("🚀 正在加载模型...")
    global_hotword_str = load_hotwords("hotwords.txt")

    try:
        vad_model = AutoModel(
            model=VAD_MODEL_PATH,
            model_revision="v2.0.4",
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        sd_model = pipeline(
            task='speaker-diarization',
            model=SD_MODEL_PATH,
            model_revision="v1.0.0",
            vad_model=VAD_MODEL_PATH,
            vad_model_revision="v2.0.4",
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        asr_model = pipeline(
            task=Tasks.auto_speech_recognition,
            model=ASR_MODEL_PATH,
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        punc_model = AutoModel(
            model=PUNC_MODEL_PATH,
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        xvector_model = AutoModel(
            model=XVECTOR_MODEL_PATH,
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        logger.info("✅ 所有模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise

# ==============================
# 🌐 API 接口
# ==============================

@app.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(..., description="主对话音频文件 (WAV/MP3等)"),
    doctor_enroll: UploadFile = File(..., description="医生注册音频文件 (16kHz, mono)")
):
    start_time = time.time()
    main_audio_path = None
    doctor_audio_path = None
    full_audio = None

    try:
        main_audio_path = save_upload_file(audio)
        doctor_audio_path = save_upload_file(doctor_enroll)

        full_audio = AudioSegment.from_file(main_audio_path).set_frame_rate(16000).set_channels(1)

        # 说话人日志
        sd_result = sd_model(main_audio_path)
        raw_segments = sd_result.get("text", [])
        if not raw_segments:
            raise HTTPException(status_code=400, detail="未检测到任何语音片段")

        # 提取每个说话人 embedding（使用第一段）
        spk_to_embedding = {}
        spk_to_segments = {}

        for start_sec, end_sec, spk_id in raw_segments:
            spk_key = f"spk{spk_id}"
            if spk_key not in spk_to_segments:
                spk_to_segments[spk_key] = []
            spk_to_segments[spk_key].append((start_sec, end_sec))

        for spk_key, seg_list in spk_to_segments.items():
            start_sec, end_sec = seg_list[0]
            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            seg_audio = full_audio[start_ms:end_ms]
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                seg_path = f.name
                seg_audio.export(seg_path, format="wav")
            try:
                emb = get_embedding(seg_path)
                spk_to_embedding[spk_key] = emb
            finally:
                os.unlink(seg_path)

        # 匹配医生
        doctor_emb = get_embedding(doctor_audio_path)
        doctor_spk = None
        best_sim = -1.0
        for spk_key, emb in spk_to_embedding.items():
            sim = float(np.dot(doctor_emb, emb))
            if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
                best_sim = sim
                doctor_spk = spk_key

        if doctor_spk is None:
            doctor_spk = list(spk_to_embedding.keys())[0] if spk_to_embedding else "spk0"

        # ASR + 标点 + 构建结果
        final_segments = []
        for start_sec, end_sec, spk_id in raw_segments:
            spk_key = f"spk{spk_id}"
            role = 0 if spk_key == doctor_spk else 1

            start_ms = int(start_sec * 1000)
            end_ms = int(end_sec * 1000)
            seg_audio = full_audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                seg_path = f.name
                seg_audio.export(seg_path, format="wav")

            try:
                asr_res = asr_model(seg_path, hotword=global_hotword_str)
                text = ""
                if isinstance(asr_res, list) and len(asr_res) > 0:
                    text = asr_res[0].get("text", "").strip()
                elif isinstance(asr_res, dict):
                    text = asr_res.get("text", "").strip()

                if text:
                    punc_res = punc_model.generate(input=text)
                    text = punc_res[0].get("text", text) if punc_res else text

                if text:
                    speaker_label = "doctor" if role == 0 else "customer"
                    final_segments.append({
                        "time": format_time(start_sec),
                        "endTime": format_time(end_sec),
                        "role": speaker_label,
                        "content": text
                    })
            finally:
                os.unlink(seg_path)

        total_time = time.time() - start_time
        result = {
            "status": "success",
            "processing_time_seconds": round(total_time, 2),
            "segments": final_segments
        }

        output_filename = f"transcript_{uuid.uuid4().hex[:8]}_{int(time.time())}.json"
        output_json_path = OUTPUT_DIR / output_filename
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return JSONResponse(content=result)

    except Exception as e:
        error_detail = f"处理失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_detail)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [main_audio_path, doctor_audio_path]:
            if path and os.path.exists(path):
                os.unlink(path)

@app.post("/asr")
async def asr_with_hotwords(audio: UploadFile = File(..., description="音频文件 (WAV/MP3等)")):
    """
    非流式 ASR 接口，支持热词增强
    """
    temp_path = None
    try:
        temp_path = save_upload_file(audio)

        result = asr_model(temp_path, hotword=global_hotword_str)
        text = ""
        if isinstance(result, list) and len(result) > 0:
            text = result[0].get("text", "").strip()
        elif isinstance(result, dict):
            text = result.get("text", "").strip()

        # 可选：加标点
        if text and punc_model:
            punc_res = punc_model.generate(input=text)
            text = punc_res[0].get("text", text) if punc_res else text

        return JSONResponse(content={
            "status": "success",
            "text": text,
            "hotwords_used": global_hotword_str
        })

    except Exception as e:
        logger.error(f"ASR 处理失败: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.websocket("/ws/asr")
async def websocket_asr(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected.")

    sample_rate = 16000
    audio_buffer = np.array([], dtype=np.float32)

    WINDOW_DURATION = 3.0
    STEP_DURATION = 1.0

    window_samples = int(WINDOW_DURATION * sample_rate)
    step_samples = int(STEP_DURATION * sample_rate)
    total_processed_samples = 0

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                audio_bytes = data["bytes"]
                try:
                    wav_io = io.BytesIO(audio_bytes)
                    audio_chunk, sr = soundfile.read(wav_io, dtype='float32')

                    if sr != sample_rate:
                        await websocket.send_text(json.dumps({
                            "error": f"Unsupported sample rate: {sr}Hz. Expected {sample_rate}Hz."
                        }))
                        continue

                    if audio_chunk.ndim > 1:
                        audio_chunk = audio_chunk[:, 0]

                    audio_buffer = np.concatenate([audio_buffer, audio_chunk])

                    while len(audio_buffer) >= window_samples:
                        window_audio = audio_buffer[:window_samples]
                        seg_start = total_processed_samples / sample_rate
                        seg_end = (total_processed_samples + window_samples) / sample_rate

                        loop = asyncio.get_event_loop()
                        try:
                            res = await loop.run_in_executor(
                                None,
                                lambda: asr_model(window_audio, hotword=global_hotword_str)
                            )
                            text = ""
                            if isinstance(res, list) and len(res) > 0:
                                text = res[0].get("text", "")
                            elif isinstance(res, dict):
                                text = res.get("text", "")
                            text = text.strip()

                            await websocket.send_text(json.dumps({
                                "corrected": text,
                                "segment_start": round(seg_start, 2),
                                "segment_end": round(seg_end, 2)
                            }))

                            if text:
                                check_keywords(text)

                        except Exception as e:
                            logger.error(f"ASR recognition failed: {e}")
                            await websocket.send_text(json.dumps({"error": "ASR recognition failed"}))

                        audio_buffer = audio_buffer[step_samples:]
                        total_processed_samples += step_samples

                except Exception as e:
                    logger.error(f"Error processing audio chunk: {e}")
                    await websocket.send_text(json.dumps({"error": "Audio processing failed"}))

            elif "text" in data:
                try:
                    msg = json.loads(data["text"])
                    if msg.get("is_final"):
                        final_text = ""
                        if len(audio_buffer) > 0:
                            loop = asyncio.get_event_loop()
                            try:
                                res = await loop.run_in_executor(
                                    None,
                                    lambda: asr_model(audio_buffer, hotword=global_hotword_str)
                                )
                                if isinstance(res, list) and len(res) > 0:
                                    final_text = res[0].get("text", "")
                                elif isinstance(res, dict):
                                    final_text = res.get("text", "")
                                final_text = final_text.strip()
                            except Exception as e:
                                logger.error(f"Final ASR failed: {e}")

                        await websocket.send_text(json.dumps({
                            "final": final_text,
                            "corrected_final": final_text
                        }))
                        break

                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({"error": "Invalid JSON"}))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await websocket.close()

@app.get("/")
def read_root():
    return {
        "message": "欢迎使用说话人日志 + ASR + 热词服务！",
        "endpoints": {
            "角色分离": "POST /transcribe (audio + doctor_enroll)",
            "非流式ASR": "POST /asr (audio)",
            "流式ASR": "WebSocket /ws/asr"
        }
    }

# ==============================
# 🏁 启动入口
# ==============================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ASR server on http://0.0.0.0:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
