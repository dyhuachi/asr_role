#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FastAPI 服务：说话人日志 → ASR → 标点 → 角色映射（医生/顾客）
✅ 支持两种调用方式：
  1. 上传文件（multipart/form-data） → 用于本地测试
     - /transcribe: audio + doctor_enroll (File)
     - /asr: audio (File)
  2. 传入 URL（application/json） → 用于生产
     - /transcribe: { "audio_url": "...", "doctor_enroll_url": "..." }
     - /asr: { "audio_url": "..." }
"""
import uuid
import os
import json
import tempfile
import time
import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

import httpx
from urllib.parse import urlparse

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from pydub import AudioSegment
from funasr import AutoModel
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# ==============================
# 🔧 日志 & 配置
# ==============================
# 删除原来的 logging.basicConfig 和 setLevel
logger = logging.getLogger("ASRService")

# 强制设置 logger 级别
logger.setLevel(logging.INFO)

# 防止日志传递给 root logger（避免被 root 的 WARNING 级别过滤）
logger.propagate = False

# 如果还没有 handler，手动添加一个（确保输出到控制台）
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# --- 模型缓存目录 ---
os.environ["MODELSCOPE_CACHE"] = "./"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.7

# 模型路径
VAD_MODEL_PATH = "./models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
SD_MODEL_PATH = "./models/damo/speech_campplus_speaker-diarization_common"
ASR_MODEL_PATH = "./models/iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"
PUNC_MODEL_PATH = "./models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
XVECTOR_MODEL_PATH = "./models/iic/speech_campplus_sv_zh-cn_16k-common"

# 全局模型 & 热词
vad_model = None
sd_model = None
asr_model = None
punc_model = None
xvector_model = None
global_hotword_str = ""

app = FastAPI(title="Speaker Diarization + ASR + Hotword Service", version="2.0 (Upload + URL)")
# from starlette.exceptions import HTTPException as StarletteHTTPException
#
# @app.exception_handler(StarletteHTTPException)
# async def http_exception_handler(request: Request, exc: StarletteHTTPException):
#     return JSONResponse(
#         status_code=exc.status_code,
#         content={"code": 0, "detail": exc.detail}
#     )
# ==============================
# 📥 数据模型（仅用于 URL）
# ==============================
class ASRURLRequest(BaseModel):
    audio_url: str

class TranscribeURLRequest(BaseModel):
    audio_url: str
    doctor_enroll_url: str

# ==============================
# 🧰 工具函数
# ==============================
def load_hotwords(file_path: str = "hotwords.txt") -> str:
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

def download_audio_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL")

        with httpx.Client(timeout=21600.0) as client:
            response = client.get(url)
            response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        suffix = ".wav"
        if "mpeg" in content_type or url.endswith(".mp3"):
            suffix = ".mp3"
        elif "wav" in content_type or url.endswith(".wav"):
            suffix = ".wav"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(response.content)
            logger.info(f"Downloaded: {url} → {tmp.name}")
            return tmp.name
    except Exception as e:
        logger.error(f"Download failed: {url} | {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download audio from URL: {str(e)}")

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
            # model_revision="v2.0.4",
            disable_update=True,
            update_model=False,
            device=DEVICE
        )
        # 使用 pipeline 加载 VAD 模型
        # vad_model = pipeline(
        #     task="voice-activity-detection",
        #     model=VAD_MODEL_PATH,
        #     model_revision="v2.0.4",
        #     device=DEVICE,
        #     disable_update=True,
        #     update_model=False,
        # )
        sd_model = pipeline(
            task='speaker-diarization',
            model=SD_MODEL_PATH,
            # model_revision="v1.0.0",
            # vad_model=VAD_MODEL_PATH,
            # vad_model_revision="v2.0.4",
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
        print(f"DEVICE:{DEVICE}")
        logger.info("✅ 所有模型加载成功！")
    except Exception as e:
        logger.error(f"❌ 模型加载失败: {e}")
        raise
import os
import wave
from pydub import AudioSegment  # 可选，用于更详细的音频分析

def get_audio_duration(path: str) -> float:
    """返回音频时长（秒），支持 WAV、MP3 等常见格式"""
    try:
        # 尝试用 pydub（更通用）
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000.0  # 毫秒转秒
    except Exception as e:
        logger.warning(f"无法用 pydub 读取 {path}: {e}")
        # 回退到 wave（仅限 WAV）
        try:
            with wave.open(path, 'r') as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except:
            return 0.0
# ==============================
# 🔄 通用处理函数（内部使用）
# ==============================
def _process_asr(temp_audio_path: str) -> str:
    result = asr_model(temp_audio_path, hotword=global_hotword_str)
    text = ""
    if isinstance(result, list) and len(result) > 0:
        text = result[0].get("text", "").strip()
    elif isinstance(result, dict):
        text = result.get("text", "").strip()
    if text and punc_model:
        punc_res = punc_model.generate(input=text)
        text = punc_res[0].get("text", text) if punc_res else text
    return text

def _process_transcribe(main_path: str, doctor_path: str) -> dict:
    start_time = time.time()
    # ===== 新增：单独运行 VAD 并打印结果 =====
    logger.info(f"Running VAD on main audio: {main_path}")
    vad_result = vad_model.generate(input=main_path)
    logger.info(f"VAD result: {vad_result}")

    if not vad_result or not vad_result[0].get("value"):
        raise HTTPException(status_code=400, detail="VAD 未检测到任何有效语音段！音频可能全是静音。")

    vad_segments = vad_result[0]["value"]
    total_vad_duration = sum(seg[1] - seg[0] for seg in vad_segments)
    logger.info(f"VAD 检测到 {len(vad_segments)} 段语音，总有效时长: {total_vad_duration:.2f} 秒")

    if total_vad_duration < 0.5:
        raise HTTPException(status_code=400, detail=f"有效语音时长过短 ({total_vad_duration:.2f}s)，请检查录音质量。")
    # =========================================
    full_audio = AudioSegment.from_file(main_path).set_frame_rate(16000).set_channels(1)

    def standardize_audio_for_modelscope(src_path: str) -> str:
        """转换为 ModelScope 兼容的 16kHz, 单声道, 16-bit PCM WAV"""
        audio = AudioSegment.from_file(src_path)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        dst_path = str(Path(src_path).with_suffix("")) + "_modelscope.wav"
        audio.export(dst_path, format="wav")
        os.unlink(src_path)  # 清理原始临时文件
        return dst_path
    main_path = standardize_audio_for_modelscope(main_path)
    # === 3. 调用 sd_model 时传入 segments，跳过内部 VAD！===
    sd_result = sd_model(main_path,oracle_num=2)
    raw_segments = sd_result.get("text", [])
    logger.info(f"SD 分段: {raw_segments}")
    if not raw_segments:
        raise HTTPException(status_code=400, detail="未检测到任何语音片段")

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
            seg_audio.export(f.name, format="wav")
            try:
                emb = get_embedding(f.name)
                spk_to_embedding[spk_key] = emb
            finally:
                os.unlink(f.name)

    doctor_emb = get_embedding(doctor_path)
    doctor_spk = None
    best_sim = -1.0
    for spk_key, emb in spk_to_embedding.items():
        sim = float(np.dot(doctor_emb, emb))
        if sim >= SIMILARITY_THRESHOLD and sim > best_sim:
            best_sim = sim
            doctor_spk = spk_key

    if doctor_spk is None:
        doctor_spk = list(spk_to_embedding.keys())[0] if spk_to_embedding else "spk0"

    final_segments = []
    for start_sec, end_sec, spk_id in raw_segments:
        spk_key = f"spk{spk_id}"
        role = 0 if spk_key == doctor_spk else 1
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        seg_audio = full_audio[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            seg_audio.export(f.name, format="wav")
            try:
                text = _process_asr(f.name)
                if text:
                    speaker_label = "doctor" if role == 0 else "customer"
                    final_segments.append({
                        "time": format_time(start_sec),
                        "endTime": format_time(end_sec),
                        "role": speaker_label,
                        "content": text
                    })
            finally:
                os.unlink(f.name)

    total_time = time.time() - start_time
    return {
        "status": "success",
        "processing_time_seconds": round(total_time, 2),
        "segments": final_segments
    }

# ==============================
# 🌐 API 接口（兼容上传和 URL）
# ==============================

@app.post("/asr")
async def asr_with_hotwords(
    request: Request,
    audio: Optional[UploadFile] = File(None)
):
    temp_path = None
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            audio_url = body.get("audio_url")
            if not audio_url:
                raise HTTPException(status_code=400, detail="JSON body 必须包含 audio_url")
            temp_path = download_audio_from_url(audio_url)
        elif audio is not None:
            temp_path = save_upload_file(audio)
        else:
            raise HTTPException(status_code=400, detail="必须提供 audio 文件或 JSON 中的 audio_url")

        text = _process_asr(temp_path)
        return JSONResponse(content={
            "status": "success",
            "text": text,
            "hotwords_used": global_hotword_str
        })

    except Exception as e:
        logger.error(f"ASR 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

@app.post("/transcribe")
async def transcribe_audio(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    doctor_enroll: Optional[UploadFile] = File(None)
):
    logger.info("=== 进入 transcribe 接口 ===")  # 看这行是否打印
    main_path = doctor_path = None
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            audio_url = body.get("audio_url")
            doctor_url = body.get("doctor_enroll_url")
            if not audio_url or not doctor_url:
                raise HTTPException(status_code=400, detail="JSON body 必须包含 audio_url 和 doctor_enroll_url")
            main_path = download_audio_from_url(audio_url)
            doctor_path = download_audio_from_url(doctor_url)
        elif audio is not None and doctor_enroll is not None:
            main_path = save_upload_file(audio)
            doctor_path = save_upload_file(doctor_enroll)
        else:
            raise HTTPException(status_code=400, detail="必须同时提供 audio + doctor_enroll 文件，或 JSON 中的两个 URL")
        # 在调用 _process_transcribe 之前
        main_duration = get_audio_duration(main_path)
        doctor_duration = get_audio_duration(doctor_path)

        logger.info(f"Main audio duration: {main_duration:.2f}s, Doctor audio duration: {doctor_duration:.2f}s")

        if main_duration < 0.5:
            raise HTTPException(status_code=400,
                                detail=f"主音频有效时长过短 ({main_duration:.2f}s)，请提供至少 0.5 秒的语音")
        if doctor_duration < 0.5:
            raise HTTPException(status_code=400,
                                detail=f"医生注册音频有效时长过短 ({doctor_duration:.2f}s)，请提供至少 0.5 秒的语音")
        result = _process_transcribe(main_path, doctor_path)

        # 保存结果
        output_filename = f"transcript_{uuid.uuid4().hex[:8]}_{int(time.time())}.json"
        output_json_path = OUTPUT_DIR / output_filename
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Transcribe 处理失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for path in [main_path, doctor_path]:
            if path and os.path.exists(path):
                os.unlink(path)

@app.get("/")
def read_root():
    return {
        "message": "支持上传文件和 URL 两种方式！",
        "endpoints": {
            "/asr": [
                "multipart/form-data: audio=文件",
                "application/json: {\"audio_url\": \"http://...\"}"
            ],
            "/transcribe": [
                "multipart/form-data: audio=文件 & doctor_enroll=文件",
                "application/json: {\"audio_url\": \"...\", \"doctor_enroll_url\": \"...\"}"
            ]
        }
    }

# ==============================
# 🏁 启动
# ==============================
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ASR server on http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)