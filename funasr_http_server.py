import argparse
import logging
import os
import uuid
from pathlib import Path

import aiofiles
import ffmpeg
import uvicorn
from fastapi import FastAPI, File, UploadFile
from modelscope.utils.logger import get_logger

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = get_logger(log_level=logging.INFO)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=8000, required=False, help="server port")
parser.add_argument(
    "--model_dir",
    type=str,
    default="models/iic/SenseVoiceSmall",
    help="SenseVoice model directory path",
)
parser.add_argument(
    "--remote_code",
    type=str,
    default="./model.py",
    help="Path to remote code file for SenseVoice model",
)
parser.add_argument(
    "--vad_model",
    type=str,
    default="fsmn-vad",
    help="VAD model name",
)
parser.add_argument(
    "--vad_kwargs",
    type=int,
    default=30000,
    help="max_single_segment_time for VAD in milliseconds",
)
parser.add_argument("--device", type=str, default="cpu", help="cuda or cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--language",
    type=str,
    default="auto",
    help="Language for ASR (auto, zh, en, yue, ja, ko)",
)
parser.add_argument(
    "--use_itn",
    type=bool,
    default=True,
    help="Use inverse text normalization",
)
parser.add_argument(
    "--merge_vad",
    type=bool,
    default=True,
    help="Merge VAD segments",
)
parser.add_argument(
    "--merge_length_s",
    type=int,
    default=15,
    help="Maximum length in seconds for merging VAD segments",
)
parser.add_argument("--certfile", type=str, default=None, required=False, help="certfile for ssl")
parser.add_argument("--keyfile", type=str, default=None, required=False, help="keyfile for ssl")
parser.add_argument("--temp_dir", type=str, default="temp_dir/", required=False, help="temp dir")
args = parser.parse_args()
logger.info("-----------  Configuration Arguments -----------")
for arg, value in vars(args).items():
    logger.info("%s: %s" % (arg, value))
logger.info("------------------------------------------------")

os.makedirs(args.temp_dir, exist_ok=True)

logger.info("model loading")
# load SenseVoice model
model_dir = Path.cwd() / args.model_dir
model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code=args.remote_code,
    vad_model=args.vad_model,
    vad_kwargs={"max_single_segment_time": args.vad_kwargs},
    device=args.device,
    ncpu=args.ncpu,
    disable_pbar=True,
    disable_log=True,
)
logger.info("loaded models!")

app = FastAPI(title="FunASR")

param_dict = {
    "language": args.language,
    "use_itn": args.use_itn,
    "merge_vad": args.merge_vad,
    "merge_length_s": args.merge_length_s,
}


@app.post("/recognition")
async def api_recognition(audio: UploadFile = File(..., description="audio file")):
    suffix = audio.filename.split(".")[-1]
    audio_path = f"{args.temp_dir}/{str(uuid.uuid1())}.{suffix}"
    async with aiofiles.open(audio_path, "wb") as out_file:
        content = await audio.read()
        await out_file.write(content)
    try:
        audio_bytes, _ = (
            ffmpeg.input(audio_path, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=16000)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        logger.error(f"读取音频文件发生错误，错误信息：{e}")
        return {"msg": "读取音频文件发生错误", "code": 1}
    rec_results = model.generate(input=audio_bytes, is_final=True, **param_dict)
    
    # 檢查結果
    if not rec_results or len(rec_results) == 0:
        return {"text": "", "code": 0}
    
    rec_result = rec_results[0]
    if "text" not in rec_result or len(rec_result["text"]) == 0:
        return {"text": "", "code": 0}
    
    # 使用 rich_transcription_postprocess 處理結果
    text = rich_transcription_postprocess(rec_result["text"])
    
    # 簡化的回應格式
    ret = {"text": text, "code": 0}
    logger.info(f"識別結果：{ret}")
    return ret


if __name__ == "__main__":
    uvicorn.run(
        app, host=args.host, port=args.port, ssl_keyfile=args.keyfile, ssl_certfile=args.certfile
    )