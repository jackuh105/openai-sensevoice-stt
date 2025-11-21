from pathlib import Path
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = Path.cwd() / "models" / "iic" / "SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
    disable_pbar=True,
    disable_log=True,
)

res = model.generate(
    input="./audio/asr_example_zh.wav", # need path string not PosixPath
    language="auto",
    use_itn=True,
    merge_vad=True,
    merge_length_s=15,
)

print(rich_transcription_postprocess(res[0]["text"]))
