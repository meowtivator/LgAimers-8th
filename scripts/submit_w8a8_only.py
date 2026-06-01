# -*- coding: utf-8 -*-
"""
[Dev Environment]
- OS: Linux-6.6.113+-x86_64-with-glibc2.35 (Colab)
- Python: 3.12.12
- GPU: Tesla T4
- NVIDIA Driver: 580.82.07
- CUDA: nvidia-smi=13.0 / nvcc=12.8 / torch.cuda=12.8
- cuDNN: 91002

[Key Libraries]
- torch==2.10.0+cu128
- transformers==4.57.6
- datasets==4.0.0
- llmcompressor==0.10.0
- vllm: N/A (not installed)

[Data]
- External data: 없음 (대회 제공: LGAI-EXAONE/MANTA-1M만 사용)

[Reproduction]
- Run: python submit_w8a8_only.py
- Output: ./model + submit.zip
"""


import os
import shutil

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


MODEL_ID = "LGAI-EXAONE/EXAONE-4.0-1.2B"
OUT_DIR = "./model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"

NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# W8A8 Quantization
SCHEME = "W8A8"
TARGETS = ["Linear"]
IGNORE = ["embed_tokens", "lm_head"]


def main() -> None:
    print("[INFO] 모델 로드 중...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    print("[INFO] 모델/토크나이저 로드 완료")

    print("[INFO] 캘리브레이션 데이터 로드 중...")
    ds = load_dataset(
        DATASET_ID,
        split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]",
    )

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["conversations"],
                add_generation_prompt=True,
                tokenize=False,
            )
        }

    ds = ds.map(preprocess)
    print("[INFO] 데이터 전처리 완료")

    print(
        f"[INFO] W8A8 양자화 시작: scheme={SCHEME}, targets={TARGETS}, ignore={IGNORE}, "
        f"samples={NUM_CALIBRATION_SAMPLES}, max_len={MAX_SEQUENCE_LENGTH}"
    )

    # 아래 recipe/oneshot 블록은 기존에 동작했던 W8A8 방식의 핵심을 유지합니다.
    recipe = [
        QuantizationModifier(
            scheme=SCHEME,
            targets=TARGETS,
            ignore=IGNORE,
        ),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    print("[INFO] W8A8 양자화 완료")

    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR, save_compressed=True)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[INFO] 모델 저장 완료: {OUT_DIR}")

    zip_name = "submit"
    print(f"[INFO] {zip_name}.zip 생성 중...")
    shutil.make_archive(
        base_name=zip_name,
        format="zip",
        root_dir=".",
        base_dir=OUT_DIR,
    )
    print(f"[INFO] 생성 완료: {zip_name}.zip")


if __name__ == "__main__":
    main()
