# Experiment Notes

이 문서는 저장소에 포함한 스크립트와 노트북의 역할을 정리합니다.

## Public Artifacts

| Path | Role |
| --- | --- |
| `scripts/submit_w8a8_only.py` | W8A8 양자화 제출용 최소 실행 스크립트 |
| `notebooks/phase2_true_smoothquant_w8a8.ipynb` | baseline W8A8과 SmoothQuant 변형을 같은 경로에서 비교하기 위한 harness |
| `notebooks/colab_mixed_precision_quantization.ipynb` | SmoothQuant W8A8과 GPTQ W4A8 혼합 정밀도 탐색 노트북 |
| `notebooks/colab_awq_setup_final.ipynb` | llmcompressor 기반 AWQ-style 양자화 실험 노트북 |
| `notebooks/colab_smoothquant_w8a8.ipynb` | SmoothQuant + W8A8 기본 실험 노트북 |
| `notebooks/colab_smoothquant_w8a8_ignore_embed_only.ipynb` | embedding/lm head 제외 조건을 명시한 W8A8 실험 노트북 |

## Why W8A8

W8A8은 weight와 activation을 모두 8-bit로 다루는 방향입니다. LLM 경량화에서 중요한 이유는 모델 크기뿐 아니라 실제 추론 시의 activation memory와 kernel 호환성까지 같이 봐야 하기 때문입니다.

이 저장소의 대표 제출 스크립트는 다음 결정을 명확히 남깁니다.

- model: `LGAI-EXAONE/EXAONE-4.0-1.2B`
- calibration dataset: `LGAI-EXAONE/MANTA-1M`
- calibration samples: `256`
- max sequence length: `512`
- quantization scheme: `W8A8`
- targets: `Linear`
- ignored modules: `embed_tokens`, `lm_head`

## Lessons

### 1. Calibration is part of the experiment

양자화 실험에서 calibration sample 수와 sequence length는 단순한 실행 옵션이 아니라 결과 해석 조건입니다. 같은 모델이라도 calibration 설정이 바뀌면 activation range와 outlier 처리 결과가 달라질 수 있습니다.

### 2. Ignore list matters

`embed_tokens`와 `lm_head`를 제외한 이유는 token embedding과 output projection이 품질 저하에 민감할 수 있기 때문입니다. 모든 모듈을 기계적으로 양자화하기보다, 품질에 민감한 경계를 남기는 쪽이 실험적으로 안전합니다.

### 3. Packaging is also reproducibility

대회 제출에서는 모델 저장 구조와 압축 결과가 평가 파이프라인에 직접 영향을 줍니다. 따라서 `save_pretrained(..., save_compressed=True)`와 `submit.zip` 생성 과정을 스크립트 안에 포함했습니다.

## Sanitization

공개 전 노트북은 다음 처리를 거쳤습니다.

- cell outputs 제거
- execution count 제거
- widget metadata 제거
- 실제 토큰, 비밀번호, API key 값이 없는지 검색

공개 권한이 불명확한 원본 데이터와 제출 산출물 zip은 저장소에 포함하지 않았습니다.
