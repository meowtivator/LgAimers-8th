# LG Aimers 8th - LLM Lightweighting Notes

LG Aimers 8기 과정에서 학습한 LLM 구조, 평가, 경량화 관점을 정리하는 저장소입니다.  
포트폴리오에서는 이 저장소를 **대형 언어모델의 추론 비용, 메모리, 품질 사이의 균형을 설명하기 위한 AI 엔지니어링 근거**로 사용합니다.

> 현재 공개 범위는 학습 노트와 실험 설계 중심입니다.  
> 민감한 원본 데이터, 비공개 과제 자료, 공개 권한이 불명확한 산출물은 포함하지 않습니다.

## Overview

LLM 경량화는 단순히 모델 파일 크기를 줄이는 일이 아닙니다. 실제 서비스 적용 가능성을 보려면 다음 항목을 함께 비교해야 합니다.

- 추론 지연시간
- VRAM 및 메모리 사용량
- 정확도와 생성 품질 저하
- 재현 가능한 평가 기준
- 하드웨어 제약과 배포 비용
- 양자화 이후 실패 케이스

이 저장소는 위 항목을 기준으로 `baseline -> lightweighting -> evaluation -> deployment trade-off` 흐름을 정리합니다.

## Focus

| Area | What I focused on |
| --- | --- |
| LLM fundamentals | Transformer, attention, pretraining, decoding, RAG, agent 구조 |
| Lightweighting | INT8/W8A8, quantization 관점, distillation, cost optimization |
| Evaluation | 성능 숫자뿐 아니라 재현 조건, 한계, 실패 지점 기록 |
| Hardware | GPU, VRAM, precision, serving cost 관점 |
| Product judgment | 모델 경량화 결과를 실제 제품 의사결정 언어로 번역 |

## Study Map

LG Aimers 과정과 연결해 정리한 개인 학습 노트입니다.

| Topic | Note |
| --- | --- |
| AI/ML basics | [AI의 첫걸음, 머신러닝과 딥러닝의 기초](https://meowti.kr/writing/lg-aimers-ai-ml-deeplearning-basics/) |
| NLP/RNN | [자연어처리의 기초와 RNN](https://meowti.kr/writing/lg-aimers-nlp-rnn-basics/) |
| Transformer | [트랜스포머와 어텐션 메커니즘](https://meowti.kr/writing/lg-aimers-transformer-attention/) |
| LLM pretraining | [거대 언어 모델의 사전 학습과 진화](https://meowti.kr/writing/lg-aimers-llm-pretraining-evolution/) |
| Decoding | [Decoding of Large Language Models](https://meowti.kr/writing/lg-aimers-decoding-large-language-models/) |
| Evaluation | [Evaluation of Large Language Models](https://meowti.kr/writing/lg-aimers-evaluation-large-language-models/) |
| RAG | [Retrieval Augmented Generation](https://meowti.kr/writing/lg-aimers-retrieval-augmented-generation/) |
| Agent | [스스로 계획하고 실행하는 AI, LLM 에이전트](https://meowti.kr/writing/lg-aimers-llm-agent-planning-tools/) |
| Hardware | [LLM을 움직이는 힘, AI 하드웨어와 GPU](https://meowti.kr/writing/lg-aimers-llm-hardware-gpu-vram/) |
| Distillation | [Distillation from Large-Scale Intelligence to Lightweight Deployment](https://meowti.kr/writing/lg-aimers-distillation-overview/) |
| Cost optimization | [LLM 비용 절감과 효율화를 위한 비즈니스 관점](https://meowti.kr/writing/lg-aimers-lightweight-llm-cost-optimization/) |
| Research trend | [미국 경량 LLM 연구 동향](https://meowti.kr/writing/lg-aimers-lightweight-llm-research-trends/) |
| AI ethics | [AI 시대의 Ethics](https://meowti.kr/writing/lg-aimers-ai-ethics/) |

## Experiment Design Checklist

경량화 실험을 정리할 때 다음 기준을 사용합니다.

### 1. Baseline

- 원본 모델명과 파라미터 규모
- tokenizer, max sequence length, batch size
- benchmark 또는 자체 평가 데이터셋
- 추론 환경: GPU, VRAM, CUDA, PyTorch 버전
- baseline latency, memory, quality score

### 2. Lightweighting Method

- quantization type: INT8, W8A8, weight-only 등
- calibration dataset 구성
- activation outlier 처리 여부
- SmoothQuant, GPTQ, AWQ, distillation 등 적용 가능성
- 변환 전후 모델 크기와 로딩 방식

### 3. Evaluation

- latency: average, p50, p95
- memory: peak VRAM, steady memory
- quality: task accuracy, exact match, generated answer quality
- failure case: 긴 문맥, 수치 추론, hallucination, formatting error
- reproducibility: seed, environment, command, commit hash

### 4. Product Decision

경량화 결과는 단순 수치가 아니라 제품 판단으로 연결되어야 합니다.

- 품질 저하를 감수할 수 있는 화면인가
- 응답 지연을 줄이는 편이 더 중요한가
- 서버 비용 절감이 사용자 경험 손실보다 큰가
- fallback 모델이나 재시도 전략이 필요한가
- 모델 교체 시 운영 복잡도가 얼마나 늘어나는가

## Planned Repository Structure

```text
.
├── README.md
├── notes/
│   ├── 00-llm-basics.md
│   ├── 01-quantization.md
│   ├── 02-distillation.md
│   └── 03-evaluation.md
├── experiments/
│   ├── baseline.md
│   ├── int8.md
│   └── w8a8.md
└── reports/
    └── lightweighting-retrospective.md
```

## Portfolio Context

- Portfolio case study: [LG Aimers 모델 경량화](https://meowti.kr/projects/lg-aimers/)
- Role: 실험 설계 / 분석
- Keywords: PyTorch, Quantization, EXAONE, Benchmark
- Period: 2025.12 - 2026.02

## What This Repository Should Prove

이 저장소의 목적은 "모델을 가볍게 만들었다"는 한 줄이 아니라, 아래 질문에 답할 수 있음을 보여주는 것입니다.

- 어떤 기준선을 잡았는가
- 어떤 경량화 방법을 비교할 수 있는가
- 성능, 메모리, 비용, 품질을 어떻게 함께 볼 것인가
- 실패 케이스를 어떻게 기록할 것인가
- 실험 결과를 제품 의사결정으로 어떻게 번역할 것인가

## Disclaimer

이 저장소는 개인 학습 및 회고용입니다. LG Aimers 공식 자료를 대체하지 않으며, 비공개 과제 데이터나 공개 권한이 없는 산출물은 포함하지 않습니다.
