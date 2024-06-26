# Awesome Korean LLM [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
##### https://github.com/NomaDamas/awesome-korean-llm?utm_source=pytorchkr

한국어 오픈소스 LLM 정보를 모아놓은 awesome list입니다.

## 한국어 오픈소스 LLM

|                                      이름                                      |               사이즈               |                        제작자                         |       베이스 모델        | 상업적 사용 여부 |                                       가중치                                       |
|:----------------------------------------------------------------------------:|:-------------------------------:|:--------------------------------------------------:|:-------------------:|:---------:|:-------------------------------------------------------------------------------:|
|            [Polyglot-Ko](https://github.com/EleutherAI/polyglot)             |     1.3B, 3.8B, 5.8B, 12.8B     |    [EleutherAI](https://github.com/EleutherAI)     |      GPT-NeoX       |     ✅     | [🤗 Huggingface](https://huggingface.co/EleutherAI/polyglot-ko-12.8b/tree/main) |
|                [KoAlpaca](https://github.com/Beomi/KoAlpaca)                 | 7B, 13B, 30B, 65B / 5.8B, 12.8B |         [beomi](https://github.com/Beomi)          | llama / Polyglot-Ko |  ❌  /  ✅  |     [🤗 Huggingface](https://huggingface.co/beomi/KoAlpaca-Polyglot-12.8B)      |
|                 [KuLLM](https://github.com/nlpai-lab/KULLM)                  |           5.8B, 12.8B           |       [고려대학교](https://github.com/nlpai-lab)        |     Polyglot-Ko     |     ✅     |   [🤗 Huggingface](https://huggingface.co/nlpai-lab/kullm-polyglot-12.8b-v2)    |
|                [KORani](https://github.com/krafton-ai/KORani)                |           13B / 12.8B           |      [KRAFTON](https://github.com/krafton-ai)      | llama / Polyglot-Ko |  ❌  /  ✅  |         [🤗 Huggingface](https://huggingface.co/KRAFTON/KORani-v3-13B)          |
|            [K(G)OAT](https://github.com/Marker-Inc-Korea/K-G-OAT)            |              5.8B               | [Marker Inc.](https://github.com/Marker-Inc-Korea) |     Polyglot-ko     |     ✅     |     [🤗 Huggingface](https://huggingface.co/DopeorNope/KOAT-5.8b/tree/main)     |
|            [KoVicuna](https://github.com/melodysdreamj/KoVicuna)             |               7B                | [melodysdreamj](https://github.com/melodysdreamj)  |      Vicuna 7B      |     ❓     |          [🤗 Huggingface](https://huggingface.co/junelee/ko_vicuna_7b)          |
|             [Kollama](https://huggingface.co/beomi/kollama-33b)              |          7B, 13B, 33B           |         [beomi](https://github.com/Beomi)          |        llama        |     ❌     |           [🤗 Huggingface](https://huggingface.co/beomi/kollama-33b)            |
|           [Llama-2-Ko](https://huggingface.co/beomi/llama-2-ko-7b)           |             7B, 70B             |         [beomi](https://github.com/Beomi)          |       llama-2       |     ✅     |          [🤗 Huggingface](https://huggingface.co/beomi/llama-2-ko-7b)           ||    [KoLlama2](https://github.com/psymon-dev/KoLlama2)    |            ㅇ            |    [psymon-dev](https://github.com/psymon-dev)     |      llama-2       |     ❓     |           [🤗 Huggingface](https://huggingface.co/psymon/KoLlama2-7b)           |
|                 [komt](https://github.com/davidkim205/komt)                  |             7B, 13B             |   [davidkim205](https://github.com/davidkim205)    |       llama-2       |     ✅     |    [🤗 Huggingface](https://huggingface.co/davidkim205/komt-Llama-2-13b-hf)     |
| [llama-2-korean](https://huggingface.co/quantumaikr/llama-2-70b-fb16-korean) |          7B, 13B, 70B           |     [(주)퀀텀아이](https://github.com/quantumaikr)      |       llama-2       |     ✅     |  [🤗 Huggingface](https://huggingface.co/quantumaikr/llama-2-70b-fb16-korean)   |
|             [KoreanLM](https://github.com/quantumaikr/KoreanLM)              |            1.5B, 3B             |     [(주)퀀텀아이](https://github.com/quantumaikr)      |          ❓          |     ✅     |          [🤗 Huggingface](https://huggingface.co/quantumaikr/KoreanLM)          |
|              [KoRMKV](https://huggingface.co/beomi/KoRWKV-1.5B)              |            1.5B, 6B             |         [beomi](https://github.com/Beomi)          |       RMKVv4        |     ❓     |           [🤗 Huggingface](https://huggingface.co/beomi/KoRWKV-1.5B)            |
|      [KoAlpaca-KoRMKV](https://huggingface.co/beomi/KoAlpaca-KoRWKV-6B)      |            1.5B, 6B             |         [beomi](https://github.com/Beomi)          |       KoRMKV        |     ❓     |           [🤗 Huggingface](https://huggingface.co/beomi/KoRWKV-1.5B)            |
|                 [KoGPT](https://github.com/kakaobrain/kogpt)                 |               6B                |      [카카오브레인](https://github.com/kakaobrain)       |          ❓          |     ❌     | [🤗 Huggingface](https://huggingface.co/kakaobrain/kogpt/tree/KoGPT6B-ryan1.5b) |
|                  [KoGPT2](https://github.com/SKT-AI/KoGPT2)                  |              1.5B               |          [SKT](https://github.com/SKT-AI)          |        GPT-2        |     ❌     |                                        ❓                                        |
|    [Llama-2-ko-7b-Chat](https://huggingface.co/kfkas/Llama-2-ko-7b-Chat)     |               7B                |    [taemin6697](https://github.com/taemin6697)     |       llama-2       |     ✅     |        [🤗 Huggingface](https://huggingface.co/kfkas/Llama-2-ko-7b-Chat)        |
|    [42dot_LLM-PLM-1.3B](https://huggingface.co/42dot/42dot_LLM-PLM-1.3B)     |              1.3B               |         [42dot](https://github.com/42dot)          |       llama-2       |     ❌     |        [🤗 Huggingface](https://huggingface.co/42dot/42dot_LLM-PLM-1.3B)        |
|    [42dot_LLM-SFT-1.3B](https://huggingface.co/42dot/42dot_LLM-SFT-1.3B)     |              1.3B               |         [42dot](https://github.com/42dot)          |       llama-2       |     ❌     |        [🤗 Huggingface](https://huggingface.co/42dot/42dot_LLM-SFT-1.3B)        |
|        [Ko-Platypus](https://github.com/Marker-Inc-Korea/KO-Platypus)        |               7B                | [Marker Inc.](https://github.com/Marker-Inc-Korea) |       llama-2       |     ❓     |      [🤗 Huggingface](https://huggingface.co/kyujinpy/KO-Platypus2-7B-ex)       |
|           [sitebunny](https://huggingface.co/42MARU/sitebunny-13b)           |               13B               |      [42MARU](https://huggingface.co/42MARU)       |       llama-2       |     ✅     |          [🤗 Huggingface](https://huggingface.co/42MARU/sitebunny-13b)          |
|           [ChatSKKU](https://huggingface.co/jojo0217/ChatSKKU5.8B)           |              5.8B               |      [jojo0217](https://github.com/JoJo0217)       |     Polyglot-Ko     |     ❓     |         [🤗 Huggingface](https://huggingface.co/jojo0217/ChatSKKU5.8B)          |
|             [nallm](https://github.com/Nara-Information/NA-LLM)              |           1.3B, 3.8B            |   [나라지식정보](https://github.com/Nara-Information)    |     Polyglot-Ko     |     ❓     |  [🤗 Huggingface](https://huggingface.co/Nara-Lab/nallm-polyglot-ko-3.8b-base)  |
|     [Mi:dm (믿:음)](https://huggingface.co/KT-AI/midm-bitext-S-7B-inst-v1)    |               7B                |          [KT](https://huggingface.co/KT-AI)         |          ❓          |     ❌     |  [🤗 Huggingface](https://huggingface.co/KT-AI/midm-bitext-S-7B-inst-v1)  |

## 한국어 오픈소스 LLM 모델

### 1. Polyglot-Ko 기반

- [Polyglot-Ko](https://github.com/EleutherAI/polyglot)
    - [KoAlpaca](https://github.com/Beomi/KoAlpaca)
    - [KuLLM](https://github.com/nlpai-lab/KULLM)
    - [KORani](https://github.com/krafton-ai/KORani)
    - [K(G)OAT](https://github.com/Marker-Inc-Korea/K-G-OAT)
    - [ChatSKKU](https://huggingface.co/jojo0217/ChatSKKU5.8B)
    - [nallm](https://github.com/Nara-Information/NA-LLM)

### 2-1. Llama 기반

- [Kollama](https://huggingface.co/beomi/kollama-33b)
- [KoAlpaca](https://github.com/Beomi/KoAlpaca)
- [KORani](https://github.com/krafton-ai/KORani)

### 2-2. Llama-2 기반

- [Llama-2-Ko](https://huggingface.co/beomi/llama-2-ko-7b)
- [komt](https://github.com/davidkim205/komt)
- [llama-2-korean](https://huggingface.co/quantumaikr/llama-2-70b-fb16-korean)
- [Llama-2-ko-7b-Chat](https://huggingface.co/kfkas/Llama-2-ko-7b-Chat)
- [42dot_LLM-PLM-1.3B](https://huggingface.co/42dot/42dot_LLM-PLM-1.3B)
- [42dot_LLM-SFT-1.3B](https://huggingface.co/42dot/42dot_LLM-SFT-1.3B)
- [Ko-Platypus](https://github.com/Marker-Inc-Korea/KO-Platypus)
- [sitebunny](https://huggingface.co/42MARU/sitebunny-13b)

### 3. 기타

- [KoVicuna](https://github.com/melodysdreamj/KoVicuna)
- [KoRMKV](https://huggingface.co/beomi/KoRWKV-1.5B)
- [KoAlpaca-KoRMKV](https://huggingface.co/beomi/KoAlpaca-KoRWKV-6B)
- [KoGPT](https://github.com/kakaobrain/kogpt)
- [KoGPT2](https://github.com/SKT-AI/KoGPT2)
- [KoreanLM](https://github.com/quantumaikr/KoreanLM)
- [Mi:dm (믿:음)](https://huggingface.co/KT-AI/midm-bitext-S-7B-inst-v1)

## Contributing

틀린 점이 있거나 새로운 한국어 LLM이 있다면 언제나 PR을 보내주세요!
