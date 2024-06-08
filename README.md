# 2024-1_BPL_STalk_Model_Research | 기초프로젝트랩
 2024-1 기초프로젝트랩 8조 - STalk - AI Model 리서치 자료


[whisper][wespeaker][llama3][speaker verification][speaker diarization]

This model performs voice recognition and speaker separation by streaming the conversational voice in real time during an encounter with the other person, converts the voice into text, and inputs the conversation content for each speaker into llama3. This provides a function that recommends the next topic when the atmosphere becomes awkward based on analysis of conversation content.


![structure](https://github.com/b-re-w/2024-1_BPL_STalk_Model_Research/blob/main/res/stalk_structure.jpg?raw=true)


---
## Approach
### 1. Wespeaker

WeSpeaker is a research and production-oriented toolkit designed for learning speaker embeddings, which are fixed-dimensional vector representations used in tasks such as speaker recognition and diarization. This toolkit implements state-of-the-art speaker embedding models, loss functions, and scoring backends to deliver competitive results.

#### Ovarall Structure
A standard procedure contains data preparation on the disk, online feature preparation, and model training. After the converged model is obtained, it can be easily exported to a run-time format and ready for further deployment. The extracted speaker embeddings can then be applied to downstream tasks, such as speaker verification and diarization.

1. Competitive Results: Achieves highly competitive performance on datasets like VoxCeleb, CNCeleb, and VoxConverse, replicating tricks from winning systems in speaker verification challenges to enhance system performance.
2. Lightweight: Built on PyTorch and its ecosystem without dependencies on external tools like Kaldi.
3. Unified IO (UIO): Introduces the unified IO system from Wenet, providing a seamless interface to support hundreds of millions of hours of data.
4. Real-time Feature Preparation: Performs data augmentation and feature extraction in real-time, reducing disk costs and improving model robustness.
5. Distributed Learning: Supports multi-node, multi-GPU scaling using Pytorch’s DistributedDataParallel.
6. Production Ready: Easily exportable to formats like torch JIT or ONNX for deployment in production environments.

[WESPEAKER: A RESEARCH AND PRODUCTION ORIENTED SPEAKER EMBEDDING LEARNING TOOLKIT](https://arxiv.org/pdf/2210.17016)

![structure](https://github.com/b-re-w/2024-1_BPL_STalk_Model_Research/blob/main/res/wespeaker_structure.jpg?raw=true)

### 2. Whisper

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

### 3. Llama3

LLAMA3 models by Meta in April 2024, which have been pre-trained on over 15 trillion tokens, demonstrate impressive performance. The research investigates the effectiveness of low-bit quantization for these models, a technique critical for deploying LLMs in resource-limited environments.

LLAMA3 models suffer from notable performance degradation when quantized to ultra-low bit widths. This underlines a significant gap in the current capabilities of low-bit quantization methods and the actual performance needs.

[How Good Are Low-bit Quantized LLAMA3 Models? An Empirical Study](https://arxiv.org/pdf/2404.14047)


---
## Installation

