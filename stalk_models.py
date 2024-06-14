import intel_npu_acceleration_library

import gc

import torch
import torchaudio

from llama_cpp import Llama

from pyannote.audio import Audio
from pyannote.core import Segment

from faster_whisper import WhisperModel

from huggingface_hub import hf_hub_download
import wespeaker

DEVICE = "cuda"
if not torch.cuda.is_available():
    DEVICE = "cpu"
    print("INFO: CUDA is diabled on this machine.\n")

print("PyTorch:", torch.__version__)
print("TorchAudio:", torchaudio.__version__)
print("Uses Device:", DEVICE.upper())


class ChatHistory(list):
    messages = []

    @classmethod
    def add_messages(cls, role, content):
        if isinstance(content, str):
            cls.messages.append({ 'role': role, 'content': content })
        else:
            for r, c in zip(role, content):
                cls.messages.append({ 'role': r, 'content': c })

    @classmethod
    def create_prompt(cls, system_prompt: str, user_prompt: str = ""):
        return [
            {
                "role": "system",
                "content": system_prompt
            },
            *cls.messages,
            {
                "role": "user",
                "content": user_prompt
            }
        ]


def token_stream(token):
    delta = token["choices"][0]["delta"]
    if "content" not in delta:
        return ""
    else:
        return delta["content"]


def get_llama3():
    model_id = "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"

    chat = Llama.from_pretrained(
        repo_id=model_id,
        filename="*Q4_K_M.gguf",
        #chat_format="llama-3",
        verbose=False
    ).create_chat_completion

    def llama3(system_prompt, user_prompt, temp=0.5, show_prompt=False):
        prompt = ChatHistory.create_prompt(system_prompt, user_prompt)

        if show_prompt:
            print("PROMPT:")
            for line in prompt:
                print(line)
            print()

        return chat(prompt, temperature=temp, stream=True)

    return llama3


def get_whisper():
    model_size = "medium"  #@param ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    compute_type = "int8"  #@param ['float16', 'int8']

    return WhisperModel(model_size, device=DEVICE, cpu_threads=12, compute_type=compute_type).transcribe


def extract_embedding(model, pcm, sample_rate):
    pcm = pcm.to(torch.float)
    if sample_rate != model.resample_rate:
        pcm = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=model.resample_rate)(pcm)
    feats = model.compute_fbank(
        pcm,
        sample_rate=model.resample_rate,
        cmn=True
    )
    feats = feats.unsqueeze(0)
    feats = feats.to(model.device)
    model.model.eval()
    with torch.no_grad():
        outputs = model.model(feats)
        outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
    embedding = outputs[0].to(torch.device('cpu'))
    return embedding


def recognize(model, pcm, sample_rate):
    q = extract_embedding(model, pcm, sample_rate)
    best_score = 0.0
    best_name = ''
    for name, e in model.table.items():
        score = model.cosine_similarity(q, e)
        if best_score < score:
            best_score = score
            best_name = name
        del score
        gc.collect()
    return {'name': best_name, 'confidence': best_score}


def get_resnet152():
    model_id = "Wespeaker/wespeaker-voxceleb-resnet152-LM"
    model_name = model_id.replace("Wespeaker/wespeaker-", "").replace("-", "_")

    root_dir = hf_hub_download(model_id, filename=model_name+".onnx").replace(model_name+".onnx", "")

    import os
    if not os.path.isfile(root_dir+"avg_model.pt"):
        os.rename(hf_hub_download(model_id, filename=model_name+".pt"), root_dir+"avg_model.pt")
    if not os.path.isfile(root_dir+"config.yaml"):
        os.rename(hf_hub_download(model_id, filename=model_name+".yaml"), root_dir+"config.yaml")

    resnet = wespeaker.load_model_local(root_dir)

    #print("Compile model for the NPU")
    #resnet.model = intel_npu_acceleration_library.compile(resnet.model)

    def resnet152(ado, sample_rate=None):
        if isinstance(ado, str):
            return resnet.recognize(ado)
        else:
            return recognize(resnet, ado, sample_rate)

    resnet152.__dict__['register'] = lambda *args, **kwargs: resnet.register(*args, **kwargs)

    return resnet152


llama3 = get_llama3()
print("INFO: Llama3 Ready -", llama3)


whisper = get_whisper()
print("INFO: Whisper Ready -", whisper)


audio = Audio()
resnet152 = get_resnet152()
print("INFO: ResNet152 Ready -", resnet152)
