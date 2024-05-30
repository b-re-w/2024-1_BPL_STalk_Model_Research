import os
os.environ['KMP_DUPLICATE_LIB_OK'] = "True"


from faster_whisper import WhisperModel

language = None
device = "cuda"
model_size = "medium"  #@param ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
compute_type = "int8"  #@param ['float16', 'int8']

model = WhisperModel(model_size, device=device, compute_type=compute_type)


audio_path = "./sample_conversation/real/conversation_0530.wav"
speaker1 = "./sample_conversation/real/sentence_F.wav"
speaker2 = "./sample_conversation/real/sentence_M.wav"


from huggingface_hub import hf_hub_download
import wespeaker

model_id = "Wespeaker/wespeaker-voxceleb-resnet293-LM"

model_binary = model_id.replace("Wespeaker/wespeaker-", "").replace("-", "_")+".onnx"
root_dir = hf_hub_download(model_id, filename=model_binary).replace(model_binary, "")
hf_hub_download(model_id, filename="avg_model.pt")
hf_hub_download(model_id, filename="config.yaml")
resnet = wespeaker.load_model_local(root_dir)


resnet.register("민서", speaker1)
resnet.register("연우", speaker2)


from livepipe import AudioProcessPipeline

pipeline = AudioProcessPipeline(model, resnet, language=language, save_output_recording=True)


if __name__ == "__main__":
    pipeline()
