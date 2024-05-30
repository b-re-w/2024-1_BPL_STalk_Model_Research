import torch
from transformers import(
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers import BitsAndBytesConfig
import fire


def inference(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        batch_size: int = 16,
        **kwargs,
) -> list[str]:
    model.eval()
    generated_texts = []
    with torch.no_grad():
        from tqdm import tqdm
        for i in tqdm(range(0, len(prompts), batch_size)):
            batch = prompts[i:i+batch_size]
            generated_texts += process_batch(model,tokenizer,batch,**kwargs)

    generated_texts = [gen_text.replace(prompt,"").replace("$}}%","") \
                       for gen_text,prompt in zip(generated_texts,prompts)]

    return generated_texts


def process_batch(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch: list[str],
        **kwargs,
) -> list[str]:
    model_inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    try:
        model_outputs = model.generate(**model_inputs,**kwargs)
        generated_texts = tokenizer.batch_decode(model_outputs, skip_special_tokens=True)
    except KeyboardInterrupt as ke:
        print(ke)
        exit()
    except RuntimeError as re:
        print(re)
        if "CUDA" in str(re):
            import gc
            del model_inputs
            gc.collect()
            torch.cuda.empty_cache()

            temp_batch_size = len(batch)//2
            print("temp_batch_size:",temp_batch_size)

            temp_batch_1 = batch[:temp_batch_size]
            generated_text_1 = process_batch(model,tokenizer,temp_batch_1,**kwargs)

            temp_batch_2 = batch[temp_batch_size:]
            generated_text_2 = process_batch(model,tokenizer,temp_batch_2,**kwargs)

            generated_texts = generated_text_1 + generated_text_2

    return generated_texts


def main(
        model_size: str = "13",
        batch_size: int = 16,
):
    model_name = f"meta-llama/Llama-2-{model_size}b-chat-hf"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=nf4_config,
        cache_dir="/data/huggingface_models/",
    )
    model.config.pad_token_id = model.config.eos_token_id

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        padding_side="left",
        cache_dir="/data/huggingface_models/"
    )
    tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        "What is the first letter of the alphabet?\n",
        "The capital of Korea:\n",
        "Explain what is summarization:\n"
    ]

    generated_text = inference(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=batch_size,
        max_new_tokens=200,
        do_sample=False,
    )

    for text in generated_text:
        print("="*60)
        print(text)


if __name__ == "__main__":
    fire.Fire(main)



