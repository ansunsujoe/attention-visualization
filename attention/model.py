from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model_from_hub(model_name: str) -> None:
    snapshot_download(
        repo_id=model_name,
        local_dir=f"./models/{model_name.split('/')[1].lower()}",
        local_dir_use_symlinks=False,
    )


def load_model_from_file(
    model_path: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    return model, tokenizer
