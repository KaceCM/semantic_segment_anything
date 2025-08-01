

import os
from dotenv import load_dotenv, find_dotenv
from assets.utils import printr
from huggingface_hub import snapshot_download


env_loaded = load_dotenv(find_dotenv(".env"))
required_vars = ["HUGGINGFACE_MODEL_URL", "HUGGINGFACE_TOKEN", "MODEL_PATH"]

if not env_loaded or not all(var in os.environ for var in required_vars):
    raise FileNotFoundError("""Environment file '.env' not found. Please create it with the required variables :\n
                            - MODEL_PATH : Path to where the models are/will be stored\n
                            - HUGGINGFACE_MODEL_URL : HuggingFace Repository where the models are stored\n
                            - HUGGINGFACE_TOKEN : Token to access downloading HuggingFace Model Repo\n 
                            """)

MODEL_PATH = os.environ.get("MODEL_PATH")
HUGGINGFACE_MODEL_URL = os.environ.get("HUGGINGFACE_MODEL_URL")
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")


def download_sam_vit_h_checkpoint():
    """Download the SAM ViT-H checkpoint if it does not exist."""
    output_dir = os.path.join(MODEL_PATH, "sam_vit_h_4b8939.pth")
    if os.path.exists(output_dir):
        printr(f"{output_dir} already exists. Skipping download.")
        return True
    try:
        printr(f"Downloading SAM ViT-H checkpoint from HuggingFace ...")
        snapshot_download( 
            token=HUGGINGFACE_TOKEN,
            repo_id=HUGGINGFACE_MODEL_URL,
            repo_type="model",
            allow_patterns=["sam_vit_h_4b8939.pth"],
            local_dir=MODEL_PATH
        )
        printr(f"Download completed and saved to {output_dir}.")
        return True
    except Exception as e:
        printr(f"An error occurred while downloading: {e}")
        return False
    
def download_oneformer_model():
    """Download the OneFormer model if it does not exist."""
    output_dir = os.path.join(MODEL_PATH, "models--shi-labs--oneformer_ade20k_swin_large")
    if os.path.exists(output_dir):
        printr(f"{output_dir} already exists. Skipping download.")
        return True
    try:
        printr(f"Downloading OneFormer model from HuggingFace ...")
        snapshot_download(
            token=HUGGINGFACE_TOKEN,
            repo_id=HUGGINGFACE_MODEL_URL,
            repo_type="model",
            allow_patterns=["models--shi-labs--oneformer_ade20k_swin_large/*"],
            local_dir=MODEL_PATH
        )
        printr(f"Download completed and saved to {output_dir}.")
        return True
    except Exception as e:
        printr(f"An error occurred while downloading: {e}")
        return False
    

