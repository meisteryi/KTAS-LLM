import os
import sys
from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-32B-Instruct"
target_dir = os.path.abspath("models/Qwen2.5-32B-Instruct")

print(f"[*] Starting download of {model_id} to {target_dir}...")
os.makedirs(target_dir, exist_ok=True)

try:
    path = snapshot_download(
        repo_id=model_id,
        local_dir=target_dir,
        resume_download=True,
        max_workers=4
    )
    print(f"\n[+] Successfully downloaded {model_id} to {path}")
except Exception as e:
    print(f"\n[-] Error downloading model: {e}", file=sys.stderr)
    sys.exit(1)

