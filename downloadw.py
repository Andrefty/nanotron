from huggingface_hub import snapshot_download

snapshot_download(repo_id="HF1BitLLM/Llama3-8B-1.58-100B-tokens",
                  local_dir = "models/Llama3-8B-1.58-100B-tokens",
                  local_dir_use_symlinks=False,
                  ignore_patterns=["original/*"]) # Llama3 models in the Hub contain the original checkpoints. We just want the HF checkpoint stored in the safetensor format