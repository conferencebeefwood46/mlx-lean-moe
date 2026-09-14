from pathlib import Path

# The qwen3_5 real-weight validation checkpoint, resolved out of the Hugging
# Face cache. A wrong id here reads as a green run that skipped everything.
QWEN3_5_REPO = "froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit"


def _validation_checkpoint() -> Path:
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(QWEN3_5_REPO, local_files_only=True))
    except Exception:
        # Not cached. Return a path that does not exist, so the skip
        # conditions below fire rather than an import blowing up collection.
        return (
            Path.home()
            / ".cache/huggingface/hub"
            / f"models--{QWEN3_5_REPO.replace('/', '--')}"
        )


QWEN3_5_MODEL_DIR = _validation_checkpoint()
