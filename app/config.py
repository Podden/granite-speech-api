from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRANITE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    default_model: str = "ibm-granite/granite-speech-4.1-2b"

    device: Literal["auto", "cuda", "cpu"] | str = "auto"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"

    max_audio_seconds: int = 6000
    # Max upload size in bytes (default 200 MB)
    max_upload_bytes: int = 200 * 1024 * 1024
    cors_origins: str = "*"

    # Append-only JSONL file for user feedback (mounted volume in Docker).
    feedback_file: str = "feedback.jsonl"

    # generate() repetition penalty (1.0 = off). Chunking already prevents most
    # repetition loops; raise this (e.g. 1.1-1.3) only if loops persist.
    repetition_penalty: float = 1.0

    # Auto-unload the loaded model after this many seconds of inactivity to free
    # VRAM/RAM. Set to -1 (or 0) to disable. The check runs on a background
    # task; precision is roughly `idle_check_interval`.
    idle_unload_seconds: int = 600
    idle_check_interval: int = 30

    def resolved_device(self) -> str:
        if self.device == "auto":
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            return "cpu"
        # ROCm uses the cuda device string in torch
        if self.device == "rocm":
            return "cuda"
        return self.device

    def cors_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
