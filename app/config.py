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
    # Long audio is split into windows of this length before transcription.
    # Granite Speech degenerates (repetition loops, <|fim_*|> garbage) on inputs
    # longer than ~30-60s, so we chunk and concatenate. Boundaries snap to the
    # quietest point nearby to avoid cutting words.
    chunk_seconds: float = 30.0
    chunk_boundary_search_seconds: float = 3.0
    # Plus-model speaker attribution only: seed each chunk with an open
    # [Speaker N]: tag for the previous chunk's final speaker ("incremental
    # decoding / prefix passing"). EXPERIMENTAL and off by default: via the
    # public generate() API this either makes the model emit EOS early (lost
    # content) or lets speaker numbers run away. True cross-chunk speaker
    # identity needs a real diarization model (e.g. pyannote). See README.
    speaker_prefix_passing: bool = False
    cors_origins: str = "*"

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
