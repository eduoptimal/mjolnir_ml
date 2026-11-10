"""mjolnir_rlvf.config — paths, model and sampling defaults (env-overridable).

Distilled from mjolnir_research/mjolnir_experiments/rlvf/config.py with two
product decisions:

  * SUITES/BENCH ship inside the package (read-only benchmark data).
  * run ARTIFACTS never write into the installed package: they default to
    ./rlvf_data in the working directory (MJ_RLVF_DATA overrides).
"""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.abspath(__file__))          # .../mjolnir_rlvf
BENCH = os.environ.get("RLVF_BENCH", os.path.join(ROOT, "benchmark"))
SUITES = os.environ.get("RLVF_SUITES", os.path.join(BENCH, "tasks"))
SPLITS = os.environ.get("RLVF_SPLITS", os.path.join(BENCH, "splits"))

# run artifacts: cwd-scoped by default (never site-packages)
DATA = os.environ.get("MJ_RLVF_DATA",
                      os.environ.get("RLVF_DATA",
                                     os.path.join(os.getcwd(), "rlvf_data")))

# the current trainable default (Qwen 1.5B instruct); any id overrides
DEFAULT_MODEL = os.environ.get("RLVF_MODEL", "qwen2.5-coder:1.5b-instruct")
MLX_MODEL = os.environ.get("RLVF_MLX_MODEL",
                           "Qwen/Qwen2.5-Coder-1.5B-Instruct")
MLX_ADAPTER = os.environ.get("RLVF_MLX_ADAPTER") or None

OLLAMA_URL = os.environ.get("RLVF_OLLAMA_URL", "http://127.0.0.1:11434")
OPENAI_BASE_URL = os.environ.get(
    "OPENAI_BASE_URL", os.environ.get("RLVF_OPENAI_BASE_URL",
                                      "https://api.openai.com/v1"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

BACKEND = os.environ.get("RLVF_BACKEND", "ollama")   # ollama | openai | mlx
DEFAULT_K = int(os.environ.get("RLVF_K", "8"))
TEMPERATURE = float(os.environ.get("RLVF_TEMPERATURE", "0.6"))
SEED = int(os.environ.get("RLVF_SEED", "0"))
NUM_PREDICT = int(os.environ.get("RLVF_NUM_PREDICT", "512"))
REQUEST_TIMEOUT = int(os.environ.get("RLVF_REQUEST_TIMEOUT", "300"))

# -- serve: the OpenAI-compatible endpoint (rlvf_agent env names kept)
SERVED_NAME = os.environ.get("RLVF_AGENT_MODEL", "rlvf-qwen2.5-coder-1.5b")
SERVE_HOST = os.environ.get("RLVF_AGENT_HOST", "127.0.0.1")
SERVE_PORT = int(os.environ.get("RLVF_AGENT_PORT", "8000"))
SERVE_BACKEND = os.environ.get("RLVF_AGENT_BACKEND", "mlx")   # mlx | ollama
SERVE_BASE_MODEL = os.environ.get("RLVF_AGENT_BASE_MODEL", MLX_MODEL)
SERVE_ADAPTER = os.environ.get("RLVF_AGENT_ADAPTER") or None
SERVE_OLLAMA_MODEL = os.environ.get(
    "RLVF_AGENT_OLLAMA_MODEL", "qwen2.5-coder:1.5b-instruct")
SERVE_TEMPERATURE = float(os.environ.get("RLVF_AGENT_TEMPERATURE", "0.0"))
SERVE_MAX_TOKENS = int(os.environ.get("RLVF_AGENT_MAX_TOKENS", "512"))
SERVE_SEED = int(os.environ.get("RLVF_AGENT_SEED", "0"))
# the trained contract is injected when the caller sends no system prompt
INJECT_SYSTEM = os.environ.get(
    "RLVF_AGENT_INJECT_SYSTEM", "1") not in ("0", "false")
