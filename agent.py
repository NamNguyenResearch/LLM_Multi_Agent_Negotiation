"""
agent.py – single-LLM wrapper for LLM-Deliberation

Backends supported:
  • Ollama (local, OpenAI-compatible): llama2:*, llama3:*, llama3.1:*, llama3.2:*,
    qwen3:*, llava:* (also alias 'gwen3')
  • Together AI (OpenAI-compatible):
      meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
      lgai/exaone-deep-32b
    (You can also paste a full Together URL like
     'https://api.together.ai/models/lgai/exaone-deep-32b'; it will be normalized.)
  • OpenAI (GPT)
  • Azure OpenAI
  • Google Gemini
  • Hugging Face local pipelines (via hf_models mapping)

Notes:
  - Ollama and Together AI reuse the OpenAI chat.completions schema.
  - For Together AI, model can be a bare id ("meta-llama/...", "lgai/...") or a full URL.
  - Environment variables:
        # Ollama
        export OPENAI_API_BASE="http://localhost:11434/v1"
        export OPENAI_API_KEY="ollama-local"
        # Together
        export TOGETHER_API_KEY="...your key..."
        export TOGETHER_BASE_URL="https://api.together.xyz/v1"  # optional (default)
"""

import os
import json
import argparse
import time
import re
import random
import numpy as np

# OpenAI SDK (used for OpenAI, Azure OpenAI, Ollama, and Together endpoints)
from openai import OpenAI
from openai import AzureOpenAI


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# Model tags that indicate a local Ollama model
OLLAMA_PREFIXES = (
    "llama2",
    "llama3", "llama3.1", "llama3.2",
    "qwen3", "gwen3",
    "llava",
)

# Strings that indicate a Together AI model id / URL
TOGETHER_HINTS = (
    "meta-llama/",                 # Llama-3.3-* etc.
    "lgai/",                       # EXAONE family
    "https://api.together.ai/models/",
    "https://api.together.xyz/models/",
    "together:",                   # optional local alias style
    "together.ai/",
    "together.xyz/",
)

# Friendly-name aliases → canonical Together model IDs
TOGETHER_ALIASES = {
    # allow user to write just the marketing name
    "llama-3.3-70b-instruct-turbo-free": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "exaone deep 32b": "lgai/exaone-deep-32b",
    "exaone-deep-32b": "lgai/exaone-deep-32b",
}


def _normalize_together_model_id(model: str) -> str:
    """
    Accepts:
      • bare Together model id, e.g. 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free'
        or 'lgai/exaone-deep-32b'
      • full Together URL, e.g. 'https://api.together.ai/models/lgai/exaone-deep-32b'
      • alias 'together:<model_id>'
      • friendly names found in TOGETHER_ALIASES
    Returns a model id suitable for the API call.
    """
    m = model.strip()

    # Friendly-name aliases (case-insensitive)
    alias_key = m.lower().strip()
    if alias_key in TOGETHER_ALIASES:
        return TOGETHER_ALIASES[alias_key]

    low = m.lower()
    if "/models/" in low and ("api.together.ai" in low or "api.together.xyz" in low):
        # strip everything up to and including '/models/'
        return m.split("/models/", 1)[1]
    if m.startswith("together:"):
        return m.split(":", 1)[1]
    if m.startswith("together.ai/") or m.startswith("together.xyz/"):
        return m.split("/", 1)[1]
    return m


class Agent:
    # -------------------------------------------------------------------------
    def __init__(
        self,
        initial_prompt_cls,
        round_prompt_cls,
        agent_name,
        temperature,
        model,
        rounds_num=24,
        agents_num=6,
        azure=False,
        hf_models={},
    ):
        # ---------------- Basics -----------------
        self.model = model  # keep original string the user passed
        self.agent_name = agent_name
        self.temperature = temperature
        self.initial_prompt_cls = initial_prompt_cls
        self.round_prompt_cls = round_prompt_cls
        self.rounds_num = rounds_num
        self.agents_num = agents_num

        self.initial_prompt = initial_prompt_cls.return_initial_prompt()
        self.messages = [{"role": "user", "content": self.initial_prompt}]

        # Routing flags
        self.is_ollama = False
        self.is_together = False

        # --------------- Model setup -------------
        self.azure = azure
        model_lc = self.model.lower()

        # ——— Google Gemini (lazy import to avoid hard dep) ————————
        if "gemini" in model_lc:
            try:
                from vertexai.preview.generative_models import GenerativeModel
            except Exception as e:
                raise ImportError(
                    "Gemini backend requested but 'google-cloud-aiplatform' "
                    "is not installed."
                ) from e
            self.model_instance = GenerativeModel(self.model)

        # ——— Azure OpenAI ——————————————————————————
        if self.azure:
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2023-05-15",
            )
            # Azure path uses the deployment name in self.model
            self.is_ollama = False
            self.is_together = False

        # ——— Together AI (OpenAI-compatible) —————————
        else:
            # Normalize potential Together aliases early (does nothing for others)
            normalized_candidate = _normalize_together_model_id(self.model)
            normalized_lc = normalized_candidate.lower()

            if any(h in normalized_lc for h in TOGETHER_HINTS):
                together_base = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
                together_key = os.getenv("TOGETHER_API_KEY", "716e0c4477c986e754186744601e0e93f1de7eee0d5836b5886bd2170e9774e1")
                if not together_key:
                    raise ValueError("TOGETHER_API_KEY is not set")
                self.model = normalized_candidate
                self.client = OpenAI(base_url=together_base, api_key=together_key)
                self.is_together = True
                self.is_ollama = False

            # ——— Local Ollama models ———————————————————
            elif any(p in model_lc for p in OLLAMA_PREFIXES):
                base_url = os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
                api_key = os.getenv("OPENAI_API_KEY", "ollama-local")
                self.client = OpenAI(base_url=base_url, api_key=api_key)
                self.is_ollama = True
                self.is_together = False

            # ——— OpenAI-hosted GPT-3.5 / GPT-4 ——————————
            elif "gpt" in model_lc:
                self.client = OpenAI()
                self.is_ollama = False
                self.is_together = False

        # ——— Hugging Face Transformers ————————————
        self.hf_local = "hf" in model_lc
        if self.hf_local:
            (self.hf_model, self.hf_tokenizer, self.hf_pipeline_gen) = hf_models[self.model]

    # -------------------------------------------------------------------------
    def execute_round(self, answer_history, round_idx):
        slot_prompt = self.round_prompt_cls.build_slot_prompt(
            answer_history, round_idx
        )
        agent_response = self.prompt("user", slot_prompt)
        return slot_prompt, agent_response

    # -------------------------------------------------------------------------
    def prompt(self, role, msg):
        messages = self.messages + [{"role": role, "content": msg}]

        # —— GPT, Ollama, Together share OpenAI schema —————————
        if (("gpt" in self.model.lower()) or self.is_ollama or self.is_together) and not self.azure:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            # OpenAI-style response
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return content

        # —— Azure OpenAI ——————————————————————————————
        elif "gpt" in self.model.lower() and self.azure:
            response = self.client.chat.completions.create(
                model=self.model,  # deployment name
                messages=messages,
                temperature=self.temperature,
            )
            content = response.choices[0].message.content
            self.messages.append({"role": "assistant", "content": content})
            return content

        # —— Google Gemini ————————————————————————————
        elif "gemini" in self.model.lower():
            responses = self.model_instance.generate_content(
                self.initial_prompt + msg,
                generation_config={"temperature": self.temperature, "top_p": 1},
                stream=True,
            )
            text = ""
            for r in responses:
                if hasattr(r, "text"):
                    text += r.text or ""
            # keep parity with other backends and store assistant reply
            self.messages.append({"role": "assistant", "content": text})
            return text

        # —— Hugging-Face local models ——————————————
        elif self.hf_local:
            chat = [{"role": "user", "content": self.initial_prompt + msg}]
            model_input = self.hf_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            output_text = self.hf_pipeline_gen(
                model_input, do_sample=True, temperature=self.temperature
            )[0]["generated_text"]
            self.messages.append({"role": "assistant", "content": output_text})
            return output_text

        # —— Unsupported ——————————————————————————————
        raise ValueError(f"Unhandled model backend for: {self.model}")


# Optional: quick smoke test when running this file directly
if __name__ == "__main__":
    # Example env:
    #   # Ollama
    #   export OPENAI_API_BASE=http://localhost:11434/v1
    #   export OPENAI_API_KEY=ollama-local
    #   # Together
    #   export TOGETHER_API_KEY=sk-...
    #   export TOGETHER_BASE_URL=https://api.together.xyz/v1
    class _InitPrompt:
        @staticmethod
        def return_initial_prompt():
            return "You are a helpful agent. "

    class _RoundPrompt:
        @staticmethod
        def build_slot_prompt(history, idx):
            return f"Round {idx}: Propose a deal in <DEAL>...</DEAL>."

    # Try different backends by setting AGENT_TEST_MODEL, e.g.:
    #   export AGENT_TEST_MODEL="llama3.1:8b"                                   # Ollama
    #   export AGENT_TEST_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"   # Together
    #   export AGENT_TEST_MODEL="lgai/exaone-deep-32b"                           # Together
    #   export AGENT_TEST_MODEL="https://api.together.ai/models/lgai/exaone-deep-32b"  # Together (URL)
    #   export AGENT_TEST_MODEL="gpt-4o"                                         # OpenAI
    model = os.environ.get("AGENT_TEST_MODEL", "llama3.1:8b")

    agent = Agent(
        initial_prompt_cls=_InitPrompt,
        round_prompt_cls=_RoundPrompt,
        agent_name="Tester",
        temperature=0.7,
        model=model,
    )
    prompt = _RoundPrompt.build_slot_prompt([], 1)
    print(agent.prompt("user", prompt))
