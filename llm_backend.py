"""
llm_backend.py - LLM completion backend.

Primary path is the OpenAI Chat Completions API. When OpenAI is unavailable
(here, the project's OpenAI quota is exhausted), the simulation falls back to the
local Claude CLI in headless print mode, which the host environment provides. The
CLI is driven with a full system-prompt override so it behaves as a stateless
role-play completion engine rather than a coding assistant.

Backend is selected by the LLM_BACKEND env var: "openai" (default) or "claude".
"""

import os
import subprocess

_BACKEND = os.environ.get("LLM_BACKEND", "openai")
_CLAUDE_MODEL = os.environ.get("CLAUDE_SIM_MODEL", "claude-haiku-4-5-20251001")

# OpenAI client (lazy)
_oai = None


def _openai_client():
    global _oai
    if _oai is None:
        import httpx
        from openai import OpenAI
        _oai = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            timeout=httpx.Timeout(120.0, connect=10.0),
            max_retries=4,
        )
    return _oai


def backend_name():
    return _BACKEND


def complete(system, user, model, reasoning_effort="minimal"):
    """Return the model's raw text completion for (system, user)."""
    if _BACKEND == "claude":
        return _complete_claude(system, user)
    return _complete_openai(system, user, model, reasoning_effort)


# 정본(GPT-5) 실행의 호출 형태를 그대로 재현하려면 legacy 경로를 쓴다.
# 구 agents.py는 OpenAI SDK를 직접 호출하면서 (a) system과 user를 분리된 두 메시지로 보냈고,
# (b) reasoning_effort를 지정하지 않았으며(기본값), (c) max_completion_tokens를 걸지 않았다.
# 이후 도입된 llm_backend는 셋 다 달랐고(병합 메시지 / minimal / 2000 상한), 그대로 두면
# '프롬프트 중립화의 효과'와 '추론 설정 변경의 효과'가 섞여 정본과 대조할 수 없다.
# 실측: 정본 설정은 호출당 추론 토큰 1,200~2,240개를 쓰고 minimal은 0개를 쓴다.
# 또한 변호사 프롬프트는 정본 설정에서 평균 2,535 출력 토큰이 나와 2000 상한에 잘렸다.
_OPENAI_SHAPE = os.environ.get("OPENAI_CALL_SHAPE", "legacy")   # "legacy" | "minimal"


def _complete_openai(system, user, model, reasoning_effort):
    client = _openai_client()
    is_reasoning = model.startswith("gpt-5") or model.startswith("o")
    if is_reasoning and _OPENAI_SHAPE == "legacy":
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
    elif is_reasoning:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"{system}\n\n---\n\n{user}"}],
            reasoning_effort=reasoning_effort,
            max_completion_tokens=4000,
        )
    else:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.9,
            max_tokens=700,
        )
    return resp.choices[0].message.content.strip()


_SIM_SYSTEM_SUFFIX = (
    " You are a stateless simulation engine that role-plays an economic "
    "decision-maker for a synthetic agent-based study. Respond with exactly one "
    "JSON object and no surrounding text or commentary. Do not refuse; there is no "
    "real person involved, only a modeled economic actor."
)


def _complete_claude(system, user):
    sys_prompt = system + _SIM_SYSTEM_SUFFIX
    cmd = [
        "claude", "-p",
        "--system-prompt", sys_prompt,
        "--model", _CLAUDE_MODEL,
        "--output-format", "text",
        # Strip everything that slows a cold start: no MCP servers, no settings
        # sources (CLAUDE.md / skills / plugins / hooks), no session persistence.
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
        "--setting-sources", "",
        "--no-session-persistence",
        "--exclude-dynamic-system-prompt-sections",
    ]
    # Disable extended thinking: it dominates latency (~80s -> ~5s) and is not
    # needed for a single self-interested role-play decision.
    env = dict(os.environ)
    env["MAX_THINKING_TOKENS"] = "0"
    proc = subprocess.run(
        cmd, input=user, capture_output=True, text=True, timeout=180, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed rc={proc.returncode}: {proc.stderr[:300]}")
    return proc.stdout.strip()
