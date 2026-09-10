"""Budget forcing for Qwen3.8 as a sidecar module: import it from your own OpenAI-compatible proxy and call
needs_splice() on each chat completion; on True, call splice_nonstream() or splice_stream(). Env: AEON_THINK_BUDGET (0 disables),
AEON_ANSWER_BUDGET (default 2048), AEON_SPLICE_MODE (bare|sentence).

Policy (minimal intervention, measured 2026-09-08 on the 6-bit lane): pass 1 keeps the client's own max_tokens; only
when the backend stops with finish_reason=length while still inside <think> (reasoning emitted, no content, no tool
calls) do we run pass 2: re-render the same prompt with the model's chat template, append the partial reasoning and a
bare "</think>", and continue on /v1/completions for ANSWER_BUDGET tokens. The shared prefix hits the prefix cache.
Qwen <tool_call> XML in the continued answer is parsed back into OpenAI tool_calls.
"""
from __future__ import annotations
import json, os, re, time, uuid
import requests

ANSWER_BUDGET = int(os.environ.get("AEON_ANSWER_BUDGET", "2048"))
SPLICE_MODE = os.environ.get("AEON_SPLICE_MODE", "bare")
SPLICES = {"bare": "\n</think>\n\n",
           "sentence": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"}
_tok = None

def enabled() -> bool:
    return os.environ.get("AEON_THINK_BUDGET", "1") not in ("0", "false", "no")

def _tokenizer(model_dir: str):
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(model_dir)
    return _tok

def render_prompt(payload: dict, model_dir: str) -> str:
    kw = dict(payload.get("chat_template_kwargs") or {})
    kw.setdefault("enable_thinking", payload.get("enable_thinking", True))
    eff = kw.get("reasoning_effort", payload.get("reasoning_effort"))
    if eff: kw["reasoning_effort"] = eff
    msgs = []
    for m in payload["messages"]:
        m = json.loads(json.dumps(m))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try: fn["arguments"] = json.loads(fn["arguments"])
                except Exception: pass
        msgs.append(m)
    return _tokenizer(model_dir).apply_chat_template(msgs, tools=payload.get("tools"), tokenize=False, add_generation_prompt=True, **kw)

def parse_tool_calls(text: str):
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", text, re.S):
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S):
            v = pm.group(2)
            try: v = json.loads(v)
            except Exception: pass
            args[pm.group(1)] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function", "function": {"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    return re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip(), calls

def needs_splice(finish_reason, reasoning: str, content: str, tool_calls) -> bool:
    return enabled() and finish_reason == "length" and bool(reasoning.strip()) and not (content or "").strip() and not tool_calls

def _completion_body(payload: dict, prompt: str, stream: bool) -> dict:
    return {"model": payload.get("model"), "prompt": prompt, "max_tokens": ANSWER_BUDGET, "stream": stream,
            "temperature": payload.get("temperature", 1.0), "top_p": payload.get("top_p", 0.95), "top_k": payload.get("top_k", 20),
            "presence_penalty": payload.get("presence_penalty", 0.0), "seed": payload.get("seed")}

def splice_nonstream(backend_url: str, payload: dict, model_dir: str, reasoning: str, base_response: dict, timeout: float) -> dict:
    prompt = render_prompt(payload, model_dir) + reasoning.rstrip() + SPLICES[SPLICE_MODE]
    r2 = requests.post(f"{backend_url}/v1/completions", json=_completion_body(payload, prompt, False), timeout=timeout).json()
    text = r2["choices"][0].get("text", ""); fin = r2["choices"][0].get("finish_reason") or "stop"
    content, calls = parse_tool_calls(text)
    msg = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls: msg["tool_calls"] = calls
    out = dict(base_response); out["choices"] = [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else fin}]
    u1, u2 = base_response.get("usage") or {}, r2.get("usage") or {}
    out["usage"] = {**u1, "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0), "spliced": True}
    return out

def splice_stream(backend_url: str, payload: dict, model_dir: str, reasoning: str, base: dict, timeout: float):
    """Yield OpenAI chat SSE 'data:' strings continuing the answer after a forced </think>."""
    prompt = render_prompt(payload, model_dir) + reasoning.rstrip() + SPLICES[SPLICE_MODE]
    resp = requests.post(f"{backend_url}/v1/completions", json=_completion_body(payload, prompt, True), timeout=timeout, stream=True)
    buf = ""; held = False; fin = "stop"
    def chunk(delta, finish=None):
        return "data: " + json.dumps({**base, "object": "chat.completion.chunk", "created": int(time.time()), "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "): continue
        data = line[6:].strip()
        if data == "[DONE]": break
        try: c = json.loads(data)
        except Exception: continue
        ch = (c.get("choices") or [{}])[0]; text = ch.get("text") or ""; fin = ch.get("finish_reason") or fin
        if text:
            buf += text
            if not held and "<tool_call" in buf: held = True
            if not held:
                # stream everything except a possible partial '<tool_call' opener at the tail
                cut = buf.rfind("<")
                emit, buf = (buf, "") if cut < 0 or len(buf) - cut > 12 else (buf[:cut], buf[cut:])
                if emit: yield chunk({"content": emit})
    content, calls = parse_tool_calls(buf)
    if content: yield chunk({"content": content})
    if calls: yield chunk({"tool_calls": [{"index": i, **c} for i, c in enumerate(calls)]})
    yield chunk({}, "tool_calls" if calls else fin)
    yield "data: [DONE]\n\n"
