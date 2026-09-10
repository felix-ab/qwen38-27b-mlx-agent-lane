#!/usr/bin/env python3
"""Thinking-budget proxy for Qwen3.8 on mlx-dspark (non-streaming chat completions).

Pass 1: forward the chat request with max_tokens = min(client max_tokens, THINK_BUDGET).
If it stops with finish_reason=length while still inside <think> (reasoning present, no content, no tool calls),
pass 2: re-render the same prompt with the model's chat template, append the partial reasoning plus Qwen's official
budget splice ("Considering the limited time by the user, I have to give the solution based on the thinking directly
now.") and a closed </think>, and continue with /v1/completions for ANSWER_BUDGET tokens. The shared prefix hits
mlx-dspark's prefix cache, so pass 2 costs roughly one answer's worth of decode. Tool calls in pass 2 are parsed
from Qwen's <tool_call><function=...> XML back into OpenAI tool_calls.
Usage: thinkbudget_proxy.py --upstream http://127.0.0.1:18044 --port 18045 --model-dir <dir> [--think-budget 2048] [--answer-budget 1536]
"""
import argparse, json, re, time, urllib.request, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
SPLICES = {"sentence": "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n", "bare": "\n</think>\n\n"}
SPLICE = SPLICES["sentence"]
a = None; tok = None; stats = {"requests": 0, "spliced": 0, "pass2_tool_calls": 0}
def post(path, body, timeout=1800):
    req = urllib.request.Request(a.upstream + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
def parse_tool_calls(text):
    calls = []
    for m in re.finditer(r"<tool_call>\s*<function=([^>\s]+)>(.*?)</function>\s*</tool_call>", text, re.S):
        args = {}
        for pm in re.finditer(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S):
            v = pm.group(2)
            try: v = json.loads(v)
            except Exception: pass
            args[pm.group(1)] = v
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function", "function": {"name": m.group(1), "arguments": json.dumps(args, ensure_ascii=False)}})
    content = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return content, calls
def render_prompt(body):
    kw = dict(body.get("chat_template_kwargs") or {})
    if "enable_thinking" not in kw: kw["enable_thinking"] = body.get("enable_thinking", True)
    eff = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if eff: kw["reasoning_effort"] = eff
    msgs = []
    for m in body["messages"]:
        m = json.loads(json.dumps(m))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            if isinstance(fn.get("arguments"), str):
                try: fn["arguments"] = json.loads(fn["arguments"])
                except Exception: pass
        msgs.append(m)
    return tok.apply_chat_template(msgs, tools=body.get("tools"), tokenize=False, add_generation_prompt=True, **kw)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, code, obj):
        data = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        if self.path.startswith("/stats"): return self._send(200, stats)
        try:
            with urllib.request.urlopen(a.upstream + self.path, timeout=30) as r: data = r.read(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        except Exception as e: self._send(502, {"error": str(e)})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        if not self.path.endswith("/chat/completions") or body.get("stream"):
            try: return self._send(200, post(self.path, body))
            except Exception as e: return self._send(502, {"error": str(e)})
        stats["requests"] += 1
        client_max = int(body.get("max_tokens") or a.think_budget); p1 = dict(body); p1["max_tokens"] = min(client_max, a.think_budget); p1["stream"] = False
        t0 = time.perf_counter(); r1 = post("/v1/chat/completions", p1); ch = r1["choices"][0]; m = ch["message"]
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""; content = m.get("content") or ""
        if not (ch.get("finish_reason") == "length" and reasoning and not content.strip() and not m.get("tool_calls")):
            return self._send(200, r1)
        # pass 2: splice and continue into the answer
        stats["spliced"] += 1
        prompt = render_prompt(body) + reasoning.rstrip() + SPLICE
        r2 = post("/v1/completions", {"model": body.get("model"), "prompt": prompt, "max_tokens": a.answer_budget, "temperature": body.get("temperature", 1.0),
                                      "top_p": body.get("top_p", 0.95), "top_k": body.get("top_k", 20), "presence_penalty": body.get("presence_penalty", 0.0), "seed": body.get("seed"), "stream": False})
        text = r2["choices"][0].get("text", ""); fin = r2["choices"][0].get("finish_reason")
        ans, calls = parse_tool_calls(text)
        if calls: stats["pass2_tool_calls"] += 1
        msg = {"role": "assistant", "content": ans, "reasoning_content": reasoning + SPLICE.split("</think>")[0].rstrip()}
        if calls: msg["tool_calls"] = calls
        u1, u2 = r1.get("usage", {}), r2.get("usage", {})
        out = {"id": r1.get("id"), "object": "chat.completion", "created": int(time.time()), "model": r1.get("model"),
               "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else fin}],
               "usage": {"prompt_tokens": u1.get("prompt_tokens", 0), "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
                         "total_tokens": u1.get("total_tokens", 0) + u2.get("completion_tokens", 0), "spliced": True, "pass2_prompt_tokens": u2.get("prompt_tokens", 0)},
               "thinkbudget": {"pass1_secs": round(time.perf_counter() - t0, 1), "think_budget": p1["max_tokens"], "answer_budget": a.answer_budget}}
        self._send(200, out)
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--upstream", default="http://127.0.0.1:18044"); ap.add_argument("--port", type=int, default=18045)
    ap.add_argument("--model-dir", required=True); ap.add_argument("--think-budget", type=int, default=2048); ap.add_argument("--answer-budget", type=int, default=1536); ap.add_argument("--splice-mode", choices=list(SPLICES), default="sentence")
    a = ap.parse_args(); SPLICE = SPLICES[a.splice_mode]
    from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained(a.model_dir)
    print(f"thinkbudget proxy on :{a.port} -> {a.upstream} (think {a.think_budget}, answer {a.answer_budget})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
