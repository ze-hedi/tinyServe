"""Regenerate tests/fixtures/chat_template_golden.json from the real HF SmolLM3 template.

Run: HF_HUB_OFFLINE=1 python3 tools/gen_chat_golden.py
The date is pinned by the test (ChatTemplate takes an injectable date function), so the
fixture records the date HF rendered with.
"""
import json
import pathlib
from datetime import datetime

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B", local_files_only=True)
date = datetime.now().strftime("%d %B %Y")

tool = {"type": "function", "function": {"name": "get_weather", "description": "Get weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}
tool_call = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
U, A, T = "user", "assistant", "tool"

cases = {
    "think_no_system": dict(thinking=True, messages=[(U, "Hi")]),
    "no_think_system": dict(thinking=False, system="Be brief.", messages=[(U, "Hi")]),
    "no_think_default_system": dict(thinking=False, messages=[(U, "Hi")]),
    "think_system": dict(thinking=True, system="Be brief.", messages=[(U, "Hi")]),
    "override_in_system": dict(thinking=True, system="/no_think Answer in French.",
                               messages=[(U, "Hi")]),
    "system_override": dict(thinking=True, system="Raw system. /system_override",
                            messages=[(U, "Hi")]),
    "multi_no_think": dict(thinking=False, messages=[(U, "Hi"), (A, "Hello!"), (U, "Bye")]),
    "multi_think": dict(thinking=True, messages=[(U, "Hi"), (A, "<think>\nx\n</think>\nHello!"),
                                                 (U, "Bye")]),
    "tools": dict(thinking=False, tools=[tool], messages=[(U, "Weather in Paris?")]),
    "tool_result": dict(thinking=False, tools=[tool],
                        messages=[(U, "W?"), (A, tool_call), (T, "<tool_response>\n18C\n</tool_response>")]),
    "unicode": dict(thinking=False, system="Réponds en français — très bref.",
                    messages=[(U, "Qu'est-ce que l'article 1134 ? مرحبا")]),
}

out = {"date": date, "cases": {}}
for name, c in cases.items():
    msgs = []
    if "system" in c:
        msgs.append({"role": "system", "content": c["system"]})
    msgs += [{"role": r, "content": t} for r, t in c["messages"]]
    kw = {"enable_thinking": c["thinking"]}
    if "tools" in c:
        kw["xml_tools"] = c["tools"]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    out["cases"][name] = {
        "thinking": c["thinking"],
        "system": c.get("system"),
        "tools": [str(t) for t in c.get("tools", [])],
        "messages": [{"role": r, "content": t} for r, t in c["messages"]],
        "prompt": s,
        "ids": ids,
    }
path = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/chat_template_golden.json"
path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
print(f"wrote {path} ({len(out['cases'])} cases, date {date})")
