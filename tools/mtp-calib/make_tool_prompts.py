"""Make tool-calling turns for the calibration of the draft head.

A tool call has two halves that the drafter meets: the turn in which the
model emits the call, and the turn in which it answers after the tool result
comes back. Half of the prompts here are one user turn that asks for a tool,
the other half carry a finished call and its result, thus the model produces
the answer. Each line of the output is either a JSON string, a user turn, or
a JSON array of messages in the OpenAI shape that llama-server renders
through the template of the model.

The free text of the arguments and of the results comes from the calibration
prompts, thus the tool turns carry the same four buckets of language as the
chat turns. The application has no tool schema yet, thus the tools are a
representative set that matches ``TOOLS`` in ``calib_server.py``.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CITIES = ["Berlin", "Moscow", "Москва", "Санкт-Петербург", "Tokyo", "東京", "Paris", "Madrid", "Cairo",
          "القاهرة", "Istanbul", "Warsaw", "Warszawa", "Prague", "Praha", "Athens", "Αθήνα", "Tehran",
          "Jakarta", "Rome", "Amsterdam", "Lisbon", "Stockholm", "Hanoi", "Hà Nội", "Budapest",
          "Novosibirsk", "Казань", "Munich", "München", "Beijing", "北京", "New York", "London"]
ZONES = ["Europe/Berlin", "Europe/Moscow", "Asia/Tokyo", "Europe/Paris", "America/New_York",
         "Asia/Shanghai", "Europe/Istanbul", "Asia/Jakarta", "Europe/Warsaw", "Asia/Tehran"]
NAMES = ["Anna", "Grigory", "Иван", "Мария", "Wei", "Fatima", "Jonas", "Lucía", "Kenji", "Olga"]
PATHS = ["/sdcard/Documents/report.txt", "notes/todo.md", "/sdcard/Download/data.csv",
         "src/main.py", "README.md", "/sdcard/notes/идеи.txt", "config.yaml", "logs/app.log"]
CODES = ["print(sum(range(1, 101)))", "import math\nprint(math.factorial(12))",
         "for i in range(5):\n    print(i * i)", "print(sorted([3, 1, 2]))",
         "print(len('hello world'.split()))", "x = [n for n in range(20) if n % 3 == 0]\nprint(x)"]

# One template per tool per language. {q} takes a text snippet.
ASK: dict[str, list[str]] = {
    "web_search": ["Search the web for {q}", "Look up online: {q}", "Найди в интернете: {q}",
                   "Поищи информацию про {q}", "Suche im Web nach {q}", "Busca en la web: {q}",
                   "在网上搜索：{q}"],
    "get_weather": ["What's the weather in {city} right now?", "Какая сейчас погода в {city}?",
                    "Wie ist das Wetter in {city}?", "¿Qué tiempo hace hoy en {city}?", "{city}现在天气怎么样？"],
    "calculator": ["Calculate {expr}", "What is {expr}?", "Вычисли {expr}", "Сколько будет {expr}?",
                   "Berechne {expr}", "Calcula {expr}"],
    "run_python": ["Run this Python code and show the output:\n{code}",
                   "Выполни этот код на Python и покажи вывод:\n{code}",
                   "Führe diesen Python-Code aus und zeige die Ausgabe:\n{code}"],
    "read_file": ["Read the file {path} and summarize it", "Прочитай файл {path} и кратко перескажи",
                  "Lies die Datei {path} und fasse sie zusammen", "Lee el archivo {path} y resúmelo"],
    "write_file": ["Save the following text to {path}:\n{q}", "Сохрани этот текст в файл {path}:\n{q}",
                   "Speichere diesen Text in {path}:\n{q}"],
    "get_current_time": ["What time is it now in {tz}?", "Который сейчас час в {tz}?",
                         "Wie spät ist es in {tz}?", "¿Qué hora es en {tz}?"],
    "send_message": ["Send a message to {name}: {q}", "Отправь сообщение {name}: {q}",
                     "Schick {name} eine Nachricht: {q}", "Envía un mensaje a {name}: {q}"],
}


def expression(rng: random.Random) -> str:
    """Give a small arithmetic expression.

    Args:
        rng: The random source

    Returns:
        The expression
    """
    a, b, c = rng.randint(2, 999), rng.randint(2, 99), rng.randint(2, 50)
    return rng.choice([f"{a} * {b}", f"{a} + {b} * {c}", f"({a} - {b}) / {c}", f"{a} ** 2 + {b}",
                       f"{a} % {b}", f"sqrt({a * a})", f"{a} / {b}"])


def fill(tool: str, rng: random.Random, snippet: str) -> tuple[str, dict, str]:
    """Give one user turn, the arguments of the call it asks for, and a result.

    Args:
        tool: The tool name
        rng: The random source
        snippet: Free text for the arguments and the results

    Returns:
        The user turn, the arguments, and the JSON result of the tool
    """
    q = snippet[:120].strip()
    t = rng.choice(ASK[tool])
    if tool == "web_search":
        args = {"query": q}
        result = json.dumps({"results": [{"title": snippet[:60].strip(), "snippet": snippet[60:200].strip()}
                                         for _ in range(2)]}, ensure_ascii=False)
    elif tool == "get_weather":
        city = rng.choice(CITIES)
        args = {"location": city}
        result = json.dumps({"location": city, "temperature_c": rng.randint(-15, 38),
                             "condition": rng.choice(["clear", "cloudy", "rain", "snow", "fog"])})
    elif tool == "calculator":
        expr = expression(rng)
        args = {"expression": expr}
        result = json.dumps({"result": round(rng.uniform(-1000, 100000), 3)})
    elif tool == "run_python":
        code = rng.choice(CODES)
        args = {"code": code}
        result = json.dumps({"stdout": str(rng.randint(0, 10000)) + "\n", "returncode": 0})
    elif tool == "read_file":
        path = rng.choice(PATHS)
        args = {"path": path}
        result = json.dumps({"path": path, "content": snippet[:400]}, ensure_ascii=False)
    elif tool == "write_file":
        path = rng.choice(PATHS)
        args = {"path": path, "content": q}
        result = json.dumps({"path": path, "bytes": len(q.encode())})
    elif tool == "get_current_time":
        tz = rng.choice(ZONES)
        args = {"timezone": tz}
        stamp = f"2026-09-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"
        result = json.dumps({"timezone": tz, "time": stamp})
    else:
        name = rng.choice(NAMES)
        args = {"recipient": name, "text": q}
        result = json.dumps({"sent": True, "recipient": name})
    user = t.format(q=q, city=args.get("location", ""), expr=args.get("expression", ""),
                    code=args.get("code", ""), path=args.get("path", ""),
                    tz=args.get("timezone", ""), name=args.get("recipient", ""))
    return user, args, result


def main() -> None:
    """Write the tool turns, one JSON value per line, shuffled."""
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--base", type=Path, default=Path.cwd(), help="the corpus directory")
    ap.add_argument("--source", type=Path, default=None, help="default base/prompts-calib.txt")
    ap.add_argument("--out", type=Path, default=None, help="default base/prompts-tools.txt")
    ap.add_argument("--count", type=int, default=20000)
    a = ap.parse_args()
    base = a.base.resolve()
    source = a.source or base / "prompts-calib.txt"
    out_path = a.out or base / "prompts-tools.txt"

    rng = random.Random(7)
    snippets = [json.loads(x) for x in source.read_text().splitlines() if x.strip()]
    tools = list(ASK)
    out = []
    for i in range(a.count):
        tool = tools[i % len(tools)]
        user, args, result = fill(tool, rng, rng.choice(snippets))
        if i % 2 == 0:
            out.append(user)
            continue
        call_id = f"call_{i}"
        out.append([
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function",
                 "function": {"name": tool, "arguments": json.dumps(args, ensure_ascii=False)}}]},
            {"role": "tool", "tool_call_id": call_id, "name": tool, "content": result},
        ])
    rng.shuffle(out)
    out_path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in out) + "\n")
    asks = sum(1 for p in out if isinstance(p, str))
    print(f"wrote {len(out)} tool turns to {out_path}: {asks} ask for a call, "
          f"{len(out) - asks} answer after a result", flush=True)


if __name__ == "__main__":
    main()
