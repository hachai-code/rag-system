"""The deep-agent run store: events buffer server-side and replay from a cursor,
so a client that reconnects after a dropped connection misses nothing.

Run: uv run pytest rag/test_agent_run.py
"""

import json
import time

from fastapi.testclient import TestClient

from rag import app as app_module

client = TestClient(app_module.app)

EVENTS = [
    {"type": "status", "call_id": "1", "scope": "main", "tool": "retrieve", "label": "x"},
    {"type": "sources", "sources": []},
    {"type": "answer", "text": "done"},
]


def fake_stream(question, thread_id, research_budget=None):
    for event in EVENTS:
        time.sleep(0.05)  # the run outlives any single read of the stream
        yield event


def read_events(run_id: str, after: int = 0) -> list[dict]:
    with client.stream("GET", f"/ask/agent/run/{run_id}?after={after}") as res:
        return [
            json.loads(line.removeprefix("data: "))
            for line in res.iter_lines()
            if line.startswith("data: ")
        ]


def test_run_streams_and_replays_from_cursor(monkeypatch):
    monkeypatch.setattr(app_module, "stream_deepagent", fake_stream)

    run_id = client.post("/ask/agent/run", json={"question": "q", "thread_id": "t"}).json()[
        "run_id"
    ]

    assert read_events(run_id) == EVENTS  # first read follows the live run to the end
    assert read_events(run_id, after=1) == EVENTS[1:]  # reconnect replays from cursor
    assert client.get("/ask/agent/run/nope").status_code == 404
