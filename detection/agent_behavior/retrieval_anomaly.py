"""
retrieval_anomaly.py

Detects retrieval expansion anomalies in agent execution traces.

Signal: an agent retrieving significantly more documents than expected
for its established baseline for that task type.

Retrieval expansion is a common indicator of indirect prompt injection,
where injected content in a retrieved document causes the agent to
broaden its data access scope beyond what the task requires.

Baselines are computed per agent_id and per task_type.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class RetrievalAnomaly:
    agent_id: str
    task_type: str
    documents_retrieved: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    flag: str = "retrieval_expansion"
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


def compute_baselines(logs: list[dict]) -> dict:
    """
    Compute baseline retrieval counts per agent_id and task_type.

    Returns a nested dict:
        baselines[agent_id][task_type] = list of document counts
    """
    counts = defaultdict(lambda: defaultdict(list))

    for event in logs:
        agent_id = event.get("agent_id")
        task_type = event.get("task_type")
        docs = event.get("documents_retrieved")

        if all([agent_id, task_type, docs is not None]):
            counts[agent_id][task_type].append(docs)

    return counts


def detect_retrieval_anomalies(
    logs: list[dict],
    baselines: dict,
    std_multiplier: float = 2.0,
    min_baseline_samples: int = 3,
) -> list[RetrievalAnomaly]:
    """
    Detect retrieval expansion against established baselines.

    Args:
        logs: list of agent execution trace events
        baselines: baseline counts from compute_baselines()
        std_multiplier: how many standard deviations above mean to flag
        min_baseline_samples: minimum events required to compute a baseline

    Returns:
        list of RetrievalAnomaly instances for flagged events

    Design note:
        These signals are intentionally high-recall. In production they would
        be combined with contextual signals like query complexity and user role
        to reduce false positives before alerting.
    """
    anomalies = []

    for event in logs:
        agent_id = event.get("agent_id")
        task_type = event.get("task_type")
        docs = event.get("documents_retrieved")

        if not all([agent_id, task_type, docs is not None]):
            continue

        # Compare against historical retrieval behavior for this agent + task type.
        # This avoids false positives where high retrieval volume is expected
        # for certain workflows like research vs structured extraction.
        history = (
            baselines
            .get(agent_id, {})
            .get(task_type, [])
        )

        if len(history) < min_baseline_samples:
            continue

        mean = statistics.mean(history)
        std = statistics.stdev(history)

        if std == 0:
            continue

        z_score = (docs - mean) / std

        # simple z-score instead of an ML model — easier to explain and debug during incidents
        if z_score > std_multiplier:
            anomalies.append(RetrievalAnomaly(
                agent_id=agent_id,
                task_type=task_type,
                documents_retrieved=docs,
                baseline_mean=round(mean, 2),
                baseline_std=round(std, 2),
                z_score=round(z_score, 2),
                session_id=event.get("session_id"),
                timestamp=event.get("timestamp"),
            ))

    return anomalies


def load_logs(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def print_report(anomalies: list[RetrievalAnomaly]) -> None:
    if not anomalies:
        print("No retrieval anomalies detected.")
        return

    print(f"\nRetrieval Anomalies Detected: {len(anomalies)}\n")
    print("-" * 60)

    for a in anomalies:
        print(f"Flag:                {a.flag}")
        print(f"Agent ID:            {a.agent_id}")
        print(f"Task type:           {a.task_type}")
        print(f"Documents retrieved: {a.documents_retrieved}  (baseline mean: {a.baseline_mean}, std: {a.baseline_std})")
        print(f"Z-score:             {a.z_score}")
        if a.session_id:
            print(f"Session:             {a.session_id}")
        if a.timestamp:
            print(f"Timestamp:           {a.timestamp}")
        print("-" * 60)


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_agent_logs.json"

    logs = load_logs(log_path)
    baselines = compute_baselines(logs)
    anomalies = detect_retrieval_anomalies(logs, baselines)
    print_report(anomalies)
