"""
sequence_anomaly.py

Detects new data source access in agent execution traces.

Signal: an agent accessing a data source that is not in its established
known source list for that agent ID.

In production systems, agents should have well-defined retrieval
boundaries. Access to an unknown data source indicates configuration
drift, a misconfigured retrieval pipeline, or an active attempt to
expand the agent's data access scope — a common pattern in prompt
injection attacks that attempt lateral movement across data domains.

Known sources are derived from the historical log baseline.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SequenceAnomaly:
    agent_id: str
    task_type: str
    data_source: str
    known_sources: list
    flag: str = "new_data_access"
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


def build_known_sources(logs: list[dict]) -> dict:
    """
    Build the set of known data sources per agent_id from historical logs.

    In production this would be derived from a longer baseline window
    and possibly from an explicit allowlist defined at deploy time.

    Returns:
        known_sources[agent_id] = set of data source strings
    """
    known = defaultdict(set)

    for event in logs:
        agent_id = event.get("agent_id")
        data_source = event.get("data_source")

        if agent_id and data_source:
            known[agent_id].add(data_source)

    return known


def detect_sequence_anomalies(
    logs: list[dict],
    known_sources: dict,
) -> list[SequenceAnomaly]:
    """
    Detect access to data sources outside the agent's known baseline.

    Args:
        logs: list of agent execution trace events
        known_sources: known source sets from build_known_sources()

    Returns:
        list of SequenceAnomaly instances for flagged events

    Design note:
        This signal is binary rather than statistical. Either a data source
        is in the known set or it is not. In production this would be
        combined with allowlist enforcement at the infrastructure layer.
        Detection here provides an observability layer above access controls.
    """
    anomalies = []
    # intentionally high-recall — downstream filtering should handle false positives
    already_flagged = set()

    for event in logs:
        agent_id = event.get("agent_id")
        task_type = event.get("task_type")
        data_source = event.get("data_source")

        if not all([agent_id, task_type, data_source]):
            continue

        agent_known = known_sources.get(agent_id, set())

        if data_source not in agent_known:
            key = (agent_id, data_source)
            if key not in already_flagged:
                already_flagged.add(key)
                anomalies.append(SequenceAnomaly(
                    agent_id=agent_id,
                    task_type=task_type,
                    data_source=data_source,
                    known_sources=sorted(agent_known),
                    session_id=event.get("session_id"),
                    timestamp=event.get("timestamp"),
                ))

    return anomalies


def load_logs(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def print_report(anomalies: list[SequenceAnomaly]) -> None:
    if not anomalies:
        print("No new data source access detected.")
        return

    print(f"\nNew Data Source Access Detected: {len(anomalies)}\n")
    print("-" * 60)

    for a in anomalies:
        print(f"Flag:          {a.flag}")
        print(f"Agent ID:      {a.agent_id}")
        print(f"Task type:     {a.task_type}")
        print(f"Data source:   {a.data_source}")
        print(f"Known sources: {a.known_sources}")
        if a.session_id:
            print(f"Session:       {a.session_id}")
        if a.timestamp:
            print(f"Timestamp:     {a.timestamp}")
        print("-" * 60)


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_agent_logs.json"

    logs = load_logs(log_path)
    known_sources = build_known_sources(logs)
    anomalies = detect_sequence_anomalies(logs, known_sources)
    print_report(anomalies)
