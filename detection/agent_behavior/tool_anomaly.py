"""
tool_anomaly.py

Detects tool usage anomalies in agent execution traces.

Signal: an agent calling a tool at a frequency that deviates significantly
from its established baseline for that task type.

This is often an early indicator of prompt injection behavior, where an
injected instruction causes the agent to expand its tool usage beyond the
expected workflow pattern.

Baselines are computed per agent_id and per task_type. A call count that
is normal for a research workflow would flag for a structured extraction task.
"""

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolAnomaly:
    agent_id: str
    task_type: str
    tool_name: str
    call_count: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    flag: str = "anomalous_tool_usage"
    session_id: Optional[str] = None
    timestamp: Optional[str] = None


def compute_baselines(logs: list[dict]) -> dict:
    """
    Compute baseline tool call counts per agent_id and task_type.

    Returns a nested dict:
        baselines[agent_id][task_type][tool_name] = list of call counts
    """
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for event in logs:
        agent_id = event.get("agent_id")
        task_type = event.get("task_type")
        tool_name = event.get("tool_name")
        call_count = event.get("tool_call_count")

        if all([agent_id, task_type, tool_name, call_count is not None]):
            counts[agent_id][task_type][tool_name].append(call_count)

    return counts


def detect_tool_anomalies(
    logs: list[dict],
    baselines: dict,
    std_multiplier: float = 2.0,
    min_baseline_samples: int = 3,
) -> list[ToolAnomaly]:
    """
    Detect tool usage anomalies against established baselines.

    Args:
        logs: list of agent execution trace events
        baselines: baseline counts from compute_baselines()
        std_multiplier: how many standard deviations above mean to flag
        min_baseline_samples: minimum events required to compute a baseline

    Returns:
        list of ToolAnomaly instances for flagged events

    Design note:
        These signals are intentionally high-recall. In production they would
        be combined with contextual signals like user role and session history
        to reduce false positives before alerting.

    # if this fires frequently, your agent planning logic is probably off, not just security
    """
    anomalies = []

    for event in logs:
        agent_id = event.get("agent_id")
        task_type = event.get("task_type")
        tool_name = event.get("tool_name")
        call_count = event.get("tool_call_count")

        if not all([agent_id, task_type, tool_name, call_count is not None]):
            continue

        # Compare against historical behavior for this specific agent + task type.
        # This avoids false positives where high tool usage is expected for certain workflows.
        history = (
            baselines
            .get(agent_id, {})
            .get(task_type, {})
            .get(tool_name, [])
        )

        if len(history) < min_baseline_samples:
            continue

        mean = statistics.mean(history)
        std = statistics.stdev(history)

        # Avoid division by zero when all historical values are identical
        if std == 0:
            continue

        z_score = (call_count - mean) / std

        # this threshold is somewhat arbitrary — tune per agent + task type in production
        if z_score > std_multiplier:
            anomalies.append(ToolAnomaly(
                agent_id=agent_id,
                task_type=task_type,
                tool_name=tool_name,
                call_count=call_count,
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


def print_report(anomalies: list[ToolAnomaly]) -> None:
    if not anomalies:
        print("No tool usage anomalies detected.")
        return

    print(f"\nTool Usage Anomalies Detected: {len(anomalies)}\n")
    print("-" * 60)

    for a in anomalies:
        print(f"Flag:        {a.flag}")
        print(f"Agent ID:    {a.agent_id}")
        print(f"Task type:   {a.task_type}")
        print(f"Tool:        {a.tool_name}")
        print(f"Call count:  {a.call_count}  (baseline mean: {a.baseline_mean}, std: {a.baseline_std})")
        print(f"Z-score:     {a.z_score}")
        if a.session_id:
            print(f"Session:     {a.session_id}")
        if a.timestamp:
            print(f"Timestamp:   {a.timestamp}")
        print("-" * 60)


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_agent_logs.json"

    logs = load_logs(log_path)
    baselines = compute_baselines(logs)
    anomalies = detect_tool_anomalies(logs, baselines)
    print_report(anomalies)
