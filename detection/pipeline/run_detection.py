"""
run_detection.py

Unified detection pipeline for agent security signals.

Runs all detectors against a log file and correlates results by session_id,
producing a single report grouped by session with severity scoring and
likely cause inference.

This is the layer that turns individual signals into an attack narrative —
the difference between "this tool call spiked" and "this session shows
coordinated retrieval expansion consistent with prompt injection."

Usage:
    python detection/pipeline/run_detection.py
    python detection/pipeline/run_detection.py data/sample_agent_logs.json
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Import detection modules
sys.path.insert(0, ".")
from detection.agent_behavior.tool_anomaly import compute_baselines as tool_baselines
from detection.agent_behavior.tool_anomaly import detect_tool_anomalies
from detection.agent_behavior.retrieval_anomaly import compute_baselines as retrieval_baselines
from detection.agent_behavior.retrieval_anomaly import detect_retrieval_anomalies
from detection.agent_behavior.sequence_anomaly import build_known_sources
from detection.agent_behavior.sequence_anomaly import detect_sequence_anomalies


@dataclass
class SessionReport:
    session_id: str
    agent_id: str
    task_type: str
    timestamp: Optional[str]
    signals: list[str] = field(default_factory=list)
    z_scores: dict = field(default_factory=dict)
    severity: str = "LOW"
    likely_cause: str = "unknown"


def score_severity(signals: list[str], z_scores: dict) -> str:
    """
    Score severity based on number of signals and their z-scores.

    Multiple correlated signals at high z-scores indicate a coordinated
    pattern rather than random noise — that distinction matters for triage.
    """
    if len(signals) >= 2:
        max_z = max(z_scores.values()) if z_scores else 0
        if max_z > 3.0:
            return "HIGH"
        return "MEDIUM"
    if len(signals) == 1:
        max_z = max(z_scores.values()) if z_scores else 0
        if max_z > 4.0:
            return "MEDIUM"
        return "LOW"
    return "LOW"


def infer_cause(signals: list[str]) -> str:
    """
    Infer the most likely cause based on the combination of signals observed.

    This is intentionally heuristic — in production you would layer in
    additional context like user intent, query content, and prior session history.
    These are pattern-based starting points for investigation, not conclusions.
    """
    has_tool = "anomalous_tool_usage" in signals
    has_retrieval = "retrieval_expansion" in signals
    has_new_source = "new_data_access" in signals

    if has_tool and has_retrieval:
        return "prompt injection expanding retrieval scope"
    if has_tool and has_new_source:
        return "possible lateral movement across data domains"
    if has_retrieval and has_new_source:
        return "retrieval pipeline misconfiguration or data exfiltration attempt"
    if has_tool:
        return "tool usage spike — possible instruction hijacking"
    if has_retrieval:
        return "retrieval expansion — review query content and context"
    if has_new_source:
        return "new data source access — verify agent retrieval boundaries"

    return "unknown"


def run_pipeline(log_path: str) -> list[SessionReport]:
    """
    Run all agent behavioral detectors and correlate results by session.

    Returns a list of SessionReport instances for sessions with at least
    one flagged signal, sorted by severity.
    """
    with open(log_path) as f:
        logs = json.load(f)

    # Run each detector
    t_baselines = tool_baselines(logs)
    tool_anomalies = detect_tool_anomalies(logs, t_baselines)

    r_baselines = retrieval_baselines(logs)
    retrieval_anomalies = detect_retrieval_anomalies(logs, r_baselines)

    known_sources = build_known_sources(logs)
    sequence_anomalies = detect_sequence_anomalies(logs, known_sources)

    # Index sessions for correlation
    session_meta = {}
    for event in logs:
        sid = event.get("session_id")
        if sid and sid not in session_meta:
            session_meta[sid] = {
                "agent_id": event.get("agent_id", "unknown"),
                "task_type": event.get("task_type", "unknown"),
                "timestamp": event.get("timestamp"),
            }

    # Collect signals per session
    session_signals: dict = defaultdict(list)
    session_z_scores: dict = defaultdict(dict)

    for a in tool_anomalies:
        if a.session_id:
            session_signals[a.session_id].append(a.flag)
            session_z_scores[a.session_id]["tool_z"] = a.z_score

    for a in retrieval_anomalies:
        if a.session_id:
            session_signals[a.session_id].append(a.flag)
            session_z_scores[a.session_id]["retrieval_z"] = a.z_score

    for a in sequence_anomalies:
        if a.session_id:
            session_signals[a.session_id].append(a.flag)

    # Build session reports
    reports = []
    for session_id, signals in session_signals.items():
        meta = session_meta.get(session_id, {})
        z_scores = session_z_scores.get(session_id, {})
        severity = score_severity(signals, z_scores)
        cause = infer_cause(signals)

        reports.append(SessionReport(
            session_id=session_id,
            agent_id=meta.get("agent_id", "unknown"),
            task_type=meta.get("task_type", "unknown"),
            timestamp=meta.get("timestamp"),
            signals=signals,
            z_scores=z_scores,
            severity=severity,
            likely_cause=cause,
        ))

    # Sort by severity — HIGH first
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    reports.sort(key=lambda r: severity_order.get(r.severity, 3))

    return reports


def print_pipeline_report(reports: list[SessionReport]) -> None:
    if not reports:
        print("No anomalies detected across any session.")
        return

    print(f"\n{'=' * 60}")
    print(f"UNIFIED DETECTION REPORT")
    print(f"Sessions with anomalies: {len(reports)}")
    print(f"{'=' * 60}\n")

    for r in reports:
        severity_label = {
            "HIGH": "[ HIGH   ]",
            "MEDIUM": "[ MEDIUM ]",
            "LOW": "[ LOW    ]",
        }.get(r.severity, "[ UNKNOWN ]")

        print(f"{severity_label}  Session: {r.session_id}")
        print(f"  Agent:        {r.agent_id}  ({r.task_type})")
        if r.timestamp:
            print(f"  Timestamp:    {r.timestamp}")
        print(f"  Signals:")
        for signal in r.signals:
            z = ""
            if signal == "anomalous_tool_usage" and "tool_z" in r.z_scores:
                z = f"  (z-score: {r.z_scores['tool_z']:.2f})"
            elif signal == "retrieval_expansion" and "retrieval_z" in r.z_scores:
                z = f"  (z-score: {r.z_scores['retrieval_z']:.2f})"
            print(f"    - {signal}{z}")
        print(f"  Likely cause: {r.likely_cause}")
        print()


if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_agent_logs.json"
    reports = run_pipeline(log_path)
    print_pipeline_report(reports)
