"""
role_anomaly.py

Detects IAM role assumption anomalies from CloudTrail-style logs.

Signal: an IAM role assumed from a source IP not previously seen for
that principal, or a role not in the principal's known role set.

IAM signals are secondary in agentic systems. They are useful but
typically lagging indicators compared to behavioral anomalies. By the
time an unusual role assumption appears in CloudTrail, behavioral
signals in agent execution traces have usually already fired.

This module is designed to complement agent behavioral detection,
not replace it.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class IAMAnomaly:
    principal: str
    role: str
    source_ip: str
    reason: str
    flag: str = "iam_anomaly"
    event_time: Optional[str] = None
    session_id: Optional[str] = None


def build_iam_baselines(events: list[dict]) -> dict:
    """
    Build known roles and source IPs per principal from historical CloudTrail events.

    Returns:
        baselines[principal] = {
            "roles": set of known role ARNs,
            "ips": set of known source IPs
        }
    """
    baselines = defaultdict(lambda: {"roles": set(), "ips": set()})

    for event in events:
        principal = event.get("principal")
        role = event.get("role")
        source_ip = event.get("source_ip")

        if principal and role:
            baselines[principal]["roles"].add(role)
        if principal and source_ip:
            baselines[principal]["ips"].add(source_ip)

    return baselines


def detect_iam_anomalies(
    events: list[dict],
    baselines: dict,
) -> list[IAMAnomaly]:
    """
    Detect unusual IAM role assumption patterns.

    Flags when:
        - A principal assumes a role not in their known role set
        - A principal assumes a role from a source IP not in their known IP set

    Args:
        events: list of CloudTrail-style IAM events
        baselines: baseline sets from build_iam_baselines()

    Returns:
        list of IAMAnomaly instances for flagged events

    Design note:
        In production this would integrate with CloudTrail directly and
        correlate with GuardDuty findings. This implementation provides
        the detection logic layer independently of the ingestion mechanism.
    """
    anomalies = []

    for event in events:
        principal = event.get("principal")
        role = event.get("role")
        source_ip = event.get("source_ip")

        if not all([principal, role, source_ip]):
            continue

        principal_baseline = baselines.get(principal, {"roles": set(), "ips": set()})
        known_roles = principal_baseline["roles"]
        known_ips = principal_baseline["ips"]

        reasons = []

        if role not in known_roles:
            reasons.append(f"unknown role: {role}")

        if source_ip not in known_ips:
            reasons.append(f"unknown source IP: {source_ip}")

        if reasons:
            anomalies.append(IAMAnomaly(
                principal=principal,
                role=role,
                source_ip=source_ip,
                reason=", ".join(reasons),
                event_time=event.get("event_time"),
                session_id=event.get("session_id"),
            ))

    return anomalies


def load_events(filepath: str) -> list[dict]:
    with open(filepath) as f:
        return json.load(f)


def print_report(anomalies: list[IAMAnomaly]) -> None:
    if not anomalies:
        print("No IAM anomalies detected.")
        return

    print(f"\nIAM Anomalies Detected: {len(anomalies)}\n")
    print("-" * 60)

    for a in anomalies:
        print(f"Flag:       {a.flag}")
        print(f"Principal:  {a.principal}")
        print(f"Role:       {a.role}")
        print(f"Source IP:  {a.source_ip}")
        print(f"Reason:     {a.reason}")
        if a.event_time:
            print(f"Time:       {a.event_time}")
        if a.session_id:
            print(f"Session:    {a.session_id}")
        print("-" * 60)


if __name__ == "__main__":
    import sys

    log_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample_iam_logs.json"

    events = load_events(log_path)
    baselines = build_iam_baselines(events)
    anomalies = detect_iam_anomalies(events, baselines)
    print_report(anomalies)
