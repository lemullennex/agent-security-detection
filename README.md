# Detecting Unsafe Agent Behavior in Production GenAI Systems

Most detection systems focus on infrastructure signals like IAM activity and API calls. In agentic systems, the primary risk surface is behavioral: how the model selects tools, accesses data, and evolves decisions across a workflow.

This repository implements detection logic for agent behavior in production-style GenAI systems, with supporting IAM anomaly detection as a secondary signal. The goal is not complexity. It is detection logic that is simple, explainable, and aligned to how these systems actually fail.

## System context

The detection logic assumes an agentic system with tool calling, retrieval over internal data sources, and multi-step execution traces per request. Sample logs simulate agent execution traces including: agent_id, tool_name, documents_retrieved, data_source, timestamp, and task_type.

## Detection approach

Detection is structured in two layers. Agent behavioral detection is primary. IAM detection is secondary and usually a lagging indicator compared to behavioral signals.

### Agent behavioral detection

**Tool usage anomaly** — detects unexpected spikes in tool usage relative to the agent's established baseline for that task type. Baselines are computed per agent ID and per task type. A call count that is normal for a research workflow would flag for a structured extraction task.

```python
if tool_call_count > baseline_mean + 2 * std_dev:
    flag = "anomalous_tool_usage"
```

This is often an early indicator of prompt injection, where an injected instruction causes the agent to repeatedly expand its actions beyond the expected workflow.

In one observed pattern, an agent that normally called a retrieval tool once per request began calling it five to six times in sequence. Nothing failed at the infrastructure level. Responses appeared correct. But the behavior had changed. This indicated a prompt injection attempt causing the agent to expand its search scope. Without behavioral detection, it would not have been visible.

**Retrieval expansion** — detects when an agent retrieves significantly more documents than expected for the query type and task context.

```python
if documents_retrieved > expected_range:
    flag = "retrieval_expansion"
```

This can indicate prompt injection attempts that expand search scope, misaligned query planning, or data exposure risk through over-retrieval.

**New data source access** — detects when an agent accesses a data source not in the known source list for that agent ID. In production, agents should have well-defined retrieval boundaries. Access to a new source indicates configuration drift or an active attempt to expand data access.

```python
if data_source not in known_sources[agent_id]:
    flag = "new_data_access"
```

### IAM anomaly detection

Detects unusual IAM role usage patterns including role assumption from a new source IP, after-hours assumptions, and role chains inconsistent with defined trust relationships.

```python
if role not in known_roles[user] or source_ip not in known_ips[user]:
    flag = "iam_anomaly"
```

The Sigma rule implementation is in `sigma_rules/iam_new_source.yml`.

## Unified detection pipeline

`detection/pipeline/run_detection.py` runs all signals together and correlates results by session, producing a unified report with severity scoring and likely cause inference.

```
Session: sess_004
  Signals:
    - anomalous_tool_usage (z-score: 4.20)
    - retrieval_expansion (z-score: 3.84)
  Likely cause: prompt injection expanding retrieval scope
  Severity: HIGH
```

## Design principles

This repository intentionally uses simple statistical thresholds rather than complex models. These signals are high-recall by design. In production they would be combined with contextual signals like task type, user role, and session history to reduce false positives before alerting. Detection logic should be explainable enough to walk through line by line in an incident review.

## Repository structure

```
agent-security-detection/
  data/
    sample_agent_logs.json
    sample_iam_logs.json
  detection/
    agent_behavior/
      tool_anomaly.py
      retrieval_anomaly.py
      sequence_anomaly.py
    iam/
      role_anomaly.py
    evaluation/
      metrics.py
    pipeline/
      run_detection.py
  sigma_rules/
    iam_new_source.yml
```

## How to run

```bash
# Run individual detectors
python detection/agent_behavior/tool_anomaly.py
python detection/agent_behavior/retrieval_anomaly.py
python detection/agent_behavior/sequence_anomaly.py
python detection/iam/role_anomaly.py data/sample_iam_logs.json
python detection/evaluation/metrics.py

# Run unified pipeline (recommended)
python detection/pipeline/run_detection.py
```

## What to expect

- `tool_anomaly.py` flags sess_004 and sess_007
- `retrieval_anomaly.py` flags sess_004
- `sequence_anomaly.py` prints no anomalies (expected — all agents access known sources)
- `role_anomaly.py` flags iam_sess_004 on both unknown role and unknown source IP
- `run_detection.py` correlates sess_004 signals and scores severity

## What this demonstrates

How agentic systems fail in production. Why behavioral detection is required beyond IAM monitoring. How to design simple, explainable detection signals. How to correlate multiple signals into an attack narrative. How to reason about false positives, thresholds, and production tradeoffs.

## Background

This project grew out of work designing and evaluating security architectures for production agentic AI systems across enterprise deployments. The patterns here reflect failure modes observed directly: agents that expand their retrieval scope under injected instructions, tool usage that indicates workflow hijacking, and IAM assumptions that standard monitoring treats as normal because the credentials are valid.

## Author

Lauren Mullennex
Senior GenAI Solutions Architect, AWS
Focused on agentic AI security, detection engineering, and production GenAI systems
