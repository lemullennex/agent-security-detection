# AI Security Assurance

Security work for LLM systems tends to split into two separate conversations.

One is pre-deployment: can the system resist the attacks it is likely to face?
The other is post-deployment: if something goes wrong in production, can you tell from the logs what happened and why?

This repo is organized around both sides of that problem.

## Modules

### `agent_behavioral_detection/`
Behavioral detection for production agentic systems.

This module focuses on what compromise or misuse looks like once an agent is already running. It builds lightweight behavioral baselines and flags deviations such as unexpected tool usage, unusual retrieval volume, or access to data sources that are out of character for a given agent.

The goal is operational visibility. If an agent is manipulated through prompt injection, retrieval poisoning, or misuse of permissions, the detection layer should give you a concrete signal that something abnormal happened.

### `adversarial_eval/`
Adversarial evaluation for LLM systems and lightweight agents.

This module focuses on whether a system resists attack before deployment. It tests prompt-level injection resistance across five attack categories and agent-level behavioral failures where malicious content attempts to redirect the system toward unauthorized actions.

The current suite covers:
- direct injection
- roleplay framing
- indirect injection
- many-shot attacks
- token manipulation
- agent attack simulation against tool and data boundaries

The output is a structured report that can be used in CI to catch regressions after model changes, system prompt edits, or pipeline updates.

## Why both belong together

These modules solve related but different problems.

Adversarial evaluation asks whether the system holds up when you attack it on purpose.
Behavioral detection asks what compromise looks like if something still gets through.

One is pre-deployment assurance.
The other is production detection.

Together they reflect the same design principle: the real security boundary in an AI system is not the model by itself. It is what the system is allowed to do, what it can access, and whether you can observe its behavior well enough to investigate failures.

## How I think about the boundary

For LLM systems, the most important question is rarely whether the model can generate a bad sentence. The more important question is whether hostile content can push the system toward the wrong action.

That can mean:
- using a tool that was not in scope
- accessing data that was not requested
- taking an external action without a valid user intent
- shifting from retrieval or summarization into side effects like outbound email or privileged lookups

That is the thread connecting both modules in this repo.

## Recommended use

Use `adversarial_eval/` during development, before releases, and in CI whenever you update:
- model versions
- system prompts
- tool access rules
- retrieval pipelines
- agent policies

Use `agent_behavioral_detection/` in production to monitor live systems for behavior that departs from known-good baselines.

## Current state

This repo is intentionally lightweight. The point is not to build a giant platform. The point is to make the security logic easy to inspect, explain, and extend.

That keeps the signal high during reviews and makes it easier to adapt the tests and detections to a real product surface.
