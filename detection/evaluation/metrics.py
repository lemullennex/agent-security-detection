"""
metrics.py

Evaluation metrics for detection quality.

Provides utilities for measuring precision, recall, and F1 across
detection signals. Designed for offline evaluation against labeled
datasets, not real-time production use.

In production, detection quality would be measured by:
    - False positive rate (alerts that did not indicate real issues)
    - False negative rate (real issues that were not flagged)
    - Mean time to detection for known attack patterns
    - Alert fatigue ratio (how many alerts require human review)
"""

from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        """
        Fraction of flagged events that were actual anomalies.
        Low precision means high alert fatigue.
        """
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def recall(self) -> float:
        """
        Fraction of actual anomalies that were flagged.
        Low recall means real threats are being missed.
        """
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def f1(self) -> float:
        """
        Harmonic mean of precision and recall.
        Useful for comparing signal quality across detection modules.
        """
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def summary(self) -> str:
        return (
            f"Precision: {self.precision:.2f}  "
            f"Recall: {self.recall:.2f}  "
            f"F1: {self.f1:.2f}  "
            f"(TP: {self.true_positives}, FP: {self.false_positives}, FN: {self.false_negatives})"
        )


def evaluate(flagged_ids: set, actual_anomaly_ids: set) -> DetectionMetrics:
    """
    Evaluate detection quality against a labeled set of known anomalies.

    Args:
        flagged_ids: set of session IDs flagged by detection logic
        actual_anomaly_ids: set of session IDs that are known anomalies

    Returns:
        DetectionMetrics instance
    """
    tp = len(flagged_ids & actual_anomaly_ids)
    fp = len(flagged_ids - actual_anomaly_ids)
    fn = len(actual_anomaly_ids - flagged_ids)

    return DetectionMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


if __name__ == "__main__":
    # Example evaluation against the sample dataset.
    # sess_004 (agent_1 retrieval + tool spike) and sess_007 (agent_2 parser spike)
    # are the known anomalies in sample_agent_logs.json.

    flagged = {"sess_004", "sess_007"}
    actual = {"sess_004", "sess_007"}

    metrics = evaluate(flagged, actual)
    print("Sample dataset evaluation:")
    print(metrics.summary())
