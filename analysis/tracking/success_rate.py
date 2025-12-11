"""Success rate tracking for ranker evaluation.

Tracks whether the selected winner is within a threshold of the best arm's NDCG.
Per main.tex Section 5.2: Success is defined as selecting a winner
whose NDCG is within 1% of the true best arm's NDCG.
"""

from typing import Dict, List, Any, Optional, Tuple


class SuccessRateTracker:
    """Track whether final winner is within threshold of best arm.

    Per main.tex Section 5.2: Success is defined as selecting a winner
    whose NDCG is within 1% of the true best arm's NDCG.
    """

    def __init__(self, threshold_pct: float = 0.01):
        """Initialize success rate tracker.

        Args:
            threshold_pct: Maximum relative NDCG gap for success (default 1%).
        """
        self.threshold = threshold_pct
        self.arm_ndcg: Dict[str, float] = {}
        self.evaluation_records: List[Dict[str, Any]] = []

    def set_all_arm_ndcg(self, arm_ndcg_dict: Dict[str, float]) -> None:
        """Record offline NDCG for all arms at once.

        Args:
            arm_ndcg_dict: Dict of arm_name -> NDCG score.
        """
        self.arm_ndcg = arm_ndcg_dict.copy()

    def compute_success(self, winner_name: str) -> Tuple[bool, float, bool]:
        """Check if winner's NDCG is within threshold of best.

        Args:
            winner_name: Name of the selected winner arm.

        Returns:
            Tuple of (is_success, ndcg_gap_pct, has_valid_data) where:
                - is_success: True if winner is within threshold of best.
                - ndcg_gap_pct: Relative gap (best - winner) / best.
                - has_valid_data: True if we have meaningful NDCG data.
                  False when best_ndcg is 0 (no relevant docs) or no data.
        """
        if not self.arm_ndcg or winner_name not in self.arm_ndcg:
            return False, 1.0, False

        best_ndcg = max(self.arm_ndcg.values())
        winner_ndcg = self.arm_ndcg[winner_name]

        if best_ndcg == 0:
            # All arms have zero NDCG - no valid comparison possible
            # Report as "trivially successful" but flag as invalid data
            return True, 0.0, False

        ndcg_gap = (best_ndcg - winner_ndcg) / best_ndcg
        is_success = ndcg_gap <= self.threshold

        return is_success, ndcg_gap, True

    def record_evaluation(
        self,
        winner_name: str,
        round_num: int,
        is_final: bool = False
    ) -> Dict[str, Any]:
        """Record an evaluation result.

        Args:
            winner_name: Name of the selected winner.
            round_num: Current round number.
            is_final: Whether this is the final evaluation.

        Returns:
            Dict with success metrics including has_valid_data flag.
        """
        is_success, gap, has_valid_data = self.compute_success(winner_name)
        result = {
            "round": round_num,
            "winner": winner_name,
            "is_success": is_success,
            "ndcg_gap_pct": gap,
            "has_valid_data": has_valid_data,
            "is_final": is_final,
        }
        self.evaluation_records.append(result)
        return result

    def get_sample_complexity(self, success_threshold: float = 0.95) -> Optional[int]:
        """Find first round where success rate reaches threshold.

        Per main.tex: Sample complexity is the number of queries needed
        to achieve 95% success rate (within 1% of best).

        Args:
            success_threshold: Required success rate (default 0.95).

        Returns:
            Round number where success_threshold is first achieved, or None.
        """
        if not self.evaluation_records:
            return None

        successes = 0
        for i, record in enumerate(self.evaluation_records):
            if record["is_success"]:
                successes += 1
            success_rate = successes / (i + 1)
            if success_rate >= success_threshold:
                return record["round"]
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dict with evaluation counts, success rate, and sample complexity.
        """
        if not self.evaluation_records:
            return {"n_evaluations": 0}

        successes = sum(1 for r in self.evaluation_records if r["is_success"])
        final_records = [r for r in self.evaluation_records if r["is_final"]]

        return {
            "n_evaluations": len(self.evaluation_records),
            "success_rate": successes / len(self.evaluation_records),
            "sample_complexity_95": self.get_sample_complexity(0.95),
            "final_success": final_records[-1]["is_success"] if final_records else None,
            "final_gap_pct": final_records[-1]["ndcg_gap_pct"] if final_records else None,
        }
