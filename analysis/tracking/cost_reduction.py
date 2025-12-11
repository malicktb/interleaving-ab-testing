"""Cost reduction calculation for hierarchical evaluation.

Calculates inference cost reduction from hierarchical evaluation.
Per main.tex: Cost reduction = Flat MDB inferences / H-MDB inferences
"""

from typing import Dict, Any


class CostReductionCalculator:
    """Calculate inference cost reduction from hierarchical evaluation.

    Per main.tex: Cost reduction = Flat MDB inferences / H-MDB inferences

    In flat MDB, all K arms are evaluated every round.
    In H-MDB, only cluster representatives are evaluated in Level 1,
    then only winning cluster members in Level 2.
    """

    def __init__(self, total_arms: int, level1_arms: int, level2_arms: int):
        """Initialize cost reduction calculator.

        Args:
            total_arms: Total number of arms in pool (K).
            level1_arms: Number of arms in Level 1 (representatives).
            level2_arms: Number of arms in Level 2 (winning cluster members).
        """
        self.total_arms = total_arms
        self.level1_arms = level1_arms
        self.level2_arms = level2_arms

        self.level1_rounds = 0
        self.level2_rounds = 0
        self.flat_rounds = 0

    def record_round(self, level: int, n_arms_evaluated: int) -> None:
        """Record a round of evaluation.

        Args:
            level: Current level (1 or 2), or 0 for flat MDB.
            n_arms_evaluated: Number of arms evaluated this round.
        """
        if level == 1:
            self.level1_rounds += 1
        elif level == 2:
            self.level2_rounds += 1
        else:
            self.flat_rounds += 1

    def compute_hmdb_inferences(self) -> int:
        """Compute total inferences for H-MDB run.

        Returns:
            Total inference count for hierarchical evaluation.
        """
        return (self.level1_arms * self.level1_rounds +
                self.level2_arms * self.level2_rounds)

    def compute_flat_inferences(self, total_rounds: int) -> int:
        """Compute hypothetical inferences for flat MDB run.

        Args:
            total_rounds: Total rounds that would be run in flat MDB.

        Returns:
            Total inferences for flat MDB (K * total_rounds).
        """
        return self.total_arms * total_rounds

    def get_cost_reduction_ratio(self) -> float:
        """Compute cost reduction ratio: Flat / H-MDB.

        Higher values mean more efficient (H-MDB saved more computation).

        Returns:
            Cost reduction ratio (>1 means H-MDB is more efficient).
        """
        total_rounds = self.level1_rounds + self.level2_rounds
        if total_rounds == 0:
            return 1.0

        flat_inferences = self.compute_flat_inferences(total_rounds)
        hmdb_inferences = self.compute_hmdb_inferences()

        if hmdb_inferences == 0:
            return float('inf')

        return flat_inferences / hmdb_inferences

    def get_summary(self) -> Dict[str, Any]:
        """Get cost reduction summary.

        Returns:
            Dict with arm counts, round counts, and cost metrics.
        """
        total_rounds = self.level1_rounds + self.level2_rounds
        return {
            "total_arms": self.total_arms,
            "level1_arms": self.level1_arms,
            "level2_arms": self.level2_arms,
            "level1_rounds": self.level1_rounds,
            "level2_rounds": self.level2_rounds,
            "hmdb_inferences": self.compute_hmdb_inferences(),
            "flat_inferences_equivalent": self.compute_flat_inferences(total_rounds),
            "cost_reduction_ratio": self.get_cost_reduction_ratio(),
        }
