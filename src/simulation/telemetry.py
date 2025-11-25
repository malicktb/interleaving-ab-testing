"""Experiment Log (Telemetry).

This module records the history of the experiment. Tracks every decision the Bandit made, who won, and whether
the Bandit made a mistake (Regret).
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class RegretTelemetry:
    """The Experiment Scoreboard.

    Tracks the performance of the Bandit Strategy over time.

    Attributes:
        arm_names: A list of all available models (so we can report 0% for unused ones).
        regret_history: Log of "Mistakes" (1 = We failed to pick the best model, 0 = We picked it).
        click_history: Log of "Yield" (1 = User clicked something, 0 = User ignored the list).
        selection_history: Log of which models were allowed to play in each round.
        winner_history: Log of which model got the credit (if any).
        best_history: Log of which model was *actually* the best for that specific user.
    """

    arm_names: List[str] = field(default_factory=list)
    regret_history: List[float] = field(default_factory=list)
    click_history: List[int] = field(default_factory=list)
    selection_history: List[Set[str]] = field(default_factory=list)
    winner_history: List[Optional[str]] = field(default_factory=list)
    best_history: List[Optional[str]] = field(default_factory=list)

    def record_iteration(
        self,
        selected_arms: Set[str],
        winner: Optional[str],
        best_arm: Optional[str],
    ) -> float:
        """Log the results of a single round.

        Args:
            selected_arms: The set of models the Strategy decided to test.
            winner: The model that actually got a click (or None).
            best_arm: The model that ranked the clicked item highest (The "Right Answer").

        Returns:
            The regret score for this round (0 or 1).
        """
        self.selection_history.append(selected_arms)
        self.winner_history.append(winner)
        self.best_history.append(best_arm)

        # Track if the system generated value (a click)
        self.click_history.append(1 if winner is not None else 0)

        # Calculate "Regret" (Did the Bandit make a mistake?)
        # Rule: If the best model was included in the slate, the Bandit succeeded.
        # Even if the user didn't click (bad luck), the Bandit made the right choice.
        if best_arm is None:
            regret = 0.0  # No good answer existed, so no mistake was possible.
        elif best_arm in selected_arms:
            regret = 0.0  # Success: We picked the winner.
        else:
            regret = 1.0  # Failure: We left the best model on the bench.

        self.regret_history.append(regret)
        return regret

    @property
    def cumulative_regret(self) -> np.ndarray:
        """Running total of mistakes made over time."""
        return np.cumsum(self.regret_history)

    @property
    def cumulative_clicks(self) -> np.ndarray:
        """Running total of clicks received over time."""
        return np.cumsum(self.click_history)

    @property
    def total_regret(self) -> float:
        """Total number of mistakes."""
        return sum(self.regret_history)

    @property
    def num_iterations(self) -> int:
        """Total number of rounds played."""
        return len(self.regret_history)

    def get_arm_selection_rates(self, window: Optional[int] = None) -> Dict[str, float]:
        """Check how often each model is getting picked.

        Args:
            window: If set (e.g., 1000), only look at the last N rounds.
                    Useful to see if the Bandit has "settled" on a winner.
        """
        if not self.selection_history:
            return {name: 0.0 for name in self.arm_names}

        # Look at full history or just the recent window
        history = self.selection_history
        if window is not None and window > 0:
            history = history[-window:]

        # Count how many times each arm appears in the logs
        all_selections = [name for s in history for name in s]
        counts = Counter(all_selections)
        total_rounds = len(history)

        # Calculate percentage (0.0 to 1.0)
        return {name: counts.get(name, 0) / total_rounds for name in self.arm_names}

    def get_arm_win_rates(self) -> Dict[str, float]:
        """Check how often each model actually earns a click."""
        valid_wins = [w for w in self.winner_history if w is not None]
        total_wins = len(valid_wins)

        if total_wins == 0:
            return {name: 0.0 for name in self.arm_names}

        counts = Counter(valid_wins)
        return {name: counts.get(name, 0) / total_wins for name in self.arm_names}

    def get_summary(self) -> dict:
        """Get a quick summary dictionary (useful for printing to console)."""
        num_iter = self.num_iterations
        total_clicks = sum(self.click_history)
        return {
            "iterations": num_iter,
            "total_regret": self.total_regret,
            "total_clicks": total_clicks,
            "global_ctr": total_clicks / num_iter if num_iter > 0 else 0.0,
            "regret_rate": self.total_regret / num_iter if num_iter > 0 else 0.0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Export the entire log to a Pandas DataFrame for plotting."""
        n = self.num_iterations
        if n == 0:
            return pd.DataFrame()

        return pd.DataFrame({
            "iteration": np.arange(n),
            "regret": self.regret_history,
            "cumulative_regret": self.cumulative_regret,
            "click": self.click_history,
            "winner": self.winner_history,
            "best": self.best_history,
            "selected_arms": [",".join(sorted(s)) for s in self.selection_history],
        })
