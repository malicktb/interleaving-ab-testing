"""Experiment Log (Telemetry).

This module records the history of the experiment. Tracks every decision the Bandit made, who won, and whether
the Bandit made a mistake (Regret).
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RegretTelemetry:
    """The Experiment Scoreboard.

    Tracks the performance of the Bandit Strategy over time.

    Attributes:
        arm_names: A list of all available arms (so we can report 0% for unused ones).
        regret_history: Log of "Mistakes" (1 = We failed to pick the best arm, 0 = We picked it).
        click_history: Log of "Yield" (1 = User clicked something, 0 = User ignored the list).
        selection_history: Log of which arms were allowed to play in each round.
        winner_history: Log of which arm got the credit (if any).
        best_history: Log of which arm was *actually* the best for that specific user.
    """

    arm_names: list = field(default_factory=list)
    regret_history: list = field(default_factory=list)
    click_history: list = field(default_factory=list)
    selection_history: list = field(default_factory=list)
    winner_history: list = field(default_factory=list)
    best_history: list = field(default_factory=list)

    def record_iteration(self, selected_arms, winner, best_arm):
        """Log the results of a single round.

        Args:
            selected_arms: The set of arms the Strategy decided to test.
            winner: The arm that actually got a click (or None).
            best_arm: The arm that ranked the clicked item highest (The "Right Answer").

        Returns:
            The regret score for this round (0 or 1).
        """
        self.selection_history.append(selected_arms)
        self.winner_history.append(winner)
        self.best_history.append(best_arm)

        # Track if the system generated value (a click)
        self.click_history.append(1 if winner is not None else 0)

        # Calculate "Regret" (Did the Bandit make a mistake?)
        # Rule: If the best arm was included in the slate, the Bandit succeeded.
        # Even if the user didn't click (bad luck), the Bandit made the right choice.
        if best_arm is None:
            regret = 0.0  # No good answer existed, so no mistake was possible.
        elif best_arm in selected_arms:
            regret = 0.0  # Success: We picked the winner.
        else:
            regret = 1.0  # Failure: We left the best arm on the bench.

        self.regret_history.append(regret)
        return regret

    @property
    def cumulative_regret(self):
        """Running total of mistakes made over time."""
        return np.cumsum(self.regret_history)

    @property
    def cumulative_clicks(self):
        """Running total of clicks received over time."""
        return np.cumsum(self.click_history)

    @property
    def total_regret(self):
        """Total number of mistakes."""
        return sum(self.regret_history)

    @property
    def num_iterations(self):
        """Total number of rounds played."""
        return len(self.regret_history)

    def get_arm_selection_rates(self, window=None):
        """Check how often each arm is getting picked.

        Args:
            window: If set (e.g., 1000), only look at the last N rounds.

        Returns:
            Dictionary mapping arm names to selection rates (0.0 to 1.0).
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

    def get_arm_win_rates(self):
        """Check how often each arm actually earns a click.

        Returns:
            Dictionary mapping arm names to win rates (0.0 to 1.0).
        """
        valid_wins = [w for w in self.winner_history if w is not None]
        total_wins = len(valid_wins)

        if total_wins == 0:
            return {name: 0.0 for name in self.arm_names}

        counts = Counter(valid_wins)
        return {name: counts.get(name, 0) / total_wins for name in self.arm_names}

    def get_summary(self):
        """Get a quick summary dictionary (useful for printing to console).

        Returns:
            Dictionary with iterations, total_regret, total_clicks, global_ctr, regret_rate.
        """
        num_iter = self.num_iterations
        total_clicks = sum(self.click_history)
        return {
            "iterations": num_iter,
            "total_regret": self.total_regret,
            "total_clicks": total_clicks,
            "global_ctr": total_clicks / num_iter if num_iter > 0 else 0.0,
            "regret_rate": self.total_regret / num_iter if num_iter > 0 else 0.0,
        }

    def to_dataframe(self):
        """Export the entire log to a Pandas DataFrame for plotting.

        Returns:
            DataFrame with iteration, regret, cumulative_regret, click, winner, best, selected_arms.
        """
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
