"""Simulating User Clicks (The Reward Signal).

This module compares the generated Slate against the real historical data (Ground Truth)
to determine if a "Click" occurred. If it did, it calculates which Arm gets the credit.
"""

import numpy as np


def compute_reward(slate, attribution, labels, click_model="navigational"):
    """Determine who won the round based on user behavior.

    We simulate the user scanning the slate from top to bottom.

    Supported User Models:
    1. 'navigational' (Standard): The user is looking for one specific thing.
       They click the *first* relevant item they see and stop.
       -> Returns: (Winner Name, Rank)
       -> Use for: Updating the Bandit Strategy (W/N matrices).

    2. 'informational' (Metrics): The user wants to explore.
       They click *all* relevant items in the list.
       -> Returns: List of [(Winner Name, Rank), ...]
       -> Use for: Calculating Total Yield / Global CTR in reports.

    Args:
        slate: The list of item IDs shown to the user.
        attribution: Metadata tracking who drafted each item ID.
        labels: The Ground Truth (1.0 if the real user clicked, 0.0 otherwise).
        click_model: 'navigational' or 'informational'.

    Returns:
        The winner(s) and the rank position(s) of the click(s).
    """
    valid_clicks = []

    for position, item_idx in enumerate(slate):
        # Check Ground Truth: Did the real user click this item?
        if labels[item_idx] == 1.0:
            attr_data = attribution.get(item_idx)

            if attr_data:
                result = (attr_data.arm_name, position)

                if click_model == "navigational":
                    # User found what they wanted. Stop immediately.
                    return result

                # User keeps looking. Record this click and continue.
                valid_clicks.append(result)

    # If we finished the loop and are in 'navigational' mode, no one won.
    if click_model == "navigational":
        return None, None

    return valid_clicks
