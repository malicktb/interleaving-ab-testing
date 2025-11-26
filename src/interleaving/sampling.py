"""
Multileaving (Slate Generation).

Implements Team Draft Multileaving.

Multileaving is the term for mixing ranked lists from multiple
models (arms) into a single list (slate). It is the multi-player version
of Interleaving (which only compares 2 models).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AttributionData:
    """Tracks who drafted an item during Multileaving.

    In a Multileaving experiment, we need to assign credit back to the specific
    model that contributed the clicked item. This class stores that link.

    Attributes:
        arm_name: Name of the arm that drafted this item.
        original_rank: The rank (0-indexed) this item held in that arm's original list.
                       Used to analyze if arms are winning with their best items.
    """

    arm_name: str
    original_rank: int


def sample_slate(rankings, active_arms, slate_size=10, rng=None, method="team_draft"):
    """Perform Team Draft Multileaving to create a mixed slate.

    This function executes the Multileaving process:
    1. It takes rankings from all `active_arms`.
    2. It mixes them into a single list (`slate`) using the Team Draft algorithm.
    3. It records which arm contributed which item (`attribution`).

    Args:
        rankings: Dictionary of {Model Name -> List of 30 Item IDs}.
        active_arms: The set of models competing in this Multileaving round.
        slate_size: How many items to show on the final screen (default 10).
        rng: Random number generator.
        method: Mixing algorithm (currently only supports 'team_draft').

    Returns:
        slate: The final multileaved list of item IDs.
        attribution: Metadata mapping Item ID -> Info about who drafted it.
    """
    if method != "team_draft":
        raise ValueError(f"Unknown sampling method: {method}. Use 'team_draft'.")

    if rng is None:
        rng = np.random.default_rng()

    # Only look at the models selected for this Multileaving round
    active_rankings = {a: rankings[a] for a in active_arms if a in rankings}

    if not active_rankings:
        return [], {}

    # Randomize the draft order (who gets to pick first)
    draft_order = list(active_rankings.keys())
    rng.shuffle(draft_order)

    slate = []
    attribution = {}
    drafted_items = set()

    # Track how deep we are in each model's list
    pointers = {arm: 0 for arm in draft_order}

    # Safety check: don't go past the end of a list
    max_lens = {arm: len(r) for arm, r in active_rankings.items()}

    while len(slate) < slate_size:
        items_added_this_round = 0

        for arm in draft_order:
            if len(slate) >= slate_size:
                break

            ranking = active_rankings[arm]
            current_ptr = pointers[arm]

            # Find this model's top item that isn't already on the slate
            while current_ptr < max_lens[arm]:
                item_idx = ranking[current_ptr]
                current_ptr += 1

                if item_idx not in drafted_items:
                    slate.append(item_idx)
                    drafted_items.add(item_idx)

                    # Record attribution for the Multileaving update step
                    attribution[item_idx] = AttributionData(arm_name=arm, original_rank=current_ptr - 1)
                    pointers[arm] = current_ptr
                    items_added_this_round += 1
                    break

            # Save our place in the list for next time
            pointers[arm] = current_ptr

        # Stop if no models have any items left to give
        if items_added_this_round == 0:
            break

        # Reverse order for the next round
        # This mitigates position bias in the Multileaving process.
        draft_order.reverse()

    return slate, attribution
