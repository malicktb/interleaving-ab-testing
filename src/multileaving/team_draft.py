"""Team Draft Multileaving implementation.

Implements the snake draft algorithm for fair slate construction
as described in Radlinski et al. (2008).
"""

from typing import Dict, List, Tuple
import numpy as np


def interleave(
    rankings: Dict[str, List[int]],
    active_arms: List[str],
    slate_size: int = 10,
    rng: np.random.Generator = None,
) -> Tuple[List[int], Dict[int, str]]:
    """Construct a multileaved slate using Team Draft (snake draft).

    The snake draft algorithm ensures fair position allocation by reversing
    the draft order after each round:
        Round 1: A → B → C
        Round 2: C → B → A (reversed)
        Round 3: A → B → C
        ...

    Args:
        rankings: Dict mapping arm names to their ranked document lists.
        active_arms: List of arm names participating in this round.
        slate_size: Target size for the multileaved list.
        rng: Random number generator for shuffling initial draft order.

    Returns:
        Tuple of (slate, attribution_map) where:
            - slate: List of document indices in display order
            - attribution_map: Dict mapping doc_idx to the arm that contributed it
    """
    if rng is None:
        rng = np.random.default_rng()

    active_rankings = {a: rankings[a] for a in active_arms if a in rankings}

    if not active_rankings:
        return [], {}

    draft_order = list(active_rankings.keys())
    rng.shuffle(draft_order)

    slate = []
    attribution = {}
    drafted_items = set()

    pointers = {arm: 0 for arm in draft_order}
    max_lens = {arm: len(r) for arm, r in active_rankings.items()}

    while len(slate) < slate_size:
        items_added_this_round = 0

        for arm in draft_order:
            if len(slate) >= slate_size:
                break

            ranking = active_rankings[arm]
            current_ptr = pointers[arm]

            while current_ptr < max_lens[arm]:
                item_idx = ranking[current_ptr]
                current_ptr += 1

                if item_idx not in drafted_items:
                    slate.append(item_idx)
                    drafted_items.add(item_idx)
                    attribution[item_idx] = arm
                    pointers[arm] = current_ptr
                    items_added_this_round += 1
                    break

            pointers[arm] = current_ptr

        if items_added_this_round == 0:
            break

        draft_order.reverse()

    return slate, attribution
