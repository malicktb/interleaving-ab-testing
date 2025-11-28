import numpy as np


def interleave(rankings, active_arms, slate_size=10, rng=None):
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
