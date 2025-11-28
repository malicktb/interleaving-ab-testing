def get_click_winner(clicks, slate, attribution):
    if not clicks:
        return None

    first_click_pos = clicks[0]
    clicked_item = slate[first_click_pos]
    return attribution.get(clicked_item)


def compute_credit(clicks, slate, attribution, model="navigational"):
    if not clicks:
        return {}

    credit = {}

    if model == "navigational":
        first_click_pos = clicks[0]
        clicked_item = slate[first_click_pos]
        arm = attribution.get(clicked_item)
        if arm:
            credit[arm] = 1

    elif model == "informational":
        for click_pos in clicks:
            clicked_item = slate[click_pos]
            arm = attribution.get(clicked_item)
            if arm:
                credit[arm] = credit.get(arm, 0) + 1

    return credit
