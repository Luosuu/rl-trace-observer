def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """1.0 when the completion mentions the target number, else 0.0."""
    return 1.0 if ground_truth in solution_str.split() else 0.0
