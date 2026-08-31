from quant_pairs.screener import benjamini_hochberg


def test_benjamini_hochberg_is_monotone_in_rank_order() -> None:
    qvalues = benjamini_hochberg([0.04, 0.001, 0.03, 0.20])
    assert qvalues[1] <= qvalues[2] <= qvalues[0] <= qvalues[3]
    assert qvalues[1] == 0.004
