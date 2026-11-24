from flight_delay_bioopt.data import load_config
from flight_delay_bioopt.tree_baselines import build_random_forest, build_xgboost


def test_tree_baseline_imbalance_settings() -> None:
    config = load_config()
    forest = build_random_forest(config)
    boosting = build_xgboost(config, scale_pos_weight=6.5)

    assert forest.class_weight == "balanced_subsample"
    assert boosting.scale_pos_weight == 6.5
    assert boosting.tree_method == "hist"
