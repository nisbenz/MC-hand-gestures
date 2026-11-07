from mc_gesture.calibration.routine import Phase, compute
from mc_gesture.config import Config


def results(fist_scores, relaxed_fist=(), sizes=(0.2,) * 30):
    stand = Phase("stand", "", sizes=list(sizes), frames=len(sizes))
    relax = Phase("relax", "", scores={"Closed_Fist": list(relaxed_fist)}, frames=90)
    fist = Phase("Closed_Fist", "", "Closed_Fist", scores={"Closed_Fist": list(fist_scores)}, frames=90)
    return [stand, relax, fist]


def test_body_size_cutoff_is_60_percent_of_the_player():
    config, _ = compute(Config(gestures=("Closed_Fist",)), results([0.8] * 50))
    assert config.min_body_size == 0.12


def test_thresholds_follow_the_players_scores():
    scores = [0.6 + i * 0.005 for i in range(60)]  # 0.60 .. 0.895
    config, _ = compute(Config(gestures=("Closed_Fist",)), results(scores))
    enter, exit_ = config.thresholds["Closed_Fist"]
    assert 0.6 < enter < 0.7 and exit_ < enter


def test_accidental_scores_raise_enter():
    config, notes = compute(Config(gestures=("Closed_Fist",)), results([0.7] * 50, relaxed_fist=[0.8] * 30))
    assert config.thresholds["Closed_Fist"][0] >= 0.82
    assert any("accidental" in n for n in notes)


def test_unrecognized_gesture_keeps_defaults():
    config, notes = compute(Config(gestures=("Closed_Fist",)), results([0.9] * 3))
    assert "Closed_Fist" not in config.thresholds
    assert any("poor gesture" in n for n in notes)


def test_saved_config_loads_back(tmp_path):
    config, _ = compute(Config(gestures=("Closed_Fist",)), results([0.8] * 50))
    config.save(tmp_path / "c.json")
    loaded = Config.load(tmp_path / "c.json")
    assert loaded.thresholds == config.thresholds and loaded.gestures == ("Closed_Fist",)
