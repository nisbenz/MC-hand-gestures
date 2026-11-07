from mc_gesture.record.labels import Label, load, save, score


def start(t, gesture):
    return {"type": "gesture_start", "gesture": gesture, "t": t}


def test_hit_miss_and_false_activation():
    labels = [Label(1.0, 2.0, "Closed_Fist"), Label(5.0, 6.0, "Open_Palm")]
    events = [start(1.15, "Closed_Fist"), start(3.0, "Closed_Fist"), {"type": "gesture_end", "t": 2.0}]
    s = score(labels, events)
    assert [(l.gesture, round(d, 2)) for l, d in s.hits] == [("Closed_Fist", 0.15)]
    assert [l.gesture for l in s.misses] == ["Open_Palm"]
    assert [e["t"] for e in s.false] == [3.0]


def test_wrong_gesture_is_a_miss_plus_a_false_activation():
    s = score([Label(1.0, 2.0, "Closed_Fist")], [start(1.2, "Open_Palm")])
    assert len(s.hits) == 0 and len(s.misses) == 1 and len(s.false) == 1


def test_event_slightly_before_hand_timed_label_counts():
    s = score([Label(1.0, 2.0, "Closed_Fist")], [start(0.8, "Closed_Fist")])
    assert len(s.hits) == 1


def test_one_event_matches_only_one_label():
    labels = [Label(1.0, 2.0, "Closed_Fist"), Label(1.5, 2.5, "Closed_Fist")]
    s = score(labels, [start(1.6, "Closed_Fist")])
    assert len(s.hits) == 1 and len(s.misses) == 1


def test_save_and_load_roundtrip(tmp_path):
    save(tmp_path, [Label(2.0, 3.0, "Open_Palm"), Label(0.5, 1.0, "Closed_Fist")])
    assert [l.gesture for l in load(tmp_path)] == ["Closed_Fist", "Open_Palm"]
    assert load(tmp_path / "missing") is None
