"""Music the story's people hear (11-26): the cue planner is told that music played in the scene is the cue's music
for as long as it is played, and each cue says whether it is heard there (tts_reader/score.py plan_scenes)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import score as SC  # noqa: E402

PARAS = ["Orla came into the room.", "The players struck up and Tam led her out.", "They spoke as they turned.",
         "Later Dr. Marsh walked home alone."]


def planned(monkeypatch, cues):
    seen = {}

    def chat(url, model, prompt, max_tokens=3000):
        seen["prompt"] = prompt
        return json.dumps({"cues": cues})
    monkeypatch.setattr(SC, "_chat", chat)
    return SC.plan_scenes(PARAS, "http://x", style="chamber strings"), seen["prompt"]


def test_the_planner_is_told_that_music_played_in_the_scene_is_the_cue(monkeypatch):
    out, prompt = planned(monkeypatch, [])
    assert SC.HEARD_RULE in prompt and "- heard:" in prompt and '"heard": false' in prompt
    low = SC.HEARD_RULE.lower()
    for need in ("played in the scene", "that music", "without a gap", "same kind of music", "only shades"):
        assert need in low, need
    assert prompt.index(SC.HEARD_RULE) < prompt.index("chamber strings") < prompt.index("Chapter:")


def test_each_cue_says_whether_it_is_heard_there(monkeypatch):
    out, _ = planned(monkeypatch, [{"start": 1, "end": 2, "scene": "a", "mood": "b", "music": "c", "heard": True},
                                   {"start": 3, "end": 3, "scene": "a", "mood": "b", "music": "c", "heard": "yes"},
                                   ])
    assert [c["heard"] for c in out] == [True, False] and [(c["start"], c["end"]) for c in out] == [(1, 2), (3, 3)]


def test_the_rule_names_no_story():
    low = SC.HEARD_RULE.lower()
    for w in ("ball", "waltz", "minuet", "quadrille", "violin", "harpsichord", "regency", "court"):
        assert w not in low, w
