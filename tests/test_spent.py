"""The committed register of spent held-out sets (eval/spent.py): a set scored once stays refused, even when
the git-ignored result files that the older guards check (runs/, eval/results/) are missing."""

import json

import pytest

from eval import spent
from eval.interface import FROZEN_PARSE_PROMPT, guard
from rl import goal_test

SPENT_SO_FAR = ("goal_test/test", "goal_test/m09_test", "m08b/heldout/extraction", "m08b/heldout/gateway",
                "m08b/heldout/hermes")


def test_the_register_lists_every_set_scored_so_far():
    register = json.loads(spent.SPENT_FILE.read_text())
    for name in SPENT_SO_FAR:
        entry = register["sets"][name]
        assert entry["date"] and entry["commit"] and entry["result_file"]


def test_refuse_and_mark(tmp_path):
    path = tmp_path / "spent.json"
    spent.refuse_if_spent("goal_test/m10_test", path)  # unknown set: no refusal, no file needed
    spent.mark_spent("goal_test/m10_test", "runs/m10/test_result.json", "note", path)
    assert spent.spent("goal_test/m10_test", path)["result_file"] == "runs/m10/test_result.json"
    with pytest.raises(SystemExit, match="already scored"):
        spent.refuse_if_spent("goal_test/m10_test", path)


@pytest.mark.parametrize("split, min_share", [("test", None), ("test", 0.05), ("m09_test", 0.05)])
def test_goal_test_refuses_a_spent_split_without_its_result_file(tmp_path, monkeypatch, split, min_share):
    for name in ("RESULT_DIR", "FLOOR_RESULT_DIR", "M09_RESULT_DIR"):  # as on a clean clone: no runs/
        monkeypatch.setattr(goal_test, name, tmp_path / name)
    with pytest.raises(SystemExit, match="already scored"):
        goal_test.run_split(split, [], final=True, min_share=min_share)


@pytest.mark.parametrize("path", ["extraction", "gateway", "hermes"])
def test_interface_refuses_a_spent_heldout_path_without_its_result_file(tmp_path, path):
    with pytest.raises(SystemExit, match="already scored"):
        guard(tmp_path, "heldout", path, final=True, revision="abc1234", prompt_hash=FROZEN_PARSE_PROMPT)
