import polars as pl

from presnap.shell import shell_at


def play(pid, pts):
    return [dict(source="t", game_id=1, play_id=pid, x_rel=x, y_rel=y) for x, y in pts]


def test_shell_rule_on_textbook_alignments():
    rows = (play(0, [(5, 10), (6, -10), (1, 3)])                    # 0-high: everyone shallow
            + play(1, [(13, 1), (6, 10), (6, -10)])                 # 1-high: deep middle safety
            + play(2, [(12, 9), (12, -9), (6, 0)])                  # 2-high: split safeties, middle open
            + play(3, [(12, 9), (12, -9), (14, 0)])                 # MOFC wins: someone deep in the middle
            + play(4, [(12, 9), (13, 14), (6, -9)])                 # deep on one side only -> 1-high
            + play(5, [(12, 25), (12, -25), (6, 0)]))               # deep but outside the width -> 0-high
    out = shell_at(pl.DataFrame(rows), depth=10, width=16, gap=5, sep=8).sort("play_id")
    assert out["shell"].to_list() == [0, 1, 2, 1, 1, 0]
