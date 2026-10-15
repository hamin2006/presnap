"""Score the Phase 1A human review against the rule and the NGS labels.

Put the downloaded shell_review_answers.csv in data/review/, then: uv run python -m presnap.review_score
"""
import polars as pl

from presnap import io

R = io.ROOT / "data/review"
NUM = {"0-high": 0, "1-high": 1, "2-high": 2}


def main():
    key = pl.read_csv(R / "key.csv")
    ans = pl.read_csv(R / "shell_review_answers.csv").with_columns(
        you_snap=pl.col("snap").replace_strict(NUM, default=None), you_post=pl.col("post").replace_strict(NUM, default=None))
    d = key.join(ans, on="review_id").with_columns(dispute=pl.col("shell_throw") != pl.col("shell_played"))
    post = d.drop_nulls("you_post")
    for name, sub in [("disputed (rule != NGS)", post.filter("dispute")), ("control (rule == NGS)", post.filter(~pl.col("dispute")))]:
        n = sub.height
        print(f"{name}: n={n} | you agree with rule {(sub['you_post'] == sub['shell_throw']).mean():.0%} "
              f"| with NGS {(sub['you_post'] == sub['shell_played']).mean():.0%}" if n else f"{name}: no answers")
    snap = d.drop_nulls("you_snap")
    print(f"at the snap: n={snap.height} | you agree with rule {(snap['you_snap'] == snap['shell_snap']).mean():.0%}")
    print(f"unsure: snap {(d['snap'] == 'unsure').sum()}, post {(d['post'] == 'unsure').sum()}")
    gate = post.filter("dispute")
    if gate.height:
        ok = (gate["you_post"] == gate["shell_throw"]).mean() >= 0.85
        print("gate (d) — reviewer agrees with the rule on >= 85% of disputed plays:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
