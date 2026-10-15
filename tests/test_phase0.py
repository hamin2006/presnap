"""Phase 0 self-checks. The data checks need `uv run python -m presnap.build` first."""
import polars as pl
import pytest

from presnap import io, labels, normalize

P = io.PROCESSED


def test_mirror_symmetry():
    # the same play run right and left must normalize to the same coordinates
    df = pl.DataFrame({"x": [40.0, 80.0], "y": [30.0, 53.3 - 30.0], "los": [35.0, 85.0], "right": [True, False]})
    x, y = normalize.to_rel(pl.col("x"), pl.col("y"), pl.col("los"), pl.col("right"), y_flipped=False)
    out = df.select(x, y)
    assert out["x_rel"].to_list() == [5.0, 5.0]
    assert out["y_rel"][0] == pytest.approx(out["y_rel"][1])


def test_every_raw_label_maps():
    for raw, c in labels.RAW_TO_CANON.items():
        fam, mz, _ = labels.CANON[c]
        assert mz in ("MAN", "ZONE"), raw
        assert fam is None or fam in labels.FAMILIES


needs_build = pytest.mark.skipif(not (P / "plays.parquet").exists(), reason="run presnap.build first")


@pytest.fixture(scope="module")
def plays():
    return pl.read_parquet(P / "plays.parquet")


@needs_build
def test_audit_counts(plays):
    assert plays.filter(source="bdb26").height == 14_108
    assert pl.read_parquet(P / "player_labels.parquet")["coverage_assignment"].is_not_null().sum() == 3_426
    # the audit's 5,215 double-counted 4 plays that StatsBomb lists twice; those are dropped as ambiguous
    assert plays.filter(source="statsbomb")["sb_quality_ok"].sum() == 5_207


@needs_build
def test_keys_unique_and_split(plays):
    assert plays.select("source", "game_id", "play_id").is_unique().all()
    assert plays["split"].null_count() == 0
    # no game is in two splits
    assert plays.group_by("source", "game_id").agg(pl.col("split").n_unique()).filter(pl.col("split") > 1).height == 0


@needs_build
def test_label_sources_never_mix(plays):
    assert set(plays["label_source"].drop_nulls().unique()) == {"ngs"}
    assert plays.filter(pl.col("season") >= 2023)["coverage_ftn"].null_count() < 100
    assert plays.filter(pl.col("season") <= 2022)["coverage_ftn"].null_count() == plays.filter(pl.col("season") <= 2022).height


@needs_build
def test_snap_frame_geometry():
    t = pl.scan_parquet(P / "tracking.parquet").filter(pl.col("frame") == 0).collect()
    for src in ("bdb26", "statsbomb"):
        s = t.filter(source=src)
        assert s.filter(side="off")["x_rel"].median() < 0 < s.filter(side="def")["x_rel"].median(), src
        assert (s["y_rel"].abs() > 32).mean() < 0.001, src  # inside the field (+ small margin)
    assert t.filter(source="bdb26").select("game_id", "play_id").n_unique() == 14_108


@needs_build
def test_y_matches_ftn_hash(plays):
    """Passer/QB at the snap sits on the charted hash: L -> y_rel > 0, R -> y_rel < 0, in both sources."""
    t = pl.scan_parquet(P / "tracking.parquet").filter((pl.col("frame") == 0) & (pl.col("position") == "QB") & (pl.col("side") == "off")).collect()
    j = t.join(plays.select("source", "game_id", "play_id", "ftn_starting_hash"), on=["source", "game_id", "play_id"])
    for src in ("bdb26", "statsbomb"):
        m = j.filter(source=src).group_by("ftn_starting_hash").agg(pl.col("y_rel").median())
        d = dict(zip(m["ftn_starting_hash"], m["y_rel"]))
        assert d["L"] > 2 and d["R"] < -2 and abs(d["M"]) < 1, (src, d)


@needs_build
def test_statsbomb_is_10hz_with_presnap():
    t = pl.scan_parquet(P / "tracking.parquet").filter(source="statsbomb").select("frame").collect()
    assert t["frame"].min() < -10  # pre-snap frames survive resampling
    assert t.filter(pl.col("frame") <= 0).height > 0.3 * t.height


def test_report_tables_are_well_formed():
    r = io.ROOT / "reports/phi_2025_defense_v0.md"
    if not r.exists():
        pytest.skip("run presnap.report_v0 first")
    lines = r.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("|---"):
            n = lines[i - 1].count("|")
            assert line.count("|") == n, lines[i - 1]
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                assert lines[j].count("|") == n, lines[j]
                j += 1


def test_shrinkage_pulls_small_samples_to_prior():
    from presnap.report_v0 import shrink, K
    assert shrink(0, 0, 0.3) == 0.3
    assert shrink(5, 5, 0.3) == pytest.approx((5 + K * 0.3) / (5 + K))
    assert abs(shrink(500, 1000, 0.3) - 0.5) < 0.01
