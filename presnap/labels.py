"""One label vocabulary for every source.

Raw values seen in the data:
  NGS (BDB 2026/2027 team_coverage_type): COVER_0_MAN COVER_1_MAN COVER_2_MAN COVER_2_ZONE COVER_3_ZONE
      COVER_4_ZONE COVER_6_ZONE PREVENT
  nflverse participation defense_coverage_type: NGS for 2018-22, FTN from 2023, both as
      COVER_0..COVER_9, 2_MAN, PREVENT (NGS only), COMBO / BLOWN (FTN only)

Never pool label_source "ngs" with "ftn" as one target: they agree on only ~57% of shared plays.
"""
import polars as pl

# canonical coverage -> (family, man_zone, shell_played)
# shell_played is the shell of the coverage actually played (0/1/2 deep), not what was shown pre-snap.
CANON = {
    "C0": ("C0", "MAN", 0),
    "C1": ("C1", "MAN", 1),
    "C2M": ("C2M", "MAN", 2),
    "C2": ("C2", "ZONE", 2),
    "C3": ("C3", "ZONE", 1),
    "C4": ("QTR", "ZONE", 2),
    "C6": ("QTR", "ZONE", 2),
    "C9": ("QTR", "ZONE", None),  # FTN-only label, shell ambiguous in public definitions
    "PREVENT": (None, "ZONE", None),
    "COMBO": (None, "ZONE", None),
    "BLOWN": (None, "ZONE", None),
}

RAW_TO_CANON = {
    "COVER_0_MAN": "C0", "COVER_1_MAN": "C1", "COVER_2_MAN": "C2M", "COVER_2_ZONE": "C2",
    "COVER_3_ZONE": "C3", "COVER_4_ZONE": "C4", "COVER_6_ZONE": "C6",
    "COVER_0": "C0", "COVER_1": "C1", "2_MAN": "C2M", "COVER_2": "C2", "COVER_3": "C3",
    "COVER_4": "C4", "COVER_6": "C6", "COVER_9": "C9",
    "PREVENT": "PREVENT", "COMBO": "COMBO", "BLOWN": "BLOWN",
}

FAMILIES = ["C0", "C1", "C2M", "C2", "C3", "QTR"]
COVERAGE7 = ["C0", "C1", "C2M", "C2", "C3", "C4", "C6"]  # the NGS classes minus PREVENT


def canon(col: str) -> pl.Expr:
    return pl.col(col).replace_strict(RAW_TO_CANON, default=None)


def with_label_columns(df: pl.DataFrame, coverage: str = "coverage") -> pl.DataFrame:
    """Adds family, man_zone and shell_played from a canonical coverage column."""
    return df.with_columns(
        family=pl.col(coverage).replace_strict({k: v[0] for k, v in CANON.items()}, default=None),
        man_zone=pl.col(coverage).replace_strict({k: v[1] for k, v in CANON.items()}, default=None),
        shell_played=pl.col(coverage).replace_strict({k: v[2] for k, v in CANON.items()}, default=None, return_dtype=pl.Int8),
    )
