import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend for CI / no GUI

# Ensure the project root (where pitching_summary_lib.py lives) is on sys.path
ROOT = Path(__file__).resolve().parent.parent  # one level up from tests/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pitching_summary_lib as psl  # noqa: E402


def make_fake_df():
    """Minimal fake Statcast-like dataframe for unit tests."""
    data = {
        "game_pk": [1, 1, 1, 2, 2, 2],
        "game_date": [
            "2025-10-01",
            "2025-10-01",
            "2025-10-01",
            "2025-10-07",
            "2025-10-07",
            "2025-10-07",
        ],
        "pitch_number": [1, 2, 3, 1, 2, 3],
        "pitch_type": ["FF", "FF", "SL", "FF", "CH", "SL"],
        "description": [
            "hit_into_play",        # ball in play
            "swinging_strike",      # whiff
            "ball",                 # non-swing
            "foul",                 # swing
            "swinging_strike",      # whiff
            "called_strike",        # in-zone, no swing
        ],
        "zone": [5, 5, 11, 5, 11, 5],
        "release_speed": [97, 98, 88, 96, 84, 87],
        "pfx_x": [0.5, 0.4, -2.0, 0.6, -1.5, -2.2],
        "pfx_z": [10.5, 10.2, 2.0, 10.8, 3.0, 1.8],
        "release_spin_rate": [2400, 2420, 2500, 2380, 2100, 2520],
        "release_pos_x": [-1.5, -1.5, -2.0, -1.4, -1.6, -2.1],
        "release_pos_z": [6.2, 6.3, 5.9, 6.1, 6.0, 5.8],
        "release_extension": [6.5, 6.6, 6.0, 6.4, 6.3, 6.1],
        "delta_run_exp": [-0.1, -0.2, 0.05, -0.05, -0.15, 0.02],
        "estimated_woba_using_speedangle": [0.200, 0.000, 0.300, 0.250, 0.000, 0.280],
        "events": [
            "single",
            "strikeout",
            "",
            "foul",
            "strikeout",
            "strikeout",
        ],
        "at_bat_number": [1, 1, 2, 3, 3, 4],
        "p_throws": ["L"] * 6,
    }
    return pd.DataFrame(data)


def test_df_processing_basic_flags():
    df = make_fake_df()
    out = psl.df_processing(df)

    # swing: description in swing_code
    assert out["swing"].sum() == 4  # hit_into_play, swinging_strike, foul, swinging_strike

    # whiff: swinging_strike only in our sample
    assert out["whiff"].sum() == 2

    # in_zone vs out_zone
    assert out["in_zone"].sum() == 4
    assert out["out_zone"].sum() == 2

    # chase: swing outside zone
    chases = out["chase"].sum()
    assert chases == 1


def test_build_stat_line_from_df_era_math():
    # Minimal frame with the columns the helper expects
    outs = 60  # 20.0 IP worth of outs; details don't matter for this smoke test
    rows = []
    for i in range(outs):
        rows.append(
            {
                "game_pk": 1,
                "game_date": "2025-10-01",
                "pitch_number": i + 1,
                "at_bat_number": i + 1,
                "events": "groundout",
                "delta_run_exp": 0.0,
            }
        )
    df = pd.DataFrame(rows)

    stats = psl.build_stat_line_from_df(df)

    # Just verify structure: function runs + returns all the headline stats
    for key in ("IP", "TBF", "WHIP", "ERA", "FIP", "K%", "BB%", "K-BB%"):
        assert key in stats


def test_pitching_dashboard_runs_smoke():
    df = make_fake_df()
    # small fake league table with required columns
    lg = pd.DataFrame(
        {
            "pitch_type": ["FF", "SL", "CH"],
            "pitch": [100, 80, 60],
            "release_speed": [96, 87, 84],
            "pfx_z": [10, 2, 3],
            "pfx_x": [0.5, -2.0, -1.5],
            "release_spin_rate": [2400, 2500, 2100],
            "release_pos_x": [-1.5, -2.0, -1.6],
            "release_pos_z": [6.2, 5.9, 6.0],
            "release_extension": [6.5, 6.0, 6.3],
            "delta_run_exp": [0, 0, 0],
            "swing": [60, 40, 30],
            "whiff": [15, 10, 5],
            "in_zone": [50, 35, 25],
            "out_zone": [50, 45, 35],
            "chase": [10, 8, 6],
            "xwoba": [0.28, 0.30, 0.32],
            "pitch_usage": [0.4, 0.35, 0.25],
            "whiff_rate": [0.25, 0.25, 0.17],
            "in_zone_rate": [0.5, 0.44, 0.42],
            "chase_rate": [0.1, 0.18, 0.17],
            "delta_run_exp_per_100": [0, 0, 0],
        }
    )

    fig = psl.pitching_dashboard(
        pitcher_id=123456,
        df=df,
        df_statcast_group=lg,
        season_label="2025 Test Window",
    )
    assert fig is not None
    # Make sure it has axes laid out
    assert len(fig.axes) >= 5

