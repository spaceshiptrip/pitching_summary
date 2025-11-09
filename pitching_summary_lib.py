from __future__ import annotations
import math
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from matplotlib.ticker import MaxNLocator, FuncFormatter, PercentFormatter
import matplotlib.ticker as mtick

from PIL import Image
import pybaseball as pyb
import matplotlib.colors as mcolors

import os
from datetime import datetime
import datetime as _dt






# ---------- Plotting defaults ----------

font_properties = {"family": "DejaVu Sans", "size": 12}
font_properties_titles = {"family": "DejaVu Sans", "size": 20}
font_properties_axes = {"family": "DejaVu Sans", "size": 16}

sns.set_theme(
    style="whitegrid",
    palette="deep",
    font="DejaVu Sans",
    font_scale=1.2,
    color_codes=True,
)

mpl.rcParams["figure.dpi"] = 300

# ---------- Pitch color + label maps ----------

pitch_colours = {
    # Fastballs
    "FF": {"colour": "#FF007D", "name": "4-Seam Fastball"},
    "FA": {"colour": "#FF007D", "name": "Fastball"},
    "SI": {"colour": "#98165D", "name": "Sinker"},
    "FC": {"colour": "#BE5FA0", "name": "Cutter"},
    # Offspeed
    "CH": {"colour": "#F79E70", "name": "Changeup"},
    "FS": {"colour": "#FE6100", "name": "Splitter"},
    "SC": {"colour": "#F08223", "name": "Screwball"},
    "FO": {"colour": "#FFB000", "name": "Forkball"},
    # Sliders
    "SL": {"colour": "#67E18D", "name": "Slider"},
    "ST": {"colour": "#1BB999", "name": "Sweeper"},
    "SV": {"colour": "#376748", "name": "Slurve"},
    # Curves
    "KC": {"colour": "#311D8B", "name": "Knuckle Curve"},
    "CU": {"colour": "#3025CE", "name": "Curveball"},
    "CS": {"colour": "#274BFC", "name": "Slow Curve"},
    "EP": {"colour": "#648FFF", "name": "Eephus"},
    # Others
    "KN": {"colour": "#867A08", "name": "Knuckleball"},
    "PO": {"colour": "#472C30", "name": "Pitch Out"},
    "UN": {"colour": "#9C8975", "name": "Unknown"},
}

dict_colour = {k: v["colour"] for k, v in pitch_colours.items()}
dict_pitch = {k: v["name"] for k, v in pitch_colours.items()}

# ---------- Fangraphs summary formatting ----------

fangraphs_stats_dict = {
    "IP": {"table_header": r"$\bf{IP}$", "format": ".1f"},
    "TBF": {"table_header": r"$\bf{PA}$", "format": ".0f"},
    "WHIP": {"table_header": r"$\bf{WHIP}$", "format": ".2f"},
    "ERA": {"table_header": r"$\bf{ERA}$", "format": ".2f"},
    "FIP": {"table_header": r"$\bf{FIP}$", "format": ".2f"},
    "K%": {"table_header": r"$\bf{K\%}$", "format": ".1%"},
    "BB%": {"table_header": r"$\bf{BB\%}$", "format": ".1%"},
    "K-BB%": {"table_header": r"$\bf{K-BB\%}$", "format": ".1%"},
}

pitch_stats_dict = {
    "pitch": {"table_header": r"$\bf{Count}$", "format": ".0f"},
    "pitch_usage": {"table_header": r"$\bf{Pitch\%}$", "format": ".1%"},
    "release_speed": {"table_header": r"$\bf{Velocity}$", "format": ".1f"},
    "pfx_z": {"table_header": r"$\bf{iVB}$", "format": ".1f"},
    "pfx_x": {"table_header": r"$\bf{HB}$", "format": ".1f"},
    "release_spin_rate": {"table_header": r"$\bf{Spin}$", "format": ".0f"},
    "release_pos_x": {"table_header": r"$\bf{hRel}$", "format": ".1f"},
    "release_pos_z": {"table_header": r"$\bf{vRel}$", "format": ".1f"},
    "release_extension": {"table_header": r"$\bf{Ext.}$", "format": ".1f"},
    "delta_run_exp_per_100": {"table_header": r"$\bf{RV/100}$", "format": ".1f"},
    "whiff_rate": {"table_header": r"$\bf{Whiff\%}$", "format": ".1%"},
    "in_zone_rate": {"table_header": r"$\bf{Zone\%}$", "format": ".1%"},
    "chase_rate": {"table_header": r"$\bf{Chase\%}$", "format": ".1%"},
    "xwoba": {"table_header": r"$\bf{xwOBA}$", "format": ".3f"},
}

table_columns = [
    "pitch_description",
    "pitch",
    "pitch_usage",
    "release_speed",
    "pfx_z",
    "pfx_x",
    "release_spin_rate",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "delta_run_exp_per_100",
    "whiff_rate",
    "in_zone_rate",
    "chase_rate",
    "xwoba",
]




# ---------- Core data helpers ----------


def get_pitcher_id_for_name(name: str) -> Optional[int]:
    """
    Resolve a pitcher name like 'Blake Snell' to an MLBAM pitcher id using pybaseball.
    Returns None if we can't find a match.
    """
    name = name.strip()
    if not name:
        return None

    # Try "First Last"
    parts = name.split()
    if len(parts) >= 2:
        first = " ".join(parts[:-1])
        last = parts[-1]
    else:
        # Single token: let fuzzy handle it
        first = ""
        last = parts[0]

    try:
        df = pyb.playerid_lookup(last, first)
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        # prefer pitcher w/ most recent MLB season
        df = df.sort_values("mlb_played_last", ascending=False)
        return int(df.iloc[0]["key_mlbam"])

    # Fallback: fuzzy lookup
    try:
        fuzzy = pyb.playerid_lookup(last, first, fuzzy=True)
    except Exception:
        fuzzy = pd.DataFrame()

    if not fuzzy.empty:
        fuzzy = fuzzy.sort_values("mlb_played_last", ascending=False)
        return int(fuzzy.iloc[0]["key_mlbam"])

    return None


def fetch_pitcher_statcast(pitcher_id: int, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Wrapper around pybaseball.statcast_pitcher with a clean interface.
    """
    df = pyb.statcast_pitcher(start_date, end_date, pitcher_id)
    if df is None:
        return pd.DataFrame()

    # Normalize column names if needed
    # (pybaseball already returns what we expect for this project)
    return df


def df_processing(df_pyb: pd.DataFrame) -> pd.DataFrame:
    df = df_pyb.copy()

    swing_code = [
        "foul_bunt", "foul", "hit_into_play", "swinging_strike", "foul_tip",
        "swinging_strike_blocked", "missed_bunt", "bunt_foul_tip",
    ]
    whiff_code = ["swinging_strike", "foul_tip", "swinging_strike_blocked"]

    df["swing"] = df["description"].isin(swing_code)
    df["whiff"] = df["description"].isin(whiff_code)
    df["in_zone"] = df["zone"] < 10
    df["out_zone"] = df["zone"] > 10
    df["chase"] = (~df["in_zone"]) & (df["swing"] == 1)

    # Convert pfx to inches
    df["pfx_z"] = df["pfx_z"] * 12
    df["pfx_x"] = df["pfx_x"] * 12
    return df


def fangraphs_pitching_leaderboards(season: int) -> pd.DataFrame:
    url = (
        "https://www.fangraphs.com/api/leaders/major-league/data"
        f"?age=&pos=all&stats=pit&lg=all&season={season}&season1={season}"
        "&ind=0&qual=0&type=8&month=0&pageitems=500000"
    )
    data = requests.get(url).json()
    return pd.DataFrame(data["data"])



def _add_pitch_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same swing/whiff/zone/chase + pfx unit transforms
    used for the pitcher-level df_processing, but for league data.
    """
    df = df.copy()

    swing_code = [
        "foul_bunt", "foul", "hit_into_play", "swinging_strike",
        "foul_tip", "swinging_strike_blocked", "missed_bunt", "bunt_foul_tip"
    ]
    whiff_code = ["swinging_strike", "foul_tip", "swinging_strike_blocked"]

    df["swing"] = df["description"].isin(swing_code)
    df["whiff"] = df["description"].isin(whiff_code)
    df["in_zone"] = df["zone"] < 10
    df["out_zone"] = df["zone"] > 10
    df["chase"] = (~df["in_zone"]) & (df["swing"])

    # convert movement from feet → inches (to match the notebook + CSV)
    if "pfx_z" in df.columns:
        df["pfx_z"] = df["pfx_z"] * 12
    if "pfx_x" in df.columns:
        df["pfx_x"] = df["pfx_x"] * 12

    return df


def _group_league(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate league Statcast data into the same schema as statcast_2024_grouped.csv.
    """
    if df.empty:
        raise RuntimeError("No Statcast data returned for league window; cannot build comparison table.")

    df = _add_pitch_flags(df)

    g = (
        df.groupby("pitch_type", dropna=True)
        .agg(
            pitch=("pitch_type", "count"),
            release_speed=("release_speed", "mean"),
            pfx_z=("pfx_z", "mean"),
            pfx_x=("pfx_x", "mean"),
            release_spin_rate=("release_spin_rate", "mean"),
            release_pos_x=("release_pos_x", "mean"),
            release_pos_z=("release_pos_z", "mean"),
            release_extension=("release_extension", "mean"),
            delta_run_exp=("delta_run_exp", "sum"),
            swing=("swing", "sum"),
            whiff=("whiff", "sum"),
            in_zone=("in_zone", "sum"),
            out_zone=("out_zone", "sum"),
            chase=("chase", "sum"),
            xwoba=("estimated_woba_using_speedangle", "mean"),
        )
        .reset_index()
    )

    # rates
    g["pitch_usage"] = g["pitch"] / g["pitch"].sum()
    g["whiff_rate"] = g["whiff"] / g["swing"].replace(0, np.nan)
    g["in_zone_rate"] = g["in_zone"] / g["pitch"].replace(0, np.nan)
    g["chase_rate"] = g["chase"] / g["out_zone"].replace(0, np.nan)
    g["delta_run_exp_per_100"] = -g["delta_run_exp"] / g["pitch"].replace(0, np.nan) * 100

    # "All" summary row to mirror the provided CSV
    total_pitch = float(g["pitch"].sum())
    total = pd.DataFrame(
        [{
            "pitch_type": "All",
            "pitch": total_pitch,
            "release_speed": (df["release_speed"].mean() if "release_speed" in df.columns else np.nan),
            "pfx_z": (df["pfx_z"].mean() if "pfx_z" in df.columns else np.nan),
            "pfx_x": (df["pfx_x"].mean() if "pfx_x" in df.columns else np.nan),
            "release_spin_rate": (df["release_spin_rate"].mean() if "release_spin_rate" in df.columns else np.nan),
            "release_pos_x": (df["release_pos_x"].mean() if "release_pos_x" in df.columns else np.nan),
            "release_pos_z": (df["release_pos_z"].mean() if "release_pos_z" in df.columns else np.nan),
            "release_extension": (df["release_extension"].mean() if "release_extension" in df.columns else np.nan),
            "delta_run_exp": df["delta_run_exp"].sum() if "delta_run_exp" in df.columns else np.nan,
            "swing": df["swing"].sum(),
            "whiff": df["whiff"].sum(),
            "in_zone": df["in_zone"].sum(),
            "out_zone": df["out_zone"].sum(),
            "chase": df["chase"].sum(),
            "xwoba": (df["estimated_woba_using_speedangle"].mean()
                      if "estimated_woba_using_speedangle" in df.columns else np.nan),
            "pitch_usage": 1.0,
            "whiff_rate": df["whiff"].sum() / df["swing"].sum() if df["swing"].sum() > 0 else np.nan,
            "in_zone_rate": df["in_zone"].sum() / total_pitch if total_pitch > 0 else np.nan,
            "chase_rate": df["chase"].sum() / df["out_zone"].sum() if df["out_zone"].sum() > 0 else np.nan,
            "delta_run_exp_per_100": (
                -df["delta_run_exp"].sum() / total_pitch * 100
                if "delta_run_exp" in df.columns and total_pitch > 0 else np.nan
            ),
            "all": "all",
        }]
    )

    # ensure "all" column exists on main table too for schema compatibility
    g["all"] = ""

    return pd.concat([g, total], ignore_index=True)


def load_or_build_league_grouped(
    start_date: str,
    end_date: str,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    League-average pitch metrics for use in coloring & comparisons.

    Behavior:
      - If `cache_path` exists → load and return.
      - If dates overlap 2024 and bundled `statcast_2024_grouped.csv` exists → use that.
      - Otherwise:
          * fetch statcast(start_date, end_date)
          * aggregate into the standard schema
          * optionally cache to `cache_path`
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            return pd.read_csv(cache_path)

    # bundled 2024 file (same as repo)
    here = Path(__file__).resolve().parent
    bundled_2024 = here / "statcast_2024_grouped.csv"

    if bundled_2024.exists():
        if start_date.startswith("2024") or end_date.startswith("2024"):
            return pd.read_csv(bundled_2024)

    # dynamic build for arbitrary seasons
    # consider enabling pybaseball cache for speed & robustness
    try:
        # pulls full league Statcast between dates
        df_all = pyb.statcast(start_date, end_date)
    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch league Statcast data: {e}"
        )

    league = _group_league(df_all)

    if cache_path is not None:
        league.to_csv(cache_path, index=False)

    return league


def _add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add swing/whiff/zone/chase and convert pfx_x/z to inches for movement."""
    df = df.copy()

    swing_code = [
        "foul_bunt",
        "foul",
        "hit_into_play",
        "swinging_strike",
        "foul_tip",
        "swinging_strike_blocked",
        "missed_bunt",
        "bunt_foul_tip",
    ]
    whiff_code = [
        "swinging_strike",
        "foul_tip",
        "swinging_strike_blocked",
    ]

    df["swing"] = df["description"].isin(swing_code)
    df["whiff"] = df["description"].isin(whiff_code)
    df["in_zone"] = df["zone"] < 10
    df["out_zone"] = df["zone"] > 10
    df["chase"] = (~df["in_zone"]) & (df["swing"])

    # movement in inches
    if "pfx_z" in df.columns:
        df["pfx_z"] = df["pfx_z"] * 12
    if "pfx_x" in df.columns:
        df["pfx_x"] = df["pfx_x"] * 12

    return df


def _group_league_by_pitch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate league-wide Statcast into the columns used for comparison / coloring.
    Output structure is compatible with pitch_table's expectations.
    """
    needed = [
        "pitch_type",
        "release_speed",
        "pfx_z",
        "pfx_x",
        "release_spin_rate",
        "release_pos_x",
        "release_pos_z",
        "release_extension",
        "delta_run_exp",
        "swing",
        "whiff",
        "in_zone",
        "out_zone",
        "chase",
        "estimated_woba_using_speedangle",
    ]
    df = df[[c for c in needed if c in df.columns]].copy()

    if df.empty:
        return pd.DataFrame(columns=["pitch_type"])

    g = (
        df.groupby("pitch_type")
        .agg(
            pitch=("pitch_type", "count"),
            release_speed=("release_speed", "mean"),
            pfx_z=("pfx_z", "mean"),
            pfx_x=("pfx_x", "mean"),
            release_spin_rate=("release_spin_rate", "mean"),
            release_pos_x=("release_pos_x", "mean"),
            release_pos_z=("release_pos_z", "mean"),
            release_extension=("release_extension", "mean"),
            delta_run_exp=("delta_run_exp", "sum"),
            swing=("swing", "sum"),
            whiff=("whiff", "sum"),
            in_zone=("in_zone", "sum"),
            out_zone=("out_zone", "sum"),
            chase=("chase", "sum"),
            xwoba=("estimated_woba_using_speedangle", "mean"),
        )
        .reset_index()
    )

    # derived rates
    total_pitches = g["pitch"].sum()
    g["pitch_usage"] = g["pitch"] / total_pitches

    g["whiff_rate"] = g["whiff"] / g["swing"].replace(0, np.nan)
    g["in_zone_rate"] = g["in_zone"] / g["pitch"].replace(0, np.nan)
    g["chase_rate"] = g["chase"] / g["out_zone"].replace(0, np.nan)
    g["delta_run_exp_per_100"] = -g["delta_run_exp"] / g["pitch"].replace(0, np.nan) * 100

    return g


def load_or_build_league_grouped(
    start_date: str,
    end_date: str,
    cache_path: str = "statcast_grouped_cache.csv",
    postseason_only: bool = False,
    regular_only: bool = False,
) -> pd.DataFrame:
    """
    Used by ps_cli.py to get league-average by pitch type for the same window.
    - If cache_path exists, use it.
    - Else, pull Statcast for the window, aggregate, and cache.
    """
    # If a cached table exists, trust it (user can regenerate manually if needed)
    if os.path.exists(cache_path):
        try:
            df_cached = pd.read_csv(cache_path)
            if "pitch_type" in df_cached.columns:
                return df_cached
        except Exception:
            pass  # fall through to rebuild

    # Build from Statcast
    df_all = pyb.statcast(start_date, end_date)
    if df_all is None or df_all.empty:
        # Fall back to empty; calling code should handle gracefully
        return pd.DataFrame(columns=["pitch_type"])

    # Optional postseason/regular filters
    if "game_type" in df_all.columns:
        if postseason_only:
            df_all = df_all[df_all["game_type"].isin(["F", "D", "L", "W"])]
        elif regular_only:
            df_all = df_all[df_all["game_type"] == "R"]

    if df_all.empty:
        return pd.DataFrame(columns=["pitch_type"])

    df_all = _add_derived_flags(df_all)
    g = _group_league_by_pitch(df_all)

    # cache for next time
    try:
        g.to_csv(cache_path, index=False)
    except Exception:
        pass

    return g


# ---------- Bio & images ----------

def player_headshot(pitcher_id: int, ax: plt.Axes) -> None:
    url = (
        "https://img.mlbstatic.com/mlb-photos/image/"
        "upload/d_people:generic:headshot:67:current.png"
        f"/w_640,q_auto:best/v1/people/{pitcher_id}/headshot/silo/current.png"
    )
    img = Image.open(BytesIO(requests.get(url).content))
    ax.imshow(img, extent=[0, 1, 0, 1], origin="upper")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def player_bio(pitcher_id: int, season_label: str, ax: plt.Axes) -> None:
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"
    data = requests.get(url).json()["people"][0]

    name = data["fullName"]
    hand = data["pitchHand"]["code"]
    age = data.get("currentAge", "?")
    height = data.get("height", "?")
    weight = data.get("weight", "?")

    ax.text(0.5, 1.0, f"{name}", va="top", ha="center", fontsize=40)
    ax.text(0.5, 0.65, f"{hand}HP, Age {age}, {height}/{weight}",
            va="top", ha="center", fontsize=18)
    ax.text(0.5, 0.38, "Season Pitching Summary",
            va="top", ha="center", fontsize=24)
    ax.text(0.5, 0.15, season_label,
            va="top", ha="center", fontsize=18, fontstyle="italic")
    ax.axis("off")


def plot_logo(pitcher_id: int, ax: plt.Axes) -> None:
    player_url = f"https://statsapi.mlb.com/api/v1/people?personIds={pitcher_id}&hydrate=currentTeam"
    pdata = requests.get(player_url).json()["people"][0]
    team_url = "https://statsapi.mlb.com" + pdata["currentTeam"]["link"]
    tdata = requests.get(team_url).json()
    team_abb = tdata["teams"][0]["abbreviation"]

    # ESPN logo URL pattern
    logo_url = (
        "https://a.espncdn.com/combiner/i?img=/i/teamlogos/mlb/500/scoreboard/"
        f"{team_abb.lower()}.png&h=500&w=500"
    )

    img = Image.open(BytesIO(requests.get(logo_url).content))
    ax.imshow(img, extent=[0.0, 1.0, 0, 1], origin="upper")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ---------- Plots ----------

def velocity_kdes(df: pd.DataFrame,
                  ax: plt.Axes,
                  gs,
                  gs_x: List[int],
                  gs_y: List[int],
                  fig: plt.Figure,
                  df_statcast_group: pd.DataFrame) -> None:

    ax.axis("off")
    ax.set_title("Pitch Velocity Distribution", fontdict=font_properties_titles)

    sorted_value_counts = df["pitch_type"].value_counts().sort_values(ascending=False)
    items_in_order = sorted_value_counts.index.tolist()

    inner = gridspec.GridSpecFromSubplotSpec(
        len(items_in_order), 1,
        subplot_spec=gs[gs_x[0]:gs_x[-1], gs_y[0]:gs_y[-1]]
    )
    subaxes = [fig.add_subplot(s) for s in inner]

    vmin = math.floor(df["release_speed"].min() / 5) * 5
    vmax = math.ceil(df["release_speed"].max() / 5) * 5

    for idx, pitch_type in enumerate(items_in_order):
        ax_i = subaxes[idx]
        data = df[df["pitch_type"] == pitch_type]["release_speed"]

        if data.nunique() == 1:
            val = data.iloc[0]
            ax_i.plot([val, val], [0, 1], linewidth=4,
                      color=dict_colour.get(pitch_type, "#000000"))
        else:
            sns.kdeplot(
                data,
                ax=ax_i,
                fill=True,
                clip=(data.min(), data.max()),
                color=dict_colour.get(pitch_type, "#000000"),
            )

        # Player mean
        pm = data.mean()
        ax_i.axvline(pm, linestyle="--",
                     color=dict_colour.get(pitch_type, "#000000"))

        # League mean (if available)
        lg = df_statcast_group[df_statcast_group["pitch_type"] == pitch_type]
        if not lg.empty and not pd.isna(lg["release_speed"].iloc[0]):
            lm = lg["release_speed"].iloc[0]
            ax_i.axvline(lm, linestyle=":",
                         color=dict_colour.get(pitch_type, "#000000"))

        ax_i.set_xlim(vmin, vmax)
        ax_i.set_yticks([])
        ax_i.grid(axis="x", linestyle="--", alpha=0.3)
        ax_i.set_xlabel("")
        ax_i.set_ylabel("")
        if idx < len(items_in_order) - 1:
            ax_i.tick_params(axis="x", labelbottom=False)
        ax_i.text(-0.01, 0.5, pitch_type,
                  transform=ax_i.transAxes,
                  va="center", ha="right", fontsize=10)

    subaxes[-1].set_xlabel("Velocity (mph)", font_properties_axes)



def rolling_pitch_usage(df: pd.DataFrame, ax: plt.Axes, window: int = 5) -> None:
    """
    Rolling pitch usage by game for the filtered sample.

    - df must be the already-processed pitcher Statcast (df_processing)
    - window in games; will be clamped if there are fewer games
    """
    if df.empty or "game_pk" not in df.columns or "pitch_type" not in df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "No games in selected range", ha="center", va="center")
        return

    # sort games chronologically and map to 1..N
    df = df.sort_values(["game_date", "game_pk"])
    game_ids = df["game_pk"].unique()
    num_games = len(game_ids)
    if num_games == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No games in selected range", ha="center", va="center")
        return

    eff_window = min(window, num_games)
    game_num_map = {g: i + 1 for i, g in enumerate(game_ids)}

    # per-game pitch counts by pitch_type
    counts = (
        df.groupby(["game_pk", "pitch_type"])["pitch_type"]
          .count()
          .rename("count")
          .reset_index()
    )
    totals = (
        df.groupby("game_pk")["pitch_type"]
          .count()
          .rename("total")
          .reset_index()
    )
    merged = counts.merge(totals, on="game_pk", how="left")
    merged["usage"] = merged["count"] / merged["total"]
    merged["game_number"] = merged["game_pk"].map(game_num_map)

    pitch_types = df["pitch_type"].value_counts().index.tolist()
    ymax = 0.0

    for pt in pitch_types:
        sub = merged[merged["pitch_type"] == pt]
        if sub.empty:
            continue

        # reindex over all games so rolling window handles zero-usage games
        series = (
            sub.set_index("game_number")["usage"]
               .reindex(range(1, num_games + 1), fill_value=0.0)
        )

        roll = series.rolling(eff_window, min_periods=1).mean()
        if roll.max() == 0:
            continue

        sns.lineplot(
            x=roll.index,
            y=roll.values,
            ax=ax,
            linewidth=2,
            color=dict_colour.get(pt, "#000000"),
            label=pt,
        )
        ymax = max(ymax, float(roll.max()))

    if ymax == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "Not enough data for rolling usage",
                ha="center", va="center")
        return

    ax.set_xlim(1, num_games)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel("Game", fontdict=font_properties_axes)
    ax.set_ylabel("Pitch Usage", fontdict=font_properties_axes)
    ax.set_title(f"{eff_window}-Game Rolling Pitch Usage",
                 fontdict=font_properties_titles)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
    )

    ax.legend(fontsize=8, frameon=False)


def build_stat_line_from_df(
    df: pd.DataFrame,
    pitcher_id: Optional[int] = None,
    season: Optional[int] = None,
    game_type: Optional[str] = None,
) -> dict:
    """
    Build the header-line stats for the filtered selection.

    Priority:
      1) If (pitcher_id, season, game_type) define a clean MLB split:
         - game_type == "R" => regular season
         - game_type == "P" => postseason
         Use MLB Stats API to get official values.
      2) Otherwise, derive from the filtered Statcast df (approximate).

    Returns raw values; K% etc are fractions (0.273) so the caller
    can format with .1%% etc.
    """
    nan = float("nan")

    def _to_float(v):
        try:
            if v in (None, ""):
                return nan
            return float(v)
        except Exception:
            return nan

    def _pct_field(row, key):
        """
        MLB Stats API gives e.g. "27.3" or "27.3%".
        Convert to fraction: 0.273
        """
        v = row.get(key)
        if v in (None, ""):
            return nan
        if isinstance(v, str) and v.endswith("%"):
            try:
                return float(v[:-1]) / 100.0
            except Exception:
                return nan
        try:
            return float(v) / 100.0
        except Exception:
            return nan

    FIP_CONST = 3.1  # simple global constant; fine for our use

    # ----- 1) Official MLB split when clearly defined -----
    if pitcher_id is not None and season is not None and game_type in {"R", "P"}:
        try:
            mlb_game_type = "R" if game_type == "R" else "P"
            url = (
                f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats"
                f"?stats=season&group=pitching&gameType={mlb_game_type}&season={season}"
            )
            js = requests.get(url, timeout=10).json()
            stats_list = js.get("stats", [])
            if stats_list and stats_list[0].get("splits"):
                row = stats_list[0]["splits"][0]["stat"]

                # IP (inningsPitched is a weird string sometimes)
                ip_str = row.get("inningsPitched", "0.0")
                ip_val = nan
                try:
                    ip_val = float(ip_str)
                except ValueError:
                    if "." in ip_str:
                        whole, frac = ip_str.split(".", 1)
                        if frac.isdigit():
                            ip_val = float(whole) + int(frac) / 3.0

                bf = _to_float(row.get("battersFaced"))
                so = _to_float(row.get("strikeOuts"))
                bb = _to_float(row.get("baseOnBalls"))
                hbp = _to_float(row.get("hitBatsmen"))
                hr = _to_float(row.get("homeRuns"))

                era = _to_float(row.get("era"))
                whip = _to_float(row.get("whip"))

                # Use API percentages when present; fall back to bf-based
                k_pct = _pct_field(row, "strikeoutPercentage")
                bb_pct = _pct_field(row, "baseOnBallPercentage")
                if not np.isfinite(k_pct) and bf and bf > 0:
                    k_pct = so / bf
                if not np.isfinite(bb_pct) and bf and bf > 0:
                    bb_pct = bb / bf
                kbb_pct = (
                    k_pct - bb_pct
                    if np.isfinite(k_pct) and np.isfinite(bb_pct)
                    else nan
                )

                # FIP from official component stats
                if ip_val and ip_val > 0 and np.isfinite(hr) and np.isfinite(bb) and np.isfinite(hbp) and np.isfinite(so):
                    fip = ((13 * hr) + 3 * (bb + hbp) - 2 * so) / ip_val + FIP_CONST
                else:
                    fip = nan

                return {
                    "IP": ip_val,
                    "TBF": bf if np.isfinite(bf) else 0,
                    "WHIP": whip,
                    "ERA": era,
                    "FIP": fip,
                    "K%": k_pct,
                    "BB%": bb_pct,
                    "K-BB%": kbb_pct,
                }
        except Exception:
            # On any failure, fall back to Statcast-based calc
            pass

    # ----- 2) Fallback: derive from Statcast df -----
    if df is None or df.empty:
        return {
            "IP": nan, "TBF": 0, "WHIP": nan,
            "ERA": nan, "FIP": nan,
            "K%": nan, "BB%": nan, "K-BB%": nan,
        }

    df_sorted = df.sort_values(
        ["game_pk", "at_bat_number", "pitch_number"],
        ignore_index=True,
    )

    # One row per PA (last pitch)
    df_pa = (
        df_sorted
        .groupby(["game_pk", "at_bat_number"], as_index=False)
        .tail(1)
        .copy()
    )
    pa = len(df_pa)

    # Outs -> IP
    OUT_EVENTS = {
        "strikeout": 1,
        "strikeout_double_play": 2,
        "strikeout_triple_play": 3,
        "groundout": 1,
        "grounded_into_double_play": 2,
        "double_play": 2,
        "triple_play": 3,
        "field_out": 1,
        "force_out": 1,
        "sac_bunt": 1,
        "sac_fly": 1,
        "sac_fly_double_play": 2,
        "sacrifice_bunt": 1,
        "sacrifice_fly": 1,
        "pickoff": 1,
    }
    ev_pa = df_pa["events"].fillna("")
    outs = int(sum(OUT_EVENTS.get(e, 0) for e in ev_pa))
    ip = outs / 3.0 if outs > 0 else nan

    # Approx RA from score deltas (still an approximation)
    runs_allowed = nan
    needed_cols = {
        "home_score", "away_score",
        "post_home_score", "post_away_score",
        "inning_topbot", "game_pk",
    }
    if needed_cols.issubset(df_pa.columns):
        runs_allowed = 0.0
        for game_id, g in df_sorted.groupby("game_pk"):
            g_pa = df_pa[df_pa["game_pk"] == game_id]
            if g_pa.empty:
                continue

            top_ct = (g_pa["inning_topbot"] == "Top").sum()
            bot_ct = (g_pa["inning_topbot"] == "Bot").sum()
            if top_ct == 0 and bot_ct == 0:
                continue

            is_home_pitcher = top_ct > bot_ct
            if is_home_pitcher:
                diffs = (g_pa["post_away_score"] - g_pa["away_score"]).clip(lower=0)
            else:
                diffs = (g_pa["post_home_score"] - g_pa["home_score"]).clip(lower=0)

            runs_allowed += float(diffs.sum())

    if ip and ip > 0 and np.isfinite(runs_allowed):
        era = runs_allowed * 9.0 / ip
    else:
        era = nan

    # Components for WHIP / FIP / K%, etc.
    walks = ev_pa.isin(["walk", "intent_walk"]).sum()
    hbp = ev_pa.eq("hit_by_pitch").sum()
    strikeouts = ev_pa.str.startswith("strikeout").sum()
    hr = ev_pa.eq("home_run").sum()

    whip = (walks + hbp) / ip if ip and ip > 0 else nan
    k_rate = strikeouts / pa if pa > 0 else nan
    bb_rate = walks / pa if pa > 0 else nan
    kbb_rate = k_rate - bb_rate if pa > 0 else nan

    if ip and ip > 0:
        fip = ((13 * hr) + 3 * (walks + hbp) - 2 * strikeouts) / ip + FIP_CONST
    else:
        fip = nan

    return {
        "IP": ip,
        "TBF": pa,
        "WHIP": whip,
        "ERA": era,
        "FIP": fip,
        "K%": k_rate,
        "BB%": bb_rate,
        "K-BB%": kbb_rate,
    }




def break_plot(df: pd.DataFrame, ax: plt.Axes) -> None:
    throws = df["p_throws"].iloc[0]

    if throws == "R":
        x = -df["pfx_x"]
    else:
        x = df["pfx_x"]

    sns.scatterplot(
        x=x,
        y=df["pfx_z"],
        hue=df["pitch_type"],
        palette=dict_colour,
        ec="black",
        ax=ax,
        alpha=0.9,
        zorder=2,
        legend=False,
    )

    ax.axhline(0, color="#808080", alpha=0.5, linestyle="--")
    ax.axvline(0, color="#808080", alpha=0.5, linestyle="--")

    ax.set_xlabel("Horizontal Break (in)", font_properties_axes)
    ax.set_ylabel("Induced Vertical Break (in)", font_properties_axes)
    ax.set_title("Pitch Breaks", font_properties_titles)

    ax.set_xticks(range(-20, 21, 10))
    ax.set_yticks(range(-20, 21, 10))
    ax.set_xlim(-25, 25)
    ax.set_ylim(-25, 25)
    ax.set_aspect("equal", "box")

    if throws == "R":
        ax.text(-24, -24, "← Glove Side", fontsize=8,
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"))
        ax.text(24, -24, "Arm Side →", fontsize=8,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"))
    else:
        ax.invert_xaxis()
        ax.text(24, -24, "← Arm Side", fontsize=8,
                ha="left", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"))
        ax.text(-24, -24, "Glove Side →", fontsize=8,
                ha="right", va="bottom",
                bbox=dict(facecolor="white", edgecolor="black"))

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: int(v)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: int(v)))

# ---------- Pitch table helpers ----------

def df_grouping(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    grp = df.groupby("pitch_type").agg(
        pitch=("pitch_type", "count"),
        release_speed=("release_speed", "mean"),
        pfx_z=("pfx_z", "mean"),
        pfx_x=("pfx_x", "mean"),
        release_spin_rate=("release_spin_rate", "mean"),
        release_pos_x=("release_pos_x", "mean"),
        release_pos_z=("release_pos_z", "mean"),
        release_extension=("release_extension", "mean"),
        delta_run_exp=("delta_run_exp", "sum"),
        swing=("swing", "sum"),
        whiff=("whiff", "sum"),
        in_zone=("in_zone", "sum"),
        out_zone=("out_zone", "sum"),
        chase=("chase", "sum"),
        xwoba=("estimated_woba_using_speedangle", "mean"),
    ).reset_index()

    grp["pitch_description"] = grp["pitch_type"].map(dict_pitch).fillna(grp["pitch_type"])
    grp["pitch_usage"] = grp["pitch"] / grp["pitch"].sum()
    grp["whiff_rate"] = grp["whiff"] / grp["swing"].replace(0, np.nan)
    grp["in_zone_rate"] = grp["in_zone"] / grp["pitch"]
    grp["chase_rate"] = grp["chase"] / grp["out_zone"].replace(0, np.nan)
    grp["delta_run_exp_per_100"] = -grp["delta_run_exp"] / grp["pitch"] * 100
    grp["colour"] = grp["pitch_type"].map(dict_colour).fillna("#CCCCCC")

    grp = grp.sort_values("pitch_usage", ascending=False)
    colours = grp["colour"].tolist()

    # Add "All" row
    total = pd.Series({
        "pitch_type": "All",
        "pitch_description": "All",
        "pitch": df["pitch_type"].count(),
        "pitch_usage": 1.0,
        "release_speed": np.nan,
        "pfx_z": np.nan,
        "pfx_x": np.nan,
        "release_spin_rate": np.nan,
        "release_pos_x": np.nan,
        "release_pos_z": np.nan,
        "release_extension": df["release_extension"].mean(),
        "delta_run_exp_per_100": -df["delta_run_exp"].sum() / df["pitch_type"].count() * 100,
        "whiff_rate": df["whiff"].sum() / df["swing"].sum(),
        "in_zone_rate": df["in_zone"].sum() / df["pitch_type"].count(),
        "chase_rate": df["chase"].sum() / df["out_zone"].sum(),
        "xwoba": df["estimated_woba_using_speedangle"].mean(),
    })
    grp = pd.concat([grp, total.to_frame().T], ignore_index=True)
    return grp, colours


def plot_pitch_format(df_group: pd.DataFrame) -> pd.DataFrame:
    df = df_group[table_columns].copy()
    for col, props in pitch_stats_dict.items():
        if col in df.columns:
            fmt = props["format"]
            def _fmt(v):
                if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
                    return format(v, fmt)
                return v
            df[col] = df[col].map(_fmt)
    return df


def pitch_table(df: pd.DataFrame,
                df_statcast_group: pd.DataFrame,
                ax: plt.Axes,
                fontsize: int = 12) -> None:
    df_group, row_colours = df_grouping(df)

    # Build background heatmap colors vs league
    cmap_sum = mpl.colors.LinearSegmentedColormap.from_list("", ["#648FFF","#FFFFFF","#FFB000"])
    cmap_sum_r = mpl.colors.LinearSegmentedColormap.from_list("", ["#FFB000","#FFFFFF","#648FFF"])

    def get_color(val, vmin, vmax, cmap):
        # Missing/invalid value → no color
        if val is None or not np.isfinite(val):
            return "#ffffff"
        if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax):
            return "#ffffff"

        # Ensure min <= max (handles negative means cleanly)
        if vmin > vmax:
            vmin, vmax = vmax, vmin

        # Degenerate range → no gradient
        if vmin == vmax:
            return "#ffffff"

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        return mcolors.to_hex(cmap(norm(val)))

    cell_colours = []
    for _, row in df_group.iterrows():
        pt = row["pitch_type"]
        lg = df_statcast_group[df_statcast_group["pitch_type"] == pt]
        row_cols = []
        for col in table_columns:
            if col not in pitch_stats_dict or lg.empty:
                row_cols.append("#FFFFFF")
                continue

            props = pitch_stats_dict[col]
            val = row[col]
            lg_vals = pd.to_numeric(lg[col], errors="coerce").dropna()
            if lg_vals.empty or pd.isna(val):
                row_cols.append("#FFFFFF")
                continue

            if col == "delta_run_exp_per_100":
                c = get_color(val, -1.5, 1.5, cmap_sum)
            elif col == "xwoba":
                mean = lg_vals.mean()
                c = get_color(val, mean * 0.7, mean * 1.3, cmap_sum_r)
            else:
                mean = lg_vals.mean()
                c = get_color(val, mean * 0.7, mean * 1.3, cmap_sum)
            row_cols.append(c)
        cell_colours.append(row_cols)

    df_display = plot_pitch_format(df_group)

    table = ax.table(
        cellText=df_display.values,
        colLabels=[pitch_stats_dict.get(c, {"table_header": c}).get("table_header", c)
                   if i > 0 else r"$\bf{Pitch\ Name}$"
                   for i, c in enumerate(table_columns)],
        cellLoc="center",
        cellColours=cell_colours,
        bbox=[0, 0, 1, 1],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 0.6)

    # First column: pitch names with pitch-specific color
    for i in range(1, len(df_display) + 1):
        pitch_name = table[(i, 0)].get_text().get_text()
        if i - 1 < len(row_colours):
            table[(i, 0)].set_facecolor(row_colours[i - 1])
        table[(i, 0)].get_text().set_fontweight("bold")
        table[(i, 0)].get_text().set_color("#000000")

    ax.axis("off")

# ---------- Dashboard entrypoint ----------

def pitching_dashboard(
    pitcher_id: int,
    df: pd.DataFrame,
    df_statcast_group: pd.DataFrame,
    season_label: Optional[str] = None,
    stats: Optional[List[str]] = None,
    title_suffix: Optional[str] = None,
) -> plt.Figure:
    if stats is None:
        stats = ["IP", "TBF", "WHIP", "ERA", "FIP", "K%", "BB%", "K-BB%"]

    if df.empty:
        raise ValueError("No Statcast data for this selection; nothing to plot.")

    # Process pitcher data for plots
    df = df_processing(df)

    # Infer season_label if not provided
    if not season_label:
        if "game_date" in df.columns:
            dmin = pd.to_datetime(df["game_date"].min(), errors="coerce")
            dmax = pd.to_datetime(df["game_date"].max(), errors="coerce")
            if pd.notnull(dmin) and pd.notnull(dmax):
                if dmin.year == dmax.year:
                    season_label = f"{dmin.year} Season ({dmin.date()} – {dmax.date()})"
                else:
                    season_label = f"{dmin.date()} – {dmax.date()}"
            else:
                season_label = "Season Summary"
        else:
            season_label = "Season Summary"

    # Infer season number
    season = None
    if "game_date" in df.columns:
        season = pd.to_datetime(df["game_date"], errors="coerce").dt.year.mode()
        if len(season):
            season = int(season.iloc[0])

    # Infer game_type for split detection
    game_type = None
    if "game_type" in df.columns:
        gtypes = set(df["game_type"].dropna().unique())
        # Pure regular season
        if gtypes == {"R"}:
            game_type = "R"
        # Pure postseason: Statcast uses F/D/L/W/etc; MLB Stats uses P for postseason
        elif gtypes.issubset({"F", "D", "L", "W", "C"}):
            game_type = "P"

    # ---------- layout ----------
    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(
        6, 8,
        height_ratios=[2, 20, 9, 36, 36, 7],
        width_ratios=[1, 18, 18, 18, 18, 18, 18, 1],
        figure=fig,
    )

    ax_headshot     = fig.add_subplot(gs[1, 1:3])
    ax_bio          = fig.add_subplot(gs[1, 3:5])
    ax_logo         = fig.add_subplot(gs[1, 5:7])
    ax_season_table = fig.add_subplot(gs[2, 1:7])
    ax_plot_1       = fig.add_subplot(gs[3, 1:3])
    ax_plot_2       = fig.add_subplot(gs[3, 3:5])
    ax_plot_3       = fig.add_subplot(gs[3, 5:7])
    ax_table        = fig.add_subplot(gs[4, 1:7])
    ax_footer       = fig.add_subplot(gs[5, 1:7])

    # border strips
    fig.add_subplot(gs[0, 1:7]).axis("off")
    fig.add_subplot(gs[:, 0]).axis("off")
    fig.add_subplot(gs[:, -1]).axis("off")

    # ---------- top summary table from filtered df ----------
    summary = build_stat_line_from_df(
        df,
        pitcher_id=pitcher_id,
        season=season,
        game_type=game_type,
    )

    row_vals = []
    col_labels = []

    for key in stats:
        col_labels.append(fangraphs_stats_dict[key]["table_header"])
        val = summary.get(key, float("nan"))
        fmt = fangraphs_stats_dict[key]["format"]
        try:
            disp = format(val, fmt) if np.isfinite(val) else "---"
        except Exception:
            disp = "---"
        row_vals.append(disp)

    table = ax_season_table.table(
        cellText=[row_vals],
        colLabels=col_labels,
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax_season_table.axis("off")

    # ---------- bio strip ----------
    player_headshot(pitcher_id, ax_headshot)
    player_bio(pitcher_id, season_label, ax_bio)
    plot_logo(pitcher_id, ax_logo)

    # ---------- plots ----------
    velocity_kdes(df, ax_plot_1, gs, [3, 4], [1, 3], fig, df_statcast_group)
    rolling_pitch_usage(df, ax_plot_2, window=5)
    break_plot(df, ax_plot_3)

    # ---------- pitch table ----------
    pitch_table(df, df_statcast_group, ax_table, fontsize=10)

    # ---------- footer ----------
    ax_footer.axis("off")
    ax_footer.text(
        0, 1,
        "By: @TJStats (layout)  |  CLI wrapper & filters by @spaceshiptrip",
        ha="left", va="top", fontsize=10,
    )
    ax_footer.text(
        0.5, 1,
        "Colour coding vs league averages by pitch type",
        ha="center", va="top", fontsize=9,
    )
    ax_footer.text(
        1, 1,
        "Data: MLB Statcast, FanGraphs (for league)  |  Images: MLB, ESPN",
        ha="right", va="top", fontsize=9,
    )

    plt.tight_layout()
    return fig

