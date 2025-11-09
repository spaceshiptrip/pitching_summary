#!/usr/bin/env python
import argparse
import sys
from datetime import datetime
from typing import Optional, Tuple

import pybaseball as pyb
from pitching_summary_lib import (
    get_pitcher_id_for_name,
    fetch_pitcher_statcast,
    load_or_build_league_grouped,
    pitching_dashboard,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TJStats-style pitching summary dashboards from Statcast."
    )
    parser.add_argument(
        "pitcher",
        help='Pitcher name, e.g. "Tarik Skubal" or "Shohei Ohtani"',
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Season year (e.g. 2024). Used for defaults & postseason windows.",
    )
    parser.add_argument(
        "--start-date",
        dest="start_date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Overrides --season default.",
    )
    parser.add_argument(
        "--end-date",
        dest="end_date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Overrides --season default.",
    )
    parser.add_argument(
        "--postseason",
        action="store_true",
        help="Use only postseason games for the given season.",
    )
    parser.add_argument(
        "--regular",
        action="store_true",
        help="Use only regular-season games for the given season.",
    )
    parser.add_argument(
        "--team-filter",
        type=str,
        default=None,
        help=(
            "Optional: limit to games where this team (MLB abbrev) is either the "
            "pitcher's team or the opponent. Example: --team-filter LAD"
        ),
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out_path",
        type=str,
        default=None,
        help="Output PNG file. If omitted, shows the figure interactively.",
    )
    parser.add_argument(
        "--league-cache",
        type=str,
        default="statcast_grouped_cache.csv",
        help="CSV path to cache league-wide grouped metrics.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    return parser.parse_args()


def resolve_date_range(
    season: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    postseason: bool,
    regular: bool,
) -> Tuple[str, str]:
    """
    Priority:
    1. If start/end explicitly set → use them.
    2. Else if postseason → tight-ish postseason window for that season.
    3. Else if regular → broad regular-season-ish window.
    4. Else if season → full year window.
    5. Else → fallback to a very broad window (you really should pass --season).
    """
    if start_date and end_date:
        return start_date, end_date
    if start_date and not end_date:
        # open-ended end: default to late Nov of that year
        year = int(start_date[:4])
        return start_date, f"{year}-11-30"
    if end_date and not start_date:
        year = int(end_date[:4])
        return f"{year}-03-01", end_date

    if season:
        if postseason:
            # MLB postseason window (you can tweak if you want to be exact per year)
            return f"{season}-10-01", f"{season}-11-30"
        if regular:
            # Rough regular season window
            return f"{season}-03-01", f"{season}-10-01"
        # Default: whole season + postseason wiggle
        return f"{season}-03-01", f"{season}-11-30"

    # No season, no dates: last-resort fallback
    return "2024-03-01", "2024-11-30"


def filter_postseason(df):
    if "game_type" not in df.columns:
        return df
    # Common Statcast postseason flags: 'F' (WC), 'D' (DS), 'L' (LCS), 'W' (WS)
    return df[df["game_type"].isin(["F", "D", "L", "W"])]


def filter_regular_season(df):
    if "game_type" not in df.columns:
        return df
    # Regular season is typically 'R'
    return df[df["game_type"] == "R"]


def filter_team(df, team_abbr: str):
    """
    Restrict to games where the given team is either batting or fielding team.
    Uses Statcast columns: 'home_team', 'away_team' if present.
    Safe no-op if columns missing.
    """
    if not team_abbr:
        return df
    team_abbr = team_abbr.upper()
    cols = df.columns
    if "home_team" in cols and "away_team" in cols:
        mask = (df["home_team"] == team_abbr) | (df["away_team"] == team_abbr)
        return df[mask]
    return df


def main():
    args = parse_args()

    # Resolve date window
    start, end = resolve_date_range(
        season=args.season,
        start_date=args.start_date,
        end_date=args.end_date,
        postseason=args.postseason,
        regular=args.regular,
    )

    if args.verbose:
        print(f"Using date range: {start} → {end}", file=sys.stderr)

    # Look up pitcher ID
    pitcher_id = get_pitcher_id_for_name(args.pitcher)
    if pitcher_id is None:
        print(f"Could not resolve pitcher name: {args.pitcher}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        print(f"Pitcher ID for {args.pitcher}: {pitcher_id}", file=sys.stderr)

    # Fetch pitcher-level Statcast
    print(f"Fetching Statcast for pitcher {pitcher_id} from {start} to {end}...")
    df = fetch_pitcher_statcast(pitcher_id, start, end)
    if df is None or df.empty:
        print("No Statcast data found for that pitcher/date range.", file=sys.stderr)
        sys.exit(1)

    # Apply postseason / regular filters (if user requested and didn't hard-code dates)
    # Note: if user explicitly set start/end, we respect that window but still allow
    # --postseason/--regular as an extra filter.
    if args.postseason:
        df = filter_postseason(df)
    elif args.regular:
        df = filter_regular_season(df)

    # Optional team filter
    if args.team_filter:
        df = filter_team(df, args.team_filter)

    if df.empty:
        print("No data left after applying filters (date/game-type/team).", file=sys.stderr)
        sys.exit(1)

    # League comparison table: use same date range & filters (except pitcher-id)
    print("Loading / building league-average comparison table...")
    league_df = load_or_build_league_grouped(
        start,
        end,
        cache_path=args.league_cache,
        postseason_only=args.postseason,
        regular_only=args.regular,
    )

    # Build dashboard
    print("Building dashboard...")
    fig = pitching_dashboard(
        pitcher_id=pitcher_id,
        df=df,
        df_statcast_group=league_df,
        title_suffix=_make_title_suffix(args, start, end),
    )

    if args.out_path:
        fig.savefig(args.out_path, dpi=300, bbox_inches="tight")
        print(f"Saved dashboard to {args.out_path}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


def _make_title_suffix(args: argparse.Namespace, start: str, end: str) -> str:
    # Nice little label for the figure header/footer.
    parts = []
    if args.postseason:
        parts.append("Postseason")
    elif args.regular:
        parts.append("Regular Season")

    if args.season:
        parts.append(str(args.season))

    if args.start_date or args.end_date:
        # Use explicit window
        parts = [f"{start} to {end}"]

    if args.team_filter:
        parts.append(f"vs/with {args.team_filter.upper()}")

    return " • ".join(parts) if parts else f"{start} to {end}"


if __name__ == "__main__":
    main()

