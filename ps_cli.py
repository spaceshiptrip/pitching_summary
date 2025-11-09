import argparse
import sys

import matplotlib.pyplot as plt
import pybaseball as pyb

from pitching_summary_lib import (
    pitching_dashboard,
    load_or_build_league_grouped,
)

def resolve_pitcher_id(pitcher: str) -> int:
    # If numeric, treat as MLBAM id directly
    if pitcher.isdigit():
        return int(pitcher)

    # Expect "First Last"
    parts = pitcher.strip().split()
    if len(parts) < 2:
        print("Please pass either MLBAM ID or 'First Last'", file=sys.stderr)
        sys.exit(1)
    first, last = parts[0], parts[-1]

    df = pyb.playerid_lookup(last, first)
    if df.empty:
        print(f"No player match for {pitcher}", file=sys.stderr)
        sys.exit(1)
    return int(df.iloc[0]["key_mlbam"])


def main():
    p = argparse.ArgumentParser(
        description="Generate TJStats-style pitching summary graphics."
    )
    p.add_argument("pitcher", help="Pitcher name 'First Last' or MLBAM ID")
    p.add_argument(
        "--season",
        type=int,
        help="Season year (e.g. 2024). If set, overrides --start/--end.",
    )
    p.add_argument("--start", help="Start date YYYY-MM-DD")
    p.add_argument("--end", help="End date YYYY-MM-DD")
    p.add_argument(
        "--out",
        help="Output image path (e.g. skubal_2024.png). If omitted, just shows the plot.",
    )
    p.add_argument(
        "--league-cache",
        default="statcast_grouped_cache.csv",
        help="Path to cache league averages (reused across runs).",
    )
    args = p.parse_args()

    pitcher_id = resolve_pitcher_id(args.pitcher)

    if args.season:
        start = f"{args.season}-03-01"
        end = f"{args.season}-11-30"
        season_label = f"{args.season} Season"
    else:
        if not (args.start and args.end):
            print("Provide --season OR both --start and --end", file=sys.stderr)
            sys.exit(1)
        start, end = args.start, args.end
        season_label = f"{start} to {end}"

    print(f"Fetching Statcast for pitcher {pitcher_id} from {start} to {end}...")
    df_pyb = pyb.statcast_pitcher(start, end, pitcher_id)
    if df_pyb.empty:
        print("No Statcast data for this pitcher/date range.", file=sys.stderr)
        sys.exit(1)

    pitcher_df = pyb.statcast_pitcher(start, end, pitcher_id)

    if pitcher_df is None or pitcher_df.empty:
        print(f"No Statcast pitching data found for {args.player} in {args.season}.")
        raise SystemExit(1)

    print("Loading / building league-average comparison table...")
    league_df = load_or_build_league_grouped(start, end, cache_path=args.league_cache)

    print("Building dashboard...")
    fig = pitching_dashboard(
        pitcher_id=pitcher_id,
        df_pyb=df_pyb,
        df_statcast_group=league_df,
        season_label=season_label,
    )

    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Saved to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

