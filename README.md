# Pitching Summary Graphic (CLI-Enabled Fork)

This project generates TJStats-style pitching summary dashboards using MLB Statcast, FanGraphs, and related data.

This fork adds:

- A **command-line interface (`ps_cli.py`)** to generate dashboards for any pitcher.
- Flexible **date filters** (by season or explicit start/end dates).
- **Postseason-only** or **regular-season-only** views.
- Optional **team-based filters** (e.g. only games vs/with LAD).
- Dynamic **league-average comparison** table (with caching).
- Basic **pytest**-based tests for core helpers.

The goal is to make it easy to analyze pitchers throughout the season (or in specific slices) without touching the notebook.

---

## Example Output

Example layout (from the original project, style preserved in this fork):

![alt text](images/output.png)

---

## CLI Usage

From the repo root (inside your virtual environment):

```bash
python ps_cli.py --help
```

Output:

```text
usage: ps_cli.py [-h] [--season SEASON] [--start-date START_DATE] [--end-date END_DATE]
                 [--postseason] [--regular] [--team-filter TEAM_FILTER]
                 [--out OUT_PATH] [--league-cache LEAGUE_CACHE] [--verbose]
                 pitcher

Generate TJStats-style pitching summary dashboards from Statcast.

positional arguments:
  pitcher               Pitcher name, e.g. "Tarik Skubal" or "Shohei Ohtani"

options:
  -h, --help            show this help message and exit
  --season SEASON       Season year (e.g. 2024). Used for defaults & postseason windows.
  --start-date START_DATE
                        Start date (YYYY-MM-DD). Overrides --season default.
  --end-date END_DATE   End date (YYYY-MM-DD). Overrides --season default.
  --postseason          Use only postseason games for the given season.
  --regular             Use only regular-season games for the given season.
  --team-filter TEAM_FILTER
                        Optional: limit to games where this team (MLB abbrev) is either
                        the pitcher's team or the opponent. Example: --team-filter LAD
  --out OUT_PATH, -o OUT_PATH
                        Output PNG file. If omitted, shows the figure interactively.
  --league-cache LEAGUE_CACHE
                        CSV path to cache league-wide grouped metrics.
  --verbose, -v         Verbose logging.
```

### Common Examples

Generate a full-season dashboard:

```bash
python ps_cli.py "Yoshinobu Yamamoto" --season 2025 --out images/yamamoto2025.png
```

Postseason-only for a pitcher:

```bash
python ps_cli.py "Shohei Ohtani" --season 2025 --postseason --out images/ohtani_2025_post.png
```

Filter to games involving a specific team (either on his team or opponent):

```bash
python ps_cli.py "Blake Snell" --season 2025 --postseason --team-filter LAD   --out images/snell_lad_post2025.png
```

Custom date window:

```bash
python ps_cli.py "Tarik Skubal" --start-date 2025-04-01 --end-date 2025-06-30   --out images/skubal_apr_jun_2025.png
```

If `--out` is omitted, the figure will be displayed instead of written to disk.

League-wide comparison metrics are automatically loaded from `statcast_2024_grouped.csv`
when appropriate, or dynamically built & cached via `--league-cache` for other windows.

---

## Article / Original Notebook

For methodology details and the original implementation, see:

- Original notebook: `pitcher_summary.ipynb`
- Original article (by @tnestico): Medium – *Creating the Perfect Pitching Summary*

This fork keeps the original layout/visual identity while exposing a scriptable interface.

---

## Requirements

### Python Version

You **must** use:

- **Python 3.9** or **Python 3.10**

Python **3.11+ is not supported** at this time due to upstream dependency compatibility (notably `pybaseball` and plotting stack versions).

### Recommended Versions / Packages

Install via:

```bash
pip install -r requirements.txt
```

The environment should include (or be compatible with):

```text
pandas==1.5.2
numpy==1.23.5
seaborn==0.11.1
pybaseball==2.2.7
matplotlib==3.5.1
Pillow==10.3.0
requests==2.31.0
pytest>=8.0.0   # for running tests (optional)
```

---

## Running Tests

From the repo root:

```bash
pytest -v
```

This runs basic sanity checks on:

- Statcast dataframe processing helpers.
- ERA math and summary line construction.
- A smoke test that `pitching_dashboard` runs end-to-end on fake data.

---

## Contributing

Feel free to:

- Open issues or PRs against this fork.
- Open a PR from this fork back to the original `tnestico/pitching_summary` repo
  if you’d like these CLI and filtering features to be upstreamed.

Layout credit: **@TJStats**  
CLI wrapper, filters & tests: **@spaceshiptrip**
