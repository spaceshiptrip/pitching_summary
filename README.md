# ⚾ Pitching Summary CLI (Enhanced TJStats Edition)

This project builds on the original "TJStats" Pitching Summary by adding a **Command-Line Interface (CLI)** for generating custom pitching dashboards for any MLB pitcher, any season — directly from Statcast data.  

The CLI version supports automatic Statcast retrieval, caching, and saves output as high-quality summary graphics.

---

## 🧠 Overview

Originally developed by [Thomas Nestico](https://github.com/tnestico/pitching_summary), this framework visualizes advanced pitch metrics such as movement, velocity, whiff rate, xwOBA, chase rate, and league comparison.

The new CLI extension allows analysts, fans, and developers to:
- Generate summaries without Jupyter or manual notebook editing
- Quickly compare pitchers across seasons
- Cache league-wide stat averages for faster re-use
- Run headless, making it suitable for automation or TUI integration

---

## 🚀 Example CLI Usage

```bash
# Activate your virtual environment
source .venv/bin/activate

# Run the CLI to analyze a pitcher
python ps_cli.py "Shohei Ohtani" --season 2025 --out ohtani_2025.png
```

**Output:**  
- A `.png` dashboard summarizing pitch usage, whiff/chase rates, and run-value impact  
- Cached Statcast and league data for faster subsequent runs

---

## 🧩 Example Output

![alt text](images/output.png)

---

## 📰 Article

Refer to both the [.IPYNB file](https://github.com/tnestico/pitching_summary/blob/main/pitcher_summary.ipynb) and the accompanying  
[Medium Article](https://medium.com/@thomasjamesnestico/creating-the-perfect-pitching-summary-7b8a981ef0c5)  
for methodology and visualization breakdown.

---

## ⚙️ Requirements

#### Python Versions
✅ Works with **Python 3.9** or **3.10**  
⚠️ *Not compatible with Python 3.11+ due to `matplotlib` and `pybaseball` dependency issues.*

#### Required Packages
```
pandas==1.5.2
numpy==1.23.5
seaborn==0.11.1
pybaseball==2.2.7
matplotlib==3.5.1
PIL.Image==10.3.0
requests==2.31.0
```

To install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧰 CLI Options

| Option | Description |
|--------|--------------|
| `--season` | Year to fetch Statcast data (e.g., `--season 2024`) |
| `--out` | Output filename for the image (e.g., `--out skubal_2024.png`) |
| `--league-cache` | (Optional) Path to cached league averages CSV |
| `--help` | Show available arguments |

Example:
```bash
python ps_cli.py "Tarik Skubal" --season 2024 --out skubal_2024.png
```

---

## ⚾ Methodology Highlights

Each summary includes:
- **Pitch usage %**
- **Velocity & spin rate**
- **Movement (pfx_z / pfx_x)**
- **Whiff / chase / in-zone rates**
- **xwOBA and delta run expectancy**
- **Comparison to league-average baselines**

---

## 🧱 Caching

Statcast requests can be **slow** — enable caching to save partial progress and speed up future runs:

```python
from pybaseball import cache
cache.enable()
```

This will cache data locally in `~/.pybaseball/`.

---

## 🧾 License

MIT © 2025 — Forked and CLI-enhanced by [spaceshiptrip](https://github.com/spaceshiptrip)  
Original concept and notebook by [Thomas Nestico](https://github.com/tnestico)
