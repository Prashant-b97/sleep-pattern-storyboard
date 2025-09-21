# 🌙 Sleep Pattern Storyboard  

> _From messy CSVs ➝ clean insights ➝ interactive dashboard_  

A personal sleep diary that gradually transformed into a **data analysis toolkit** and **storytelling dashboard**.  
This repository documents the journey—from raw exports to repeatable workflows and an interactive visualization app.  

---

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/streamlit-app-red.svg?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/tests-passing-brightgreen.svg?style=for-the-badge&logo=pytest" alt="Tests">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg?style=for-the-badge" alt="License">
</p>

---

## ✨ Highlights  

- **Beginner-friendly:** Guided walkthroughs that explain metrics in plain English.  
- **Analyst-ready:** Deep-dive tables, models, and forecasting tools.  
- **Reproducible & tested:** CLI workflows, automated charts, and ingestion layer unit tests.  

---

## 📸 Dashboard Preview  

<p align="center">
  <img src="docs/assets/dashboard-hero-1.png" width="75%">
</p>
<p align="center">
  <img src="docs/assets/dashboard-hero-2.png" width="75%">
</p>
<p align="center">
  <img src="docs/assets/dashboard-hero-3.png" width="75%">
</p>

---

## 🛠️ Act I – From Export to Evidence  

- `sleep-analysis` **CLI**: cleans tracker dumps, runs sanity checks, and saves results in timestamped folders.  
- Generates **charts, tables, forecasts,** and a plain-English `run_summary.txt`.  
- Uses **seasonal decomposition + AutoReg forecasting** to detect volatility or drift.  
- Handles **secondary session data** (start/end times, devices) for richer context.  

### Quickstart  

```bash
# Create environment
conda create -n sleep-analysis python=3.10
conda activate sleep-analysis

# Install in dev mode
pip install -e '.[dev]'

# Run analysis
sleep-analysis --output-dir analysis_output --run-id first_story --lags 21 --forecast-horizon 14
```

---

## 📊 Act II – Storytelling Dashboard  

Run the interactive app:  

```bash
streamlit run streamlit_app.py -- --primary sleep_data.csv --secondary sleep_data_new.csv
```

### Features:
- **Beginner Walkthrough** → explains metrics in simple language + highlights healthy-night streaks.  
- **Analyst Deep Dive** → variance tables, downloads, and model tuning.  
- **Control Room Sidebar** → adjust AutoReg lags/horizon live + export cleaned dataset.  

🎥 **Demo:**

<p align="center">
  <img src="docs/assets/dashboard-story.gif" width="75%">
</p>

---

## 📖 Act III – Documentation & Reliability  

- Evolving findings: `docs/findings.md` → key insights, highlights, and next steps.  
- Testing: `tests/test_loaders.py` ensures data ingestion is consistent.  
- Packaging: `pyproject.toml` makes it installable with dev extras (`pytest`, `ruff`).  

---

## 🗂️ Repository Map
```
.
├── pyproject.toml
├── README.md
├── streamlit_app.py                # Interactive dashboard
├── sleep_pattern_analysis.py       # CLI entry (legacy support)
├── src/
│   └── sleep_analysis/
│       ├── __init__.py
│       └── cli.py                  # Core CLI logic
├── tests/
│   └── test_loaders.py             # Loader unit tests
├── docs/
│   ├── findings.md                 # Insight documentation
│   └── assets/                     # Screenshots & GIFs
├── sleep_data.csv
└── sleep_data_new.csv
```

---

## 💡 Why This Repo Stands Out

- **Data storytelling:** Clear progression from raw exports ➝ structured insights.  
- **Engineering practices:** Clean packaging, testing, reproducible CLI.  
- **Educational design:** Beginners can follow step-by-step, while analysts can deep dive.  
- **Practical value:** Shows how personal data can evolve into a reliable analysis framework.  

<p align="center"> 🌙 <b>Sleep better, ship better.</b> </p>
