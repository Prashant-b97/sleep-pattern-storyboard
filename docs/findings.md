# Sleep Pattern Findings

_Last refreshed: 2025-02-14_

## Data Overview

- Observation window: 2015-02-19 → 2021-12-31 (2,354 nights)
- Mean nightly sleep: **7.36 h** (median 6.82 h, σ ≈ 2.21 h)
- Lowest observed sleep: **1.27 h** (2016-02-19); highest: **17.43 h** (2018-05-21)
- Sub-5-hour nights: **72** (≈3.1% of the sample)
- Weekday average range: **7.16 h (Mon)** → **7.66 h (Sat)**
- Most recent 12 months average: **6.75 h**

## Diagnostic Highlights

- **Autocorrelation**: Daily sleep exhibits significant short-lag autocorrelation, justifying the AutoReg model choice. Inspect `primary_sleep_hours_acf*.png` for the spike pattern.
- **Seasonality**: Weekly periodicity dominates (7-day seasonality). After running the CLI, consult `primary_sleep_hours_decomposition.png` and `primary_sleep_hours_decomposition.csv`.
- **AutoReg Forecast**: With `--lags 30 --forecast-horizon 14`, the model provides a two-week projection saved as `primary_autoreg_forecast.csv`. Opening `run_summary.txt` reveals the next-day estimate.

## Narrative Insight

Overall sleep is stable around 7–7.5 h, with modest weekend increases. Outliers (e.g., 17 h recovery sessions) inflate variance; removing them (optional `--max-hours` future enhancement) would tighten spread. Less than 5 h occurs sporadically—flag for intervention.

## Next Steps

1. Capture a “curated run” by executing `sleep-analysis --run-id blog_showcase` and commit the resulting artefacts for reference.
2. Expand the report with rolling averages or circadian phase clustering.
3. Translate this summary into a Medium article—figure captions can reuse the generated PNG assets.

*Remember to re-run the CLI and refresh these stats whenever new sleep exports become available.*
