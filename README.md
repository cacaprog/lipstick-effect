# 💄 Lipstick Effect Analysis

> *"When times get tough, people still buy lipstick."* — Leonard Lauder, 2001

An empirical data science project testing the **Lipstick Effect** hypothesis: do consumers increase spending on small, affordable luxuries (like cosmetics) during economic downturns?

---

## 🔬 Research Question

Does personal care consumption increase during economic recessions?

We measure this through three lenses:
- **Search interest** (Google Trends): `Lipstick` vs `Louis Vuitton`
- **Retail sales** (FRED `RSHPCS`): Personal Care Stores
- **Stock performance**: Estée Lauder (`EL`) vs S&P 500 (`^GSPC`)

---

## 📊 Data Sources

| Source | Series | Description |
|---|---|---|
| [FRED](https://fred.stlouisfed.org/) | `UNRATE` | Unemployment Rate |
| FRED | `PCE` | Personal Consumption Expenditures |
| FRED | `CPIAUCSL` | Consumer Price Index |
| FRED | `RSHPCS` | Retail Sales: Personal Care Stores |
| [Google Trends](https://trends.google.com/) | — | Search interest: Lipstick, Louis Vuitton, Mascara, Eyeliner, Perfume |
| [Yahoo Finance](https://finance.yahoo.com/) | `EL`, `^GSPC` | Estée Lauder & S&P 500 stock prices |

---

## 🗂️ Project Structure

```
lipstick-effect/
├── lipstick.py               # Main analysis script
├── lipstick.ipynb            # Jupyter notebook version
├── lipstick_25years.ipynb    # Extended 25-year analysis
├── correlation.ipynb         # Correlation deep-dive
├── img/                      # Saved plots (gitignored)
├── .env                      # API keys (gitignored)
└── .gitignore
```

---

## 📈 Analyses Performed

1. **Correlation Analysis** — Pearson correlations between unemployment and consumer behavior signals
2. **Time Series Visualization** — Unemployment vs lipstick search, retail sales, and stock performance
3. **Lagged Correlation Analysis** — Tests whether the lipstick effect has a delayed response (0–6 months lag)
4. **The Mascara Effect** — Beauty category shifts during COVID-19 mask-wearing period (April 2020 – May 2021)
5. **Stock Performance** — Estée Lauder vs S&P 500, normalized to index 100

---

## ⚙️ Configuration

The time range is controlled by two variables at the top of `lipstick.py`:

```python
START_DATE = '2020-01-01'
END_DATE   = '2024-12-31'
```

All output files are automatically tagged with the date range (e.g. `_2020_2024`) so results from different runs never overwrite each other.

---

## 🖼️ Output Files

Each run produces:

| Type | Naming pattern |
|---|---|
| Plots | `img/<chart_name>_{START}_{END}.png` (300 dpi) |
| Analysis summary | `analysis_summary_and_findings_{START}_{END}.txt` |

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone git@github.com:cacaprog/lipstick-effect.git
cd lipstick-effect
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas numpy requests fredapi pytrends matplotlib seaborn scipy python-dotenv yfinance
```

### 4. Set up your API key

Create a `.env` file in the project root:
```
FRED_API_KEY=your_key_here
```

Get a free key at → https://fred.stlouisfed.org/docs/api/api_key.html

### 5. Run the analysis
```bash
python lipstick.py
```

---

## 🔑 Key Findings

> All correlations are statistically significant (p < 0.0001, 2008–2025 range).

| Relationship | Correlation | Direction |
|---|---|---|
| Unemployment → Lipstick Search | **-0.33** | Unemployment ↑, search ↓ |
| Unemployment → Lipstick/Luxury Ratio | **-0.34** | Unemployment ↑, ratio ↓ |
| Unemployment → Retail Sales (Personal Care) | **-0.64** | Unemployment ↑, sales ↓ |
| Unemployment → Estée Lauder Stock | **-0.48** | Unemployment ↑, stock ↓ |
| Estée Lauder vs S&P 500 | **+0.60** | Moves broadly with the market |

- ❌ **The data does not support the classic Lipstick Effect**: all key correlations are *negative* — when unemployment rises, lipstick search interest, the lipstick/luxury ratio, and personal care retail sales all *decline* rather than increase.
- 📉 **Retail sales are the strongest signal**: the `-0.64` correlation (lagged up to 6 months, reaching `-0.67`) is the most robust relationship in the dataset.
- 💄 **The Mascara Effect (COVID-19)**: during the mask-wearing period (April 2020 – May 2021), mascara search interest rose **+28%** and eyeliner **+20.4%** — both statistically significant — while lipstick dropped **-13.4%** (not significant). This confirms a *category substitution* effect driven by context, not economics.
- 📈 **Estée Lauder vs S&P 500** (2008–2025): Estée Lauder returned **+548.9%** vs S&P 500's **+397.1%**, a **+151.9 percentage point** outperformance, though this reflects long-term brand strength more than recession resilience.


---

## 📦 Dependencies

- `pandas`, `numpy`, `scipy` — data manipulation & statistics
- `fredapi` — FRED economic data
- `pytrends` — Google Trends
- `yfinance` — stock market data
- `matplotlib`, `seaborn` — visualizations
- `python-dotenv` — environment variable management

---

## 📄 License

MIT
