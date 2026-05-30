# B4: Bias to Behavior from Bull-Bear Dynamics

This repository implements **B4** (Bias to Behavior from Bull-Bear Dynamics), a deep learning framework for multi-view financial text and price prediction with causal contrastive modeling. It supports various mainstream methods (such as B4, CE, SupCon, InfoNCE, StockNet, ALSTM, etc.) and can be used for joint modeling of stock market news text and price data, topic analysis, and investment strategy evaluation.

## Key Design Choices (Aligned with Paper)

- **Directional Label**: The prediction target is **next-day price direction** (`close_{t+1} > close_t` → bullish, otherwise bearish), *not* a future-window ranking.
- **Causal Inertial Pairing (IP)**: The contrastive positive/negative sampling in `loss_func.py` strictly uses **past time steps only** (`j < i`). No look-ahead bias.
- **Long-Flat Trading Protocol**: Positions are `{1=long, 0=flat}`. The model does not take short positions.
- **Friction-Aware Evaluation**: Transaction costs (5 bps per side) and slippage (5 bps per side) are deducted on every position change in `evaluate.py`.
- **LookAhead = 1**: Default horizon is next-day (`globals.py`).

## Directory Structure

```
B4/
├── evaluate.py            # Evaluation metrics and evaluators
├── globals.py             # Global parameters and configuration
├── lda.ipynb              # LDA topic analysis
├── load_data.py           # Data loading utilities
├── loss_func.py           # Loss function definitions
├── main.py                # Main training and testing entry
├── plot.py                # Plotting utilities
├── post.py                # Post-processing and table generation
├── topic.py               # Topic-related processing
├── dataset/               # Dataset directory
│   └── USStock/
│       ├── news/          # US stock news data
│       └── price/         # US stock price data
├── exp/                   # Experiment scripts
│   ├── exp_baseline.py
│   └── exp_triviews.py
├── generator/             # Data generator
│   └── generator.py
├── methods/               # Model methods
│   ├── alstm.py
│   ├── b4.py
│   ├── bert.py
│   ├── espmp.py
│   ├── indexgan.py
│   ├── stocknet.py
│   ├── taureau.py
│   ├── tradeevents.py
│   └── transam.py
├── preprocess/            # Data preprocessing
│   ├── price_process.py
│   └── text_process.py
└── README.md
```

## Requirements

- Python 3.9+
- PyTorch
- transformers
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Other dependencies are listed at the top of each script

It is recommended to use `conda` or `venv` to create a virtual environment.

## Quick Start

1. **Prepare Data**  
   Place news and price data into [`dataset/USStock/news/`] and [`dataset/USStock/price/`] directories, or modify the data paths in [`B4/globals.py`] as needed.

2. **Train and Test**  
   For example, to run on US stock data:

   ```sh
   python main.py
   ```

   You can modify parameters (such as dataset, method, epochs, etc.) at the end of [`B4/main.py`].

3. **Result Analysis and Visualization**  
   - Topic analysis: Run [`B4/lda.ipynb`] or `bertopic.ipynb`.
   - Post-processing and table generation: Run [`B4/post.py`].
   - Plotting and visualization: See `plot.py` or related notebooks.

## Supported Methods

- B4 (multi-view fusion)
- LSTM/Attentive-LSTM
- StockNet
- Transformer
- ESPMP
- Topic modeling (LDA, BERTopic)

## Main Features

- Joint modeling of news text and price data
- Training and evaluation of various deep learning models
- **Causal contrastive learning**: Inertial Pairing uses strictly past samples to avoid look-ahead bias
- **Long-flat backtesting**: No short-selling; positions are either long or flat
- **Friction-aware evaluation**: Deducts 10 bps round-trip cost + 5 bps slippage per side on every trade
- **Factor-adjusted alpha**: Supports FF3 / Carhart 4-factor regression (supply `factors_df` to `Evaluator.save()`)
- Topic analysis and industry distribution visualization
- Automatic generation of LaTeX tables

## Parameter Description

Main parameters can be configured in [`globals.py`] and [`main.py`], including:

- `dataset_dir`: Dataset root directory
- `dataset_type`: Dataset type (e.g., `USStock/`, `CNStock/`, `SP500/`)
- `lookback`: Historical window length for time-series input (default: 20)
- `lookahead`: **Must be 1** for next-day directional labels. Do not change.
- `series_type` / `text_type`: Input types (`price` or `indicators`, `news`)
- `method`: Selected model method (`b4`, `ce`, `scl`, `infonce`, etc.)
- `alpha`: Weight balancing CE loss and contrastive losses in `DualLoss` (default: 0.5)
- `temp`: Temperature for NT-Xent contrastive loss (default: 0.1)
- `losspull`: **Causal backward window size Δ** for Inertial Pairing. Must be a non-negative integer (`0, 1, 2, 3`). Positive values define how many *past* steps are included as inertial positives. No future steps are ever used.

## Output

- Training and testing results are saved in the [`result/`] directory
- Evaluation metrics per stock include:
  - `B&H`: Buy-and-Hold cumulative return
  - `Strategy_net`: Strategy cumulative return after costs
  - `sharpe`: Annualised Sharpe ratio (net returns)
  - `sortino`: Annualised Sortino ratio
  - `turnover`: Average daily position-change frequency
  - `max_drawdown`: Maximum drawdown
  - `acc`: Directional accuracy
  - `ff3_alpha` / `carhart_alpha`: Factor-adjusted alphas (if factor data provided)
- Topic analysis results: `bull_bear_plot(compare&ablation)/`, etc.
- LaTeX table outputs: see `latex/` directory

## Acknowledgements

This project implements and refers to various financial text analysis and multi-modal fusion methods. Thanks to the open-source community for their contributions.

---

For questions or suggestions, please contact the author or submit an issue.