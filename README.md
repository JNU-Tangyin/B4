# B4: Learning Bull-Bear Market Dynamics with Contrastive Modeling

This project provides a deep learning framework for multi-view financial text and price prediction. It supports various mainstream methods (such as B4, LSTM, StockNet, ESPMP, etc.) and can be used for joint modeling of stock market news text and price data, topic analysis, and investment strategy evaluation.

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
- Topic analysis and industry distribution visualization
- Investment strategy backtesting and financial metrics output
- Automatic generation of LaTeX tables

## Parameter Description

Main parameters can be configured in [`B4/globals.py`] and [`B4/main.py`], including:

- `dataset_dir`: Dataset root directory
- `dataset_type`: Dataset type (e.g., USStock/, CNStock/, SP500/)
- `lookback` / `lookahead`: Window length and prediction steps
- `series_type` / `text_type`: Input types
- `method`: Selected model method
- Other hyperparameters (alpha, temp, losspull, etc.)

## Output

- Training and testing results are saved in the [`result/`] directory
- Topic analysis results: `bull_bear_plot(compare&ablation)/`, etc.
- LaTeX table outputs: see `latex/` directory

## Acknowledgements

This project implements and refers to various financial text analysis and multi-modal fusion methods. Thanks to the open-source community for their contributions.

---

For questions or suggestions, please contact the author or submit an issue.