# Enhancing Portfolio Performance with the Long-Short Beta Trading Strategy

## Introduction
The long-short beta trading strategy presented in this repository is designed to capitalize on market inefficiencies by constructing a beta-neutral portfolio. The strategy goes long on stocks with low beta values and short on those with high beta values, aiming to exploit relative mispricings while mitigating market risk.

The performance of this strategy is compared to an equal-weighted benchmark, which simply assigns equal weights to all stocks in the portfolio.

## Installation
To use this code, you need to have Python 3.x installed on your system along with the following libraries:
```
numpy
pandas
datetime
matplotlib
yfinance
```
You can install the required libraries using pip:
```python
pip install numpy pandas datetime matplotlib statsmodels yfinance
```
## Usage
1. Clone this repository to your local machine:
```git
git clone https://github.com/yourusername/Long-Short-Beta-Trading-Strategy.git
```
2. Open the long_short_beta_trading.py script in your favorite code editor and update the file paths and tickers as needed.
3. Run the script:
```python
python long_short_beta_trading.py
```
The script will download the stock data, estimate beta values, and calculate portfolio weights and values for both the long-short beta trading strategy and the equal-weighted benchmark. The Sharpe ratios and portfolio weights will be displayed in the console.

## Results
The long-short beta trading strategy demonstrates improved performance compared to the equal-weighted benchmark. The Sharpe ratios of the two strategies indicate that the long-short beta trading strategy is more effective at enhancing returns while mitigating risks.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
