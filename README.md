## Enhancing Portfolio Performance with the Long-Short Beta Trading Strategy

### ABSTRACT

This study presents a novel long-short beta trading strategy aimed at boosting portfolio performance by exploiting relative stock market mispricings. Using stock data from the National Stock Exchange of India (NSE), we construct a beta-neutral portfolio by going long on stocks with low beta values and shorting those with high beta values. The performance of this strategy is compared to an equal-weighted benchmark, with both strategies being analyzed in terms of their Sharpe ratios. Our results demonstrate that the long-short beta trading strategy outperforms the equal-weighted benchmark, offering investors a valuable tool to enhance their returns while mitigating risks.

### Code Process Explanation:

1. Import necessary libraries and modules.
2. Define file paths, tickers, and dates for the stock data.
3. Retrieve stock data and process it by calculating adjusted closing prices and returns.
4. Prepare Fama-French factors by downloading data from the Fama-French dataset.
5. Store the stock data and Fama-French factors in an HDF5 file for future use.
6. Calculate end-of-month dates and create a placeholder for beta values and portfolio weights.
7. Estimate beta values for each stock using rolling 12-month windows and the market return factor from Fama-French data.
8. Calculate portfolio weights based on median beta values, going long on low-beta stocks and short on high-beta stocks.
9. Backtest the long-short beta trading strategy by calculating daily portfolio values and rebalancing at the end of each month.
10. Calculate equal-weighted portfolio values for comparison.
11. Calculate the Sharpe ratio for the long-short beta trading strategy and the equal-weighted benchmark.
12. Plot the performance of the long-short beta strategy, the equal-weighted benchmark, and the index.
13. Print portfolio weights and units to be purchased for both strategies on the last day.

By following this code process, we can analyze and compare the performance of the long-short beta trading strategy and the equal-weighted benchmark, thereby demonstrating the effectiveness of our proposed strategy in enhancing portfolio performance.
