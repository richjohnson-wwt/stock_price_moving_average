
Stock Signal Generator POC
====================

This is a simple stock signal generator that uses CUDA to calculate moving averages and generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses the CSV file to generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses shared memory to calculate the moving average, which is faster than using global memory.

The stock_signal.cu app uses a configurable command line args to backtest the buy/sell/hold signals.

To download the stock data, use the following command:

    curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&outputsize=full&datatype=csv&apikey=<api_key>" -o tsla_intraday.csv

Smoke testing from project root:

    nvcc -o stock_signal src/stock_signal.cu
    ./stock_signal


CLI arguments:

    ./stock_signal --window-size 30 --tolerance 2.0 --trade-increment 10
    ./stock_signal --tolerance 3.0 --window-size 25 --trade-increment 20
    ./stock_signal --tolerance 1.5 --window-size 35 --trade-increment 50
    ./stock_signal --csv-file other_stock.csv --tolerance 2.5
    ./stock_signal --tolerance 3.0 --window-size 25 --trade-increment 20 --verbose
    ./stock_signal --help


# Bash script to run all combinations:

    for tol in 1.5 2.0 2.5 3.0 3.5; do
        for win in 20 25 30 35; do
            echo "=== Testing tolerance=$tol, window=$win ==="
            ./stock_signal --tolerance $tol --window-size $win
        done
    done

Best result is from tolerance=2.5 and window=30.

    ./stock_signal --window-size 30 --tolerance 2.5