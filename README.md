
Stock Signal Generator POC
====================

This is a simple stock signal generator that uses CUDA to calculate moving averages and generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses the CSV file to generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses shared memory to calculate the moving average, which is faster than using global memory.

The stock_signal.cu app uses a configurable command line args to backtest the buy/sell/hold signals.

The price_generator.cpp app generates a random stock price series based on a geometric Brownian motion model. The generated prices are only used in tests.

## What is Backtesting?

**Backtesting** is a financial/trading term that refers to testing a trading strategy or investment algorithm against historical market data to evaluate how it would have performed in the past. This is completely different from software testing:

- **Financial Backtesting**: Simulates trading decisions using historical price data to measure strategy performance (profit/loss, win rate, drawdown, etc.)
- **Software Testing**: Verifies that code functions correctly through unit tests, integration tests, etc.

In this project:
- **Backtesting** = Running our moving average trading strategy against historical TSLA price data to see if it would have been profitable
- **Unit Testing** = Verifying our C++/CUDA code works correctly (using Catch2 framework in the `test/` directory)

The backtesting results show metrics like:
- Total profit/loss from the trading strategy
- Number of winning vs losing trades
- Portfolio value over time
- Maximum drawdown (largest peak-to-trough decline)



## Intial Setup - Do every time a new VM is started

    uv venv
    source .venv/bin/activate
    uv pip install conan
    conan profile detect
    Install C++ and CMake Extensions in VSCode
    vi ~/.gitconfig

    [user]
        email = rich.johnson@wwt.com
        name = Rich Johnson


#### Debug Config

    conan install . --output-folder=build/debug --build=missing --settings=build_type=Debug
    cd build/debug 
    
    # All commands in build/debug
    cmake ../.. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug
    cmake --build .

    ./test/test_runner
    ctest
    ctest --verbose

#### Smoke testing from project root (not using CMake):

    nvcc -o stock_signal src/stock_signal.cu
    ./src/stock_signal


Usage:

    Usage: ./src/stock_signal [options]
    Options:
    --csv-file <filename>     CSV file path (default: src/tsla_intraday.csv)
    --window-size <int>       Moving average window size (default: 30)
    --tolerance <float>       Buy/sell tolerance (default: 2.0)
    --max-shares <int>        Maximum shares to own (default: 100)
    --trade-increment <int>   Shares per trade (default: 10)
    --verbose                 Show detailed trade logs (default: off)
    --help                    Show this help message

    ./src/stock_signal --window-size 30 --tolerance 2.0 --trade-increment 10
    ./src/stock_signal --tolerance 3.0 --window-size 25 --trade-increment 20
    ./src/stock_signal --tolerance 1.5 --window-size 35 --trade-increment 50
    ./src/stock_signal --csv-file other_stock.csv --tolerance 2.5
    ./src/stock_signal --tolerance 3.0 --window-size 25 --trade-increment 20 --verbose
    ./src/stock_signal --help
    ./src/stock_signal --csv-file stock_prices.csv
    ./src/stock_signal --csv-file nvda_intraday.csv --window-size 30 --tolerance 2.5


#### Bash script to run some permutations of the tolerance and window-size combinations:

    for tol in 1.5 2.0 2.5 3.0 3.5; do
        for win in 20 25 30 35; do
            echo "=== Testing tolerance=$tol, window=$win ==="
            ./src/stock_signal --tolerance $tol --window-size $win --csv-file nvda_intraday.csv
        done
    done

Best result is from tolerance=2.5 and window=30.

    ./src/stock_signal --window-size 30 --tolerance 2.5 --verbose

#### Data Directory

To download the stock data, use the following command:

Go to https://www.alphavantage.co/support/#api-key and sign up for a free API key.

    curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&outputsize=full&datatype=csv&apikey=<api_key>" -o tsla_intraday.csv

* data/stock_prices.csv  -  contains 2521 simulated stock prices from src/price_generator.cpp using a random walking simulation.

* data/tsla_intraday.csv -  was downloaded from alpha vantage and contains 21k intraday prices for TSLA over the month of August 2025.

* data/pacb_intraday.csv -  was downloaded from alpha vantage and contains 10k intraday prices for PACB over the month of August 2025.

* data/nvda_intraday.csv -  was downloaded from alpha vantage and contains 21k intraday prices for NVDA over the month of August 2025.
