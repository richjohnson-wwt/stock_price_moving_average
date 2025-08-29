
Stock Signal Generator POC
====================

This is a simple stock signal generator that uses CUDA to calculate moving averages and generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses the CSV file to generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses shared memory to calculate the moving average, which is faster than using global memory.

The stock_signal.cu app uses a configurable command line args to backtest the buy/sell/hold signals.

The price_generator.cpp app generates a random stock price series based on a geometric Brownian motion model. The generated prices are only used in tests.

#### Intial Setup - Do every time a new VM is started

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


#### Bash script to run some permutations of the tolerance and window-size combinations:

    for tol in 1.5 2.0 2.5 3.0 3.5; do
        for win in 20 25 30 35; do
            echo "=== Testing tolerance=$tol, window=$win ==="
            ./src/stock_signal --tolerance $tol --window-size $win
        done
    done

Best result is from tolerance=2.5 and window=30.

    ./src/stock_signal --window-size 30 --tolerance 2.5 --verbose

#### Data Directory

    To download the stock data, use the following command:

    curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=TSLA&interval=1min&outputsize=full&datatype=csv&apikey=<api_key>" -o tsla_intraday.csv

data/stock_prices.csv    contains 2521 simulated stock prices from src/price_generator.cpp using a random walking simulation.
data/tsla_prices.csv     was downloaded from alpha vantage and contains 3.8k daily open prices for TSLA.
data/tsla_intraday.csv   was downloaded from alpha vantage and contains 21k intraday prices for TSLA over the month of August 2025.