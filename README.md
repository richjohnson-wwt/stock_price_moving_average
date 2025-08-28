
Stock Signal Generator POC
====================

This is a simple stock signal generator that uses CUDA to calculate moving averages and generate buy/sell/hold signals based on the price relative to the moving average.

The main.cpp app generates a walking stock price and exports it to a CSV file.

The stock_signal.cu app uses the CSV file to generate buy/sell/hold signals based on the price relative to the moving average.

The stock_signal.cu app uses shared memory to calculate the moving average, which is faster than using global memory.

The stock_signal.cu app uses a moving average of 50 days.

Smoke testing from project root:

    g++ -o stock_sim src/main.cpp
    ./stock_sim

    nvcc -o stock_signal src/stock_signal.cu
    ./stock_signal


CLI arguments:
    ./stock_signal --window-size 30 --tolerance 2.0 --trade-increment 10
    ./stock_signal --tolerance 3.0 --window-size 25 --trade-increment 20
    ./stock_signal --tolerance 1.5 --window-size 35 --trade-increment 50
    ./stock_signal --csv-file other_stock.csv --tolerance 2.5
    ./stock_signal --help