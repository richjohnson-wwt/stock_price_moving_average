#pragma once

#include <string>
#include <vector>



/*
    CSV File reader: implementation in src/csv_reader.cpp
*/
struct Config {
    std::string csv_filename = std::string(DATA_DIR) + "/tsla_intraday.csv";
    int window_size = 30;
    float tolerance = 2.0f;
    int max_shares = 100;
    int trade_increment = 10;
    bool verbose = false;
};
bool read_prices_from_csv(const std::string& filename, std::vector<float>& prices_out);



/*
    Backtesting functions: implementation in src/backtest_runner.cpp
*/
struct BacktestResult {
    float final_portfolio_value;
    float total_return;
    float max_drawdown;
    int total_trades;
    int buy_trades;
    int sell_trades;
    int final_shares_owned;
    float final_cash;
};

// Pure calculation logic (testable, no I/O)
BacktestResult calculate_backtest(const float* prices, const int* signals, 
    int N, int window_size, int max_shares, int trade_increment);

// Display wrapper (calls calculate_backtest + prints results)
void run_backtest(const float* prices, const int* signals, const float* moving_avg, 
    int N, int window_size, int max_shares, int trade_increment, bool verbose);



/*
    CLI argument parsing: implementation in src/cli.cpp
*/
void print_usage(const char* program_name);
Config parse_args(int argc, char* argv[]);