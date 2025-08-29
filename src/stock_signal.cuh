#pragma once

#include <string>
#include <vector>

// CSV file reader
struct Config {
    std::string csv_filename = std::string(DATA_DIR) + "/tsla_intraday.csv";
    int window_size = 30;
    float tolerance = 2.0f;
    int max_shares = 100;
    int trade_increment = 10;
    bool verbose = false;
};
bool read_prices_from_csv(const std::string& filename, std::vector<float>& prices_out);

// Backtesting function
void run_backtest(const float* prices, const int* signals, const float* moving_avg, 
    int N, int window_size, int max_shares = 100, int trade_increment = 10, bool verbose = false);

// CLI argument parsing
void print_usage(const char* program_name);
Config parse_args(int argc, char* argv[]);