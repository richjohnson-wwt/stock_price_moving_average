#include <iostream>
#include <cstring>
#include "stock_signal.cuh"



// -----------------------------
// CLI Argument Parsing
// -----------------------------
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --csv-file <filename>     CSV file path (default: src/tsla_intraday.csv)\n";
    std::cout << "  --window-size <int>       Moving average window size (default: 30)\n";
    std::cout << "  --tolerance <float>       Buy/sell tolerance (default: 2.0)\n";
    std::cout << "  --max-shares <int>        Maximum shares to own (default: 100)\n";
    std::cout << "  --trade-increment <int>   Shares per trade (default: 10)\n";
    std::cout << "  --verbose                 Show detailed trade logs (default: off)\n";
    std::cout << "  --help                    Show this help message\n";
}

Config parse_args(int argc, char* argv[]) {
    Config config;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "--csv-file") == 0 && i + 1 < argc) {
            config.csv_filename = std::string(DATA_DIR) + "/" + argv[++i];
        }
        else if (strcmp(argv[i], "--window-size") == 0 && i + 1 < argc) {
            config.window_size = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            config.tolerance = std::atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-shares") == 0 && i + 1 < argc) {
            config.max_shares = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--trade-increment") == 0 && i + 1 < argc) {
            config.trade_increment = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--verbose") == 0) {
            config.verbose = true;
        }
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}