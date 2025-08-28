// stock_signal.cu
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>
#include <cstring>
#include <cuda_runtime.h>

// -----------------------------
// CUDA Kernel: Moving Average (Global Memory) - now unused in favor of shared memory
// -----------------------------
__global__ void moving_average_kernel_global_memory(const float* prices, float* moving_avg, int N, int window_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= window_size - 1 && idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < window_size; i++) {
            sum += prices[idx - i];
        }
        moving_avg[idx] = sum / window_size;
    } else if (idx < N) {
        moving_avg[idx] = 0.0f; // Not enough data
    }
}

// -----------------------------
// CUDA Kernel: Moving Average (Shared Memory Optimized)
// -----------------------------
__global__ void moving_average_kernel(const float* prices, float* moving_avg, int N, int window_size) {
    extern __shared__ float shared_prices[];
    
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    int block_start = blockIdx.x * blockDim.x;
    
    // Calculate how much data this block needs
    int block_end = min(block_start + blockDim.x - 1, N - 1);
    int data_start = max(0, block_start - window_size + 1);
    int data_end = block_end;
    int shared_size = data_end - data_start + 1;
    
    // Cooperatively load data into shared memory
    // Each thread may load multiple elements if needed
    for (int i = local_idx; i < shared_size; i += blockDim.x) {
        int global_data_idx = data_start + i;
        if (global_data_idx < N) {
            shared_prices[i] = prices[global_data_idx];  // each thread moves a little part from global to shared
        }
    }
    
    __syncthreads();  // brief pause is tiny price to pay for shared memory access.
    // Like everyone waiting 1 second at traffic light to avoid 10 minute traffic jam
    
    // Compute moving average using shared memory
    if (global_idx < N) {
        if (global_idx >= window_size - 1) {
            float sum = 0.0f;
            for (int i = 0; i < window_size; i++) {
                int shared_offset = (global_idx - i) - data_start;
                if (shared_offset >= 0 && shared_offset < shared_size) {
                    sum += shared_prices[shared_offset];  // Using data loaded by other threads in the same block
                } else {
                    // Fallback to global memory if not in shared
                    sum += prices[global_idx - i];  // This line gets the 50 previous prices from global memory if not in shared memory
                }
            }
            moving_avg[global_idx] = sum / window_size;
        } else {
            moving_avg[global_idx] = 0.0f; // Not enough data
        }
    }
}

// --------------------------------------
// CUDA Kernel: Price vs Moving Average
// --------------------------------------
__global__ void detect_signals_combined(const float* prices, const float* moving_avg, int* signals, int N, float tolerance) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float price = prices[idx];
        float ma = moving_avg[idx];

        if (price > ma + tolerance) {  // If no tolerance, then we never got a HOLD signal
            signals[idx] = 1;  // BUY
        } else if (price < ma - tolerance) {
            signals[idx] = -1; // SELL
        } else {
            signals[idx] = 0;  // HOLD (within tolerance)
        }
    }
}


// -------------------------------------
// Read prices from CSV
// -------------------------------------
bool read_prices_from_csv(const std::string& filename, std::vector<float>& prices_out) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open CSV file: " << filename << std::endl;
        return false;
    }

    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false; // Skip header
            continue;
        }

        std::stringstream ss(line);
        std::string day_str, price_str;

        if (!std::getline(ss, day_str, ',') || !std::getline(ss, price_str)) {
            continue;
        }

        try {
            float price = std::stof(price_str);
            prices_out.push_back(price);
        } catch (...) {
            std::cerr << "Warning: Skipping invalid line: " << line << std::endl;
        }
    }

    return true;
}

// -----------------------------
// Configuration Structure
// -----------------------------
struct Config {
    std::string csv_filename = "src/tsla_intraday.csv";
    int window_size = 30;
    float tolerance = 2.0f;
    int max_shares = 100;
    int trade_increment = 10;
};

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
            config.csv_filename = argv[++i];
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
        else {
            std::cerr << "Unknown argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            exit(1);
        }
    }
    
    return config;
}

// -----------------------------
// CPU-Based Portfolio Backtesting Function
// -----------------------------
void run_backtest(const float* prices, const int* signals, const float* moving_avg, 
                  int N, int window_size, int max_shares = 100, int trade_increment = 10) {
    
    // Initialize portfolio state
    int shares_owned = 0;
    float initial_price = prices[window_size - 1];  // First price we can trade at
    float starting_cash = max_shares * initial_price;  // Equivalent cash value
    float cash = starting_cash;
    
    // Performance tracking
    int total_trades = 0;
    int buy_trades = 0;
    int sell_trades = 0;
    float max_portfolio_value = starting_cash;
    float min_portfolio_value = starting_cash;
    
    std::cout << "\n--- Portfolio Backtesting ---\n";
    std::cout << "Starting with: $" << starting_cash << " cash (equivalent to " << max_shares << " shares @ $" << initial_price << ")\n";
    std::cout << "Trade size: " << trade_increment << " shares\n\n";
    
    // Process each trading signal
    for (int i = window_size - 1; i < N; i++) {
        float current_price = prices[i];
        float current_portfolio_value = cash + (shares_owned * current_price);
        
        // Track portfolio extremes
        if (current_portfolio_value > max_portfolio_value) {
            max_portfolio_value = current_portfolio_value;
        }
        if (current_portfolio_value < min_portfolio_value) {
            min_portfolio_value = current_portfolio_value;
        }
        
        // Execute trades based on signals
        if (signals[i] == 1 && shares_owned < max_shares) {  // BUY signal
            int shares_to_buy = std::min(trade_increment, max_shares - shares_owned);
            float cost = shares_to_buy * current_price;
            
            if (cash >= cost) {  // Check if we have enough cash
                shares_owned += shares_to_buy;
                cash -= cost;
                total_trades++;
                buy_trades++;
                
                std::cout << "Day " << i << ": BUY  " << shares_to_buy << " shares @ $" << current_price 
                         << " | Portfolio: $" << (cash + shares_owned * current_price) 
                         << " (" << shares_owned << " shares, $" << cash << " cash)\n";
            }
        }
        else if (signals[i] == -1 && shares_owned > 0) {  // SELL signal
            int shares_to_sell = std::min(trade_increment, shares_owned);
            float proceeds = shares_to_sell * current_price;
            
            shares_owned -= shares_to_sell;
            cash += proceeds;
            total_trades++;
            sell_trades++;
            
            std::cout << "Day " << i << ": SELL " << shares_to_sell << " shares @ $" << current_price 
                     << " | Portfolio: $" << (cash + shares_owned * current_price) 
                     << " (" << shares_owned << " shares, $" << cash << " cash)\n";
        }
        // HOLD: do nothing, just track portfolio value
    }
    
    // Calculate final performance metrics
    float final_price = prices[N - 1];
    float final_portfolio_value = cash + (shares_owned * final_price);
    float total_return = ((final_portfolio_value - starting_cash) / starting_cash) * 100.0f;
    float max_drawdown = ((max_portfolio_value - min_portfolio_value) / max_portfolio_value) * 100.0f;
    
    std::cout << "\n--- Final Portfolio Results ---\n";
    std::cout << "Final Portfolio Value: $" << final_portfolio_value << "\n";
    std::cout << "Starting Value: $" << starting_cash << "\n";
    std::cout << "Total Return: " << total_return << "%\n";
    std::cout << "Maximum Drawdown: " << max_drawdown << "%\n";
    std::cout << "Total Trades: " << total_trades << " (" << buy_trades << " buys, " << sell_trades << " sells)\n";
    std::cout << "Final Position: " << shares_owned << " shares, $" << cash << " cash\n\n";

    // Output first 20 trading signals for verification
    std::cout << "--- First 20 Trading Signals (for verification) ---\n";
    int signals_shown = 0;
    for (int i = window_size - 1; i < N && signals_shown < 20; i++) {
        if (signals[i] == 1) {
            std::cout << "Day " << i << ": BUY  @ Price = " << prices[i]
                    << ", MA = " << moving_avg[i] << std::endl;
        } else if (signals[i] == -1) {
            std::cout << "Day " << i << ": SELL @ Price = " << prices[i]
                    << ", MA = " << moving_avg[i] << std::endl;
        } else {
            std::cout << "Day " << i << ": HOLD @ Price = " << prices[i]
                    << ", MA = " << moving_avg[i] << std::endl;
        }
        signals_shown++;
    }
}

// -----------------------------
// Entry Point
// -----------------------------
int main(int argc, char* argv[]) {
    Config config = parse_args(argc, argv);
    
    std::cout << "=== CUDA Stock Signal Trading System ===\n";
    std::cout << "Configuration:\n";
    std::cout << "  CSV File: " << config.csv_filename << "\n";
    std::cout << "  Window Size: " << config.window_size << "\n";
    std::cout << "  Tolerance: " << config.tolerance << "\n";
    std::cout << "  Max Shares: " << config.max_shares << "\n";
    std::cout << "  Trade Increment: " << config.trade_increment << "\n\n";

    std::vector<float> h_prices_vec;

    if (!read_prices_from_csv(config.csv_filename, h_prices_vec)) {
        return 1;
    }

    int N = static_cast<int>(h_prices_vec.size());
    std::cout << "Loaded " << N << " stock prices from CSV.\n";

    float* h_prices = h_prices_vec.data();
    float* h_moving_avg = new float[N];
    int* h_flags = new int[N];

    // Device memory
    float *d_prices, *d_moving_avg;
    int* d_flags;

    cudaMalloc(&d_prices, N * sizeof(float));
    cudaMalloc(&d_moving_avg, N * sizeof(float));
    cudaMalloc(&d_flags, N * sizeof(int));

    cudaMemcpy(d_prices, h_prices, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Calculate shared memory size needed
    // Each block needs: blockSize + (window_size - 1) elements maximum
    int shared_memory_size = (blockSize + config.window_size - 1) * sizeof(float);

    moving_average_kernel<<<gridSize, blockSize, shared_memory_size>>>(d_prices, d_moving_avg, N, config.window_size);
    detect_signals_combined<<<gridSize, blockSize>>>(d_prices, d_moving_avg, d_flags, N, config.tolerance);

    cudaMemcpy(h_moving_avg, d_moving_avg, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_flags, d_flags, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Run backtesting simulation
    run_backtest(h_prices, h_flags, h_moving_avg, N, config.window_size, config.max_shares, config.trade_increment);

    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_moving_avg);
    cudaFree(d_flags);
    delete[] h_moving_avg;
    delete[] h_flags;

    return 0;
}
