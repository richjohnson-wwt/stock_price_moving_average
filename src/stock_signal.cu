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

#include "stock_signal.cuh"

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
    run_backtest(h_prices, h_flags, h_moving_avg, N, config.window_size, config.max_shares, config.trade_increment, config.verbose);

    // Cleanup
    cudaFree(d_prices);
    cudaFree(d_moving_avg);
    cudaFree(d_flags);
    delete[] h_moving_avg;
    delete[] h_flags;

    return 0;
}
