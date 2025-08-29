#include "stock_signal.cuh"
#include <iostream>
#include <ostream>
#include <cmath>

// -----------------------------
// Pure Portfolio Calculation Logic (Testable)
// -----------------------------
BacktestResult calculate_backtest(const float* prices, const int* signals, 
                                 int N, int window_size, int max_shares, int trade_increment) {
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
            }
        }
        else if (signals[i] == -1 && shares_owned > 0) {  // SELL signal
            int shares_to_sell = std::min(trade_increment, shares_owned);
            float proceeds = shares_to_sell * current_price;
            
            shares_owned -= shares_to_sell;
            cash += proceeds;
            total_trades++;
            sell_trades++;
        }
        // HOLD: do nothing, just track portfolio value
    }
    
    // Calculate final performance metrics
    float final_price = prices[N - 1];
    float final_portfolio_value = cash + (shares_owned * final_price);
    float total_return = ((final_portfolio_value - starting_cash) / starting_cash) * 100.0f;
    float max_drawdown = ((max_portfolio_value - min_portfolio_value) / max_portfolio_value) * 100.0f;
    
    // Return results
    BacktestResult result;
    result.final_portfolio_value = final_portfolio_value;
    result.total_return = total_return;
    result.max_drawdown = max_drawdown;
    result.total_trades = total_trades;
    result.buy_trades = buy_trades;
    result.sell_trades = sell_trades;
    result.final_shares_owned = shares_owned;
    result.final_cash = cash;
    
    return result;
}

// -----------------------------
// Display Wrapper Function
// -----------------------------
void run_backtest(const float* prices, const int* signals, const float* moving_avg, 
                  int N, int window_size, int max_shares, int trade_increment, bool verbose) {
    
    // Use the pure calculation function
    BacktestResult result = calculate_backtest(prices, signals, N, window_size, max_shares, trade_increment);
    
    // Display initial information
    float initial_price = prices[window_size - 1];
    float starting_cash = max_shares * initial_price;
    
    std::cout << "\n--- Portfolio Backtesting ---\n";
    std::cout << "Starting with: $" << starting_cash << " cash (equivalent to " << max_shares << " shares @ $" << initial_price << ")\n";
    std::cout << "Trade size: " << trade_increment << " shares\n\n";
    
    // Show detailed trading activity if verbose
    if (verbose) {
        // Re-run the trading simulation for verbose output
        int shares_owned = 0;
        float cash = starting_cash;
        
        for (int i = window_size - 1; i < N; i++) {
            float current_price = prices[i];
            float current_portfolio_value = cash + (shares_owned * current_price);
            
            // Execute trades with verbose logging
            if (signals[i] == 1 && shares_owned < max_shares) {  // BUY signal
                int shares_to_buy = std::min(trade_increment, max_shares - shares_owned);
                float cost = shares_to_buy * current_price;
                
                if (cash >= cost) {
                    shares_owned += shares_to_buy;
                    cash -= cost;
                    
                    std::cout << "Tick " << i << ": BUY  " << shares_to_buy 
                              << " shares @ $" << current_price 
                              << " | Portfolio: $" << current_portfolio_value 
                              << " (" << shares_owned << " shares, $" << cash << " cash)" << std::endl;
                }
            }
            else if (signals[i] == -1 && shares_owned > 0) {  // SELL signal
                int shares_to_sell = std::min(trade_increment, shares_owned);
                float proceeds = shares_to_sell * current_price;
                
                shares_owned -= shares_to_sell;
                cash += proceeds;
                
                std::cout << "Tick " << i << ": SELL " << shares_to_sell 
                          << " shares @ $" << current_price 
                          << " | Portfolio: $" << current_portfolio_value 
                          << " (" << shares_owned << " shares, $" << cash << " cash)" << std::endl;
            }
        }
    }
    
    // Display final results
    std::cout << "\n--- Final Portfolio Results ---\n";
    std::cout << "Final Portfolio Value: $" << result.final_portfolio_value << "\n";
    std::cout << "Starting Value: $" << starting_cash << "\n";
    std::cout << "Total Return: " << result.total_return << "%\n";
    std::cout << "Maximum Drawdown: " << result.max_drawdown << "%\n";
    std::cout << "Total Trades: " << result.total_trades << " (" << result.buy_trades << " buys, " << result.sell_trades << " sells)\n";
    std::cout << "Final Position: " << result.final_shares_owned << " shares, $" << result.final_cash << " cash\n\n";

    // Output first 20 trading signals for verification (only in verbose mode)
    if (verbose) {
        std::cout << "--- First 20 Trading Signals (for verification) ---\n";
        int signals_shown = 0;
        for (int i = window_size - 1; i < N && signals_shown < 20; i++) {
            if (signals[i] == 1) {
                std::cout << "Tick " << i << ": BUY  @ Price = " << prices[i]
                        << ", MA = " << moving_avg[i] << std::endl;
            } else if (signals[i] == -1) {
                std::cout << "Tick " << i << ": SELL @ Price = " << prices[i]
                        << ", MA = " << moving_avg[i] << std::endl;
            } else {
                std::cout << "Tick " << i << ": HOLD @ Price = " << prices[i]
                        << ", MA = " << moving_avg[i] << std::endl;
            }
            signals_shown++;
        }
    }
}