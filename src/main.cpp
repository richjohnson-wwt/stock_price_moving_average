#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>

int main() {
    // Parameters
    double S0 = 50.0;     // Initial stock price
    double mu = 0.07;      // Expected return
    double sigma = 0.2;    // Volatility
    int N = 2520;                 // Number of days to simulate
    double high = 0.0;
    double low = S0;

    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> norm(0.0, 1.0);

    std::vector<double> prices;
    prices.push_back(S0);

    for (int i = 1; i <= N; ++i) {
        double Z = norm(gen);  // Equivalent to NORMINV(RAND(), 0, 1)
        double dt = 1.0 / 252.0;  // 252 is annual trading days per year
        double St = prices.back() * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
        prices.push_back(St);
        if (St > high) {
            high = St;
        }
        if (St < low) {
            low = St;
        }
    }

    // Output the simulated prices
    for (int i = 0; i <= N; ++i) {
        std::cout << "Day " << i << ": " << prices[i] << std::endl;
    }

    // Export to CSV
    std::ofstream file("stock_prices.csv");
    if (file.is_open()) {
        file << "Day,Stock Price\n";
        for (int i = 0; i <= N; ++i) {
            file << i << "," << prices[i] << "\n";
        }
        file.close();
        std::cout << "Simulation complete. Results saved to 'stock_prices.csv'.\n";
        std::cout << "High: " << high << " Low: " << low << std::endl;
    } else {
        std::cerr << "Error opening file for writing.\n";
    }


    return 0;
}