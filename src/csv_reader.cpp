#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>

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