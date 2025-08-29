#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <string>

#include "../src/stock_signal.cuh"

TEST_CASE("Read prices from CSV") {
    std::vector<float> prices;
    std::string data_file = std::string(DATA_DIR) + "/stock_prices.csv";
    bool success = read_prices_from_csv(data_file, prices);
    REQUIRE(success);
    REQUIRE(prices.size() == 2521);
}
