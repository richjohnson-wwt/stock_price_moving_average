
add_executable(stock_signal stock_signal.cu csv_reader.cpp cli.cpp backtest_simulation.cpp)

# Define the data directory as a compile-time constant
target_compile_definitions(stock_signal PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")