
find_package(Catch2 REQUIRED)

add_executable(csv_reader_test csv_reader_test.cpp ../src/csv_reader.cpp)
target_link_libraries(csv_reader_test PRIVATE Catch2::Catch2WithMain)
target_include_directories(csv_reader_test PRIVATE ../src)

# Define the test data directory as a compile-time constant
target_compile_definitions(csv_reader_test PRIVATE 
    DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data")

include(CTest)
include(Catch)
catch_discover_tests(csv_reader_test)

