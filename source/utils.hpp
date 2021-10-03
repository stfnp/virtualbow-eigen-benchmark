#pragma once
#include <chrono>

// Executes a function multiple times and returns the average
// execution time (wall clock time) in milliseconds
template<typename F>
double execute_and_measure(F function, int samples) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    auto t0 = high_resolution_clock::now();

    for(int i = 0; i < samples; ++i) {
        function();
    }

    auto t1 = high_resolution_clock::now();

    duration<double, std::milli> time_ms = t1 - t0;
    return time_ms.count()/samples;
}
