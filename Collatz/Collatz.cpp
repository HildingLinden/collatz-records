#include <iostream>
#include <vector>
#include <locale>
#include <omp.h>

#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;

#include "ThreadPool.h"

// Included from collat_asm.asm (.o/.obj)
int16_t collatz(uint64_t *, int64_t);

int64_t overflowCollatz(uint64_t number_, int64_t bufferSize, int16_t &extraSteps) {
	cpp_int number = number_;

	while (number > bufferSize) {
		if (number % 2 == 0) {
			number = number / 2;
		}
		else {
			number = number * 3 + 1;
		}
		extraSteps++;
	}

	return number.convert_to<int64_t>();
}

void fillLUT(std::vector<int16_t> &LUT, int16_t &largest) {

	std::vector<int64_t> sequence;
	int64_t LUTSum = 0;

	// Fill the lookup table with distances
	for (int64_t i = 3; i < LUT.size(); i++) {

		// If the number has already been added to the look up table
		if (LUT[i] != 0) {
			if (LUT[i] > largest) {
				largest = LUT[i];
				// VT100 escape character for clearing line only works if running through visual studio
				std::cout << "\r                                                                                           ";//"\33[2K";
				std::cout << "\r" << i << " : " << largest << "\n";
				std::cout << "Currently on: " << i << " " << std::flush;
			}
			continue;
		}

		int64_t number = i;

		// Compute sequence until we find a number that has already been computed
		while (true) {
			if (number < LUT.size()) {
				if (LUT[number] != 0) break;
				sequence.push_back(number);
			}
			else {
				// Don't need to save the number if the sequence goes above the lookup table size
				// but 0 is used as a place holder to get the correct amount of steps
				sequence.push_back(0);
			}

			if (number % 2 == 0) {
				number = number / 2;
			}
			else {
				number = 3 * number + 1;
			}
		}

		// Iterate through the sequence and add all steps to the lookup table
		int16_t sequenceSteps = 0;
		int16_t LUTSteps = LUT[number];
		for (auto it = sequence.rbegin(); it != sequence.rend(); it++) {
			sequenceSteps++;

			if (*it > 0) {
				LUT[*it] = LUTSteps + sequenceSteps;
				LUTSum += static_cast<int64_t>(LUTSteps) + sequenceSteps;
			}
		}

		sequence.clear();

		if (LUT[i] > largest) {
			largest = LUT[i];
			// VT100 escape character for clearing line only works if running through visual studio
			std::cout << "\r                                                                                           ";//"\33[2K";
			std::cout << "\r" << i << " : " << largest << "\n";
			std::cout << "Filling lookup table. Currently on: " << i << " of " << LUT.size() << std::flush;
		}
		else if (i % 1000000 == 0) {
			std::cout << "\rFilling lookup table. Currently on: " << i << " of " << LUT.size() << std::flush;
		}
	}

	std::cout << "\33[2K";
	std::cout << "\r" << "Lookup table filled. Avg steps: " << LUTSum / LUT.size() << "\n";
}

class multiThreadCollatz {
	std::vector<int16_t> &LUT, &buffer;
	int64_t start, end, offset;
public:
	multiThreadCollatz(std::vector<int16_t> &LUT_, std::vector<int16_t> &buffer_, int64_t start_, int64_t end_, int64_t offset_) : LUT(LUT_), buffer(buffer_), start(start_), end(end_), offset(offset_) {}
	~multiThreadCollatz() = default;

	multiThreadCollatz(const multiThreadCollatz &) = delete;
	multiThreadCollatz &operator=(const multiThreadCollatz &) = delete;

	multiThreadCollatz(multiThreadCollatz &&) = default;
	multiThreadCollatz &operator=(multiThreadCollatz &&) = default;

	void operator()() {
		for (int64_t j = start; j < end; j++) {
			uint64_t number = j;

			int16_t steps = collatz(&number, LUT.size());
			if (number > LUT.size()) {
				number = overflowCollatz(number, LUT.size(), steps);
			}

			buffer[j - offset] = LUT[number] + steps;
		}
	}
};

enum LUT_SIZE : long long { L1 = 16000, L2 = 130000, L3 = 1000000, RAM = 5000000000 };

int main() {
	// Adding space as a thousands separator for cout
	struct separate_thousands : std::numpunct<char> {
		char_type do_thousands_sep() const override { return ' '; }  // separate with space
		string_type do_grouping() const override { return "\3"; } // groups of 3 digit
	};
	auto thousands = std::make_unique<separate_thousands>();
	std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

	const int64_t LUTSize = LUT_SIZE::RAM;
	const int64_t bufferSize = 1e7*5;
	const int64_t iterationsPerBlock = 1e5;

	// Allocate memory for the lookup table
	std::cout << "Allocting " << LUTSize * sizeof(int16_t) << " Bytes of memory" << std::endl;
	std::vector<int16_t> LUT(LUTSize);

	// Setting up the beginning of the lookup table
	LUT[2] = 1;
	int16_t largest = LUT[2];
	std::cout << "1 : " << LUT[1] << std::endl;
	std::cout << "2 : " << LUT[2] << std::endl;

	// Fill the lookup table and update the largest distance
	fillLUT(LUT, largest);

	ThreadPool<multiThreadCollatz> pool(10);
	std::vector<int16_t> buffer(bufferSize);
	std::vector<int16_t> buffer2(bufferSize);

	for (int64_t i = LUTSize; i < INT64_MAX; i += bufferSize * 2) {
		std::cout << "\rCurrently on " << i << std::flush;

		// Start filling buffer 1
		for (int64_t j = i; j < (i + bufferSize); j += iterationsPerBlock) {
			pool.addWork(multiThreadCollatz(LUT, buffer, j, j + iterationsPerBlock, i));
		}		
		
		// Check buffer 2 
		for (int64_t j = 0; j < bufferSize; j++) {
			if (buffer2[j] > largest) {
				largest = buffer2[j];
				std::cout << "\r                                                                                           ";//"\33[2K";
				std::cout << "\r" << i + j << " : " << buffer2[j] << std::endl;
				std::cout << "\rCurrently on " << i << std::flush;
			}
		}

		// Wait for buffer 1 to finish
		pool.waitForThreads();

		std::cout << "\rCurrently on " << i + bufferSize << std::flush;

		// Start filling buffer 2
		for (int64_t j = (i + bufferSize); j < (i + bufferSize * 2); j += iterationsPerBlock) {
			pool.addWork(multiThreadCollatz(LUT, buffer2, j, j + iterationsPerBlock, i+bufferSize));
		}

		// Check buffer 1 
		for (int64_t j = 0; j < bufferSize; j++) {
			if (buffer[j] > largest) {
				largest = buffer[j];
				std::cout << "\r                                                                                           ";//"\33[2K";
				std::cout << "\r" << i + j << " : " << buffer[j] << std::endl;
				std::cout << "\rCurrently on " << i + bufferSize << std::flush;
			}
		}

		// Wait for buffer 2 to finish
		pool.waitForThreads();
		
	}
}