#include <iostream>
#include <vector>
#include <locale>
#include <omp.h>

#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;

int64_t overflowCollatz(int64_t number_, int64_t bufferSize, int16_t &extraSteps) {
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
	int64_t sequenceIdx = -1;

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

enum LUT_SIZE: long long { L1 = 16000, L2 = 130000, L3 = 1000000, RAM = 5000000000};

int main() {
	// Adding space as a thousands separator for cout
	struct separate_thousands : std::numpunct<char> {
		char_type do_thousands_sep() const override { return ' '; }  // separate with space
		string_type do_grouping() const override { return "\3"; } // groups of 3 digit
	};
	auto thousands = std::make_unique<separate_thousands>();
	std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

	const int64_t LUTSize = LUT_SIZE::RAM;
	const int64_t bufferSize = 1e6;

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

	#pragma omp parallel num_threads(8) 
	{
		std::vector<std::pair<int64_t, int64_t>> localLargest[8];
		for (int64_t i = LUTSize; i < INT64_MAX; i += bufferSize) {
			#pragma omp for 
			for (int64_t segment = i; segment < (i + bufferSize); segment++) {
				int threadId = omp_get_thread_num();
				int16_t extraSteps = 1;
				int64_t number = segment;

				for (;;) {
					// If odd
					if (number % 2 == 1) {
						// Overflow protection
						if (number > INT64_MAX / 3 - 1) {
							number = overflowCollatz(number, LUTSize, extraSteps);

							int16_t dist = static_cast<int16_t>(LUT[number]) + extraSteps;
							if (dist > largest) {
								localLargest[threadId].push_back(std::pair<int64_t, int16_t>(segment, dist));
							}

							break;
						}
						number = number * 3 + 1;
					}
					// If even
					else {
						number = number / 2;
						if (number < LUTSize) {
							int16_t dist = static_cast<int16_t>(LUT[number]) + extraSteps;
							if (dist > largest) {
								localLargest[threadId].push_back(std::pair<int64_t, int16_t>(segment, dist));
							}
							break;
						}
					}
					extraSteps++;
				}
			}

			#pragma omp single
			{
				for (int i = 0; i < 8; i++) {
					for (std::pair<int64_t, int16_t> dist : localLargest[i]) {
						if (dist.second > largest) {
							largest = dist.second;
							std::cout << "\r                                                                                           ";//"\33[2K";
							std::cout << "\r" << dist.first << " : " << dist.second << "\n";
						}
					}
					localLargest[i].clear();
				}

				std::cout << "\rCurrently on: " << i << " " << std::flush;
			}
		}

	}
}