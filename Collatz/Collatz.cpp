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

int main() {
	const int64_t bufferSize = 1e9*5;
	const int64_t secondaryBufferSize = 1e6;

	int16_t *distance = static_cast<int16_t *>(malloc(bufferSize * sizeof(int16_t)));
	if (!distance) exit(1);

	memset(distance, -1, bufferSize * sizeof(int16_t));
	distance[0] = 0;
	distance[1] = 0;

	int16_t largest = distance[0];
	std::cout << "1 : " << largest << "\n";

	struct separate_thousands : std::numpunct<char> {
		char_type do_thousands_sep() const override { return ' '; }  // separate with space
		string_type do_grouping() const override { return "\3"; } // groups of 3 digit
	};

	auto thousands = std::make_unique<separate_thousands>();
	std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

	int64_t *sequence = static_cast<int64_t *>(malloc(secondaryBufferSize * sizeof(int64_t)));
	if (!sequence) exit(1);

	int64_t bufferSum = 0;
	int64_t sequenceIdx = -1;
	// Fill the buffer with all distances
	for (int64_t i = 2; i < bufferSize; i++) {
		int64_t number = i;
		bool overflow = false;

		for (;;) {
			if (number < bufferSize) {
				if (distance[number] != -1) {
					break;
				}
				else {
					sequence[++sequenceIdx] = number;
				}
			}
			else {
				sequence[++sequenceIdx] = -1;
			}

			// If odd
			if (number % 2 == 1) {
				number = number * 3 + 1;
			}
			// If even
			else {
				number = number / 2;
			}
		}

		int64_t seqDist = 1;
		for (; sequenceIdx >= 0; sequenceIdx--) {
			// *it is -1 if larger than buffersize
			int64_t integer = sequence[sequenceIdx];
			if (integer > 0) {
				distance[integer] = distance[number] + seqDist;
				bufferSum += distance[number] + seqDist;
			}
			seqDist++;
		}

		if (distance[i] > largest) {
			largest = distance[i];
			// VT100 escape character for clearing line only works if running through visual studio
			std::cout << "\r                                                                                           ";//"\33[2K";
			std::cout << "\r" << i << " : " << largest << "\n";
			std::cout << "Currently on: " << i << " " << std::flush;
		}
		else if (i % 1000000 == 0) {
			std::cout << "\rCurrently on: " << i << " " << std::flush;
		}
	}

	std::cout << "\33[2K";
	std::cout << "\r" << "Buffersize of " << bufferSize << " filled. Avg size of buffer elements: " << bufferSum / bufferSize << "\n";
	std::cout << "Continuing without saving to buffer.\n";

	#pragma omp parallel num_threads(8) 
	{
		std::vector<std::pair<int64_t, int64_t>> localLargest[8];
		for (int64_t i = bufferSize; i < INT64_MAX; i += secondaryBufferSize) {
			#pragma omp for 
			for (int64_t segment = i; segment < (i + secondaryBufferSize); segment++) {
				int threadId = omp_get_thread_num();
				int16_t extraSteps = 1;
				int64_t number = segment;

				for (;;) {
					// If odd
					if (number % 2 == 1) {
						// Overflow protection
						if (number > INT64_MAX / 3 - 1) {
							number = overflowCollatz(number, bufferSize, extraSteps);

							int16_t dist = static_cast<int16_t>(distance[number]) + extraSteps;
							if (dist > largest) {
								localLargest[threadId].push_back(std::pair<int64_t, int16_t>(segment, dist));
							}

							std::cout << "\nOverflow when starting with: " << segment << ", with " << dist << " steps" << std::endl;

							break;
						}
						number = number * 3 + 1;
					}
					// If even
					else {
						number = number / 2;
						if (number < bufferSize) {
							int16_t dist = static_cast<int16_t>(distance[number]) + extraSteps;
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