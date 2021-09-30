#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <conio.h>
#include <iostream>
#include <bitset>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <string>
#include <chrono>

#include <boost/multiprecision/cpp_int.hpp>
using namespace boost::multiprecision;

#include "ThreadPool.h"

// Included from collatz_asm.asm (.o/.obj)
int16_t collatz(uint64_t *, int64_t);

struct record_t {
	int16_t steps;
	int64_t number;
};

constexpr int64_t iterationsPerThread = 1000;

__global__ void fillLUTGPU(int16_t *LUT, int64_t offset) {
	int64_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
	idx *= 50000;

	for (int64_t i = 0; i < 50000; i++) {
		int64_t number = idx + i + offset;
		int16_t steps = 0;

		while (number > 1) {
			if (number % 2 == 0) {
				number = number / 2;
			}
			else {
				number = number * 3 + 1;
			}
			steps++;
		}

		LUT[idx + i] = steps;
	}
}

void fillLUT(std::vector<int16_t> &LUT) {
	int16_t *devLUT;
	cudaMalloc(&devLUT, (LUT.size() / 10) * sizeof(int16_t));

	std::cout << "Filling lookup table" << std::endl;

	for (int64_t i = 0; i < 10; i++) {
		int64_t offset = i * 500000000;

		std::cout << "\rCurrently on " << offset << " of " << LUT.size();

		fillLUTGPU << < 100, 100 >> > (devLUT, offset);

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "fillLUTGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error %s, after fillLUTGPU\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		cudaStatus = cudaMemcpy(LUT.data() + offset, devLUT, (LUT.size() / 10) * sizeof(int16_t), cudaMemcpyDefault);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy returned error %s, after fillLUTGPU\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}
	}

	cudaError_t cudaStatus = cudaFree(devLUT);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy returned error %s, after fillLUTGPU\n", cudaGetErrorString(cudaStatus));
		exit(1);
	}

	std::cout << "\rFilled the lookup table with " << LUT.size() << " entries" << std::endl;
}

__global__ void collatzGPU(int16_t *recordNums, record_t *records, int64_t offset, int16_t largest) {
	int64_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

	recordNums[idx] = 0;
	for (int64_t i = 0; i < iterationsPerThread; i++) {
		int64_t number = idx * iterationsPerThread + i + offset;
		int16_t steps = 0;

		while (number > 1) {
			if (number % 2 == 0) {
				number = number / 2;
			}
			else {
				if (number > INT64_MAX / 3 + 1) {
					records[idx * 100 + recordNums[idx]].number = idx * iterationsPerThread + i + offset;
					records[idx * 100 + recordNums[idx]].steps = -1;
					recordNums[idx]++;
					steps = 0;
					if (recordNums[idx] == 100) {
						recordNums[idx] = -1;
						return;
					}
					break;
				}
				number = number * 3 + 1;
			}
			steps++;
		}

		if (steps > largest) {
			largest = steps;
			records[idx * 100 + recordNums[idx]].number = idx * iterationsPerThread + i + offset;
			records[idx * 100 + recordNums[idx]].steps = steps;
			recordNums[idx]++;
			if (recordNums[idx] == 100) {
				recordNums[idx] = -1;
				return;
			}
		}
	}
}

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

std::ofstream checkProgressFile(int16_t &largest, int64_t &startingNumber, std::chrono::duration<long long> &totalTime, bool &resumeComputation) {
	// Check if result files exists and save everything else but the last row
	if (std::filesystem::exists("collatz_results.txt")) {

		std::ifstream inputFile;
		inputFile.open("collatz_results.txt");
		if (!inputFile.is_open()) {
			std::cout << "Could not open input file" << std::endl;
			exit(EXIT_FAILURE);
		}

		std::ofstream tmpFile;
		tmpFile.open("tmpFile");
		if (!tmpFile.is_open()) {
			std::cout << "Could not open temporary file" << std::endl;
			exit(EXIT_FAILURE);
		}

		std::string str;
		int64_t recordNumber;
		while (getline(inputFile, str)) {
			if (str.find("end") == std::string::npos) {
				if (!str.empty()) {
					largest = std::stoi(str.substr(str.find(" ")));
					recordNumber = std::stoll(str.substr(0, str.find(" ")));
					tmpFile << str << "\n";
				}
			}
			else {
				startingNumber = std::stoll(str.substr(0, str.find(" ")));
				totalTime = std::chrono::seconds(std::stoll(str.substr(str.find("-") + 1)));
				resumeComputation = true;

				std::time_t t = std::time(nullptr);
				std::tm tm = *std::localtime(&t);

				std::cout << "Previous total execution time: " << totalTime.count() << " seconds" << std::endl;
				std::cout << "Largest previous record: " << recordNumber << " : " << largest << std::endl;
				std::cout << std::put_time(&tm, "\n[%T] Resuming from: ") << startingNumber << std::endl;
			}
		}

		inputFile.close();
		tmpFile.close();

		if (remove("collatz_results.txt") != 0) {
			std::cout << "Could not remove original input file" << std::endl;
			exit(EXIT_FAILURE);
		}
		if (rename("tmpFile", "collatz_results.txt") != 0) {
			std::cout << "Could not rename temporary file" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	std::ofstream resultFile;
	std::ios_base::openmode flags = resumeComputation ? std::ios::app : 0;
	resultFile.open("collatz_results.txt", flags);
	if (!resultFile.is_open()) {
		std::cout << "Could not open result file" << std::endl;
		exit(EXIT_FAILURE);
	}

	return resultFile;
}

int main(int argc, char *argv[]) {
	std::chrono::duration<long long> totalTime = std::chrono::seconds(0);
	auto startTime = std::chrono::steady_clock::now();

	struct separate_thousands : std::numpunct<char> {
		char_type do_thousands_sep() const override { return ' '; }  // separate with space
		string_type do_grouping() const override { return "\3"; } // groups of 3 digit
	};

	auto thousands = std::make_unique<separate_thousands>();
	std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

	const int64_t size = 5000000000;
	std::vector<int16_t> LUT(size);

	int16_t largest = 0;
	int64_t startingNumber = LUT.size();
	bool resumeComputation = false;

	std::ofstream resultFile = checkProgressFile(largest, startingNumber, totalTime, resumeComputation);

	fillLUT(LUT);

	int64_t lutSum = 0;
	for (int64_t i = 0; i < LUT.size(); i++) {
		lutSum += LUT[i];

		// Find records in LUT if starting from scratch
		if (!resumeComputation) {
			if (LUT[i] > largest) {
				largest = LUT[i];
				std::cout << i << " : " << largest << std::endl;
				resultFile << i << " " << largest << std::endl;
			}
		}
	}

	const int64_t blocks = 1000;
	const int64_t threads = 100;
	const int64_t bufferSize = blocks * threads;

	int16_t *recordNums;
	record_t *records;

	cudaError_t cudaStatus = cudaMallocManaged(&recordNums, bufferSize * sizeof(int16_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

	cudaStatus = cudaMallocManaged(&records, 100 * bufferSize * sizeof(record_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

	int64_t overflowCount = 0;
	for (int64_t i = startingNumber; i < INT64_MAX; i += bufferSize * iterationsPerThread) {
		if (_kbhit()) {
			std::chrono::time_point endTime = std::chrono::steady_clock::now();
			std::chrono::duration<long long> time = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
			totalTime += time;

			std::cout << "\n\nSaving on: " << i << std::endl;
			resultFile << i << " end-" << totalTime.count() << std::endl;


			resultFile.close();
			
			std::cout << "Session execution time: " << time.count() << " seconds" << std::endl;
			std::cout << "Total execution time: " << totalTime.count() << " seconds" << std::endl;
			exit(0);
		}

		std::cout << "\rCurrently on " << i << ", overflows: " << overflowCount;

		
		collatzGPU <<<blocks, threads>>> (recordNums, records, i, largest);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "collatzGPU launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error %s, after launching collatzGPU!\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		for (int64_t i = 0; i < bufferSize; i++) {
			int16_t recordNum = recordNums[i];
			if (recordNum < 0) {
				std::cout << "Negative" << std::endl;
			}
			else if (recordNum > 0) {
				for (int16_t record = 0; record < recordNum; record++) {
					if (records[i * 100 + record].steps < 0) {
						overflowCount++;
						int16_t steps = 0;
						int64_t number = overflowCollatz(records[i * 100 + record].number, 1, steps);
						if (steps > largest) {
							largest = steps;
							std::cout << "\r                                                                                ";
							std::cout << "\r" << records[i * 100 + record].number << " : " << steps << std::endl;
							resultFile << records[i * 100 + record].number << " " << steps << std::endl;
						}
					}
					else if (records[i * 100 + record].steps > largest) {
						largest = records[i * 100 + record].steps;
						std::cout << "\r                                                                                ";
						std::cout << "\r" << records[i * 100 + record].number << " : " << records[i * 100 + record].steps << std::endl;
						resultFile << records[i * 100 + record].number << " " << records[i * 100 + record].steps << std::endl;
					}
				}
			}
		}
	}
}
