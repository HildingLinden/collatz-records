
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

__global__ void collatzBitSet(unsigned long long int *result, const int largest, const unsigned long long int offset, const int iterationsPerThread) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	int64_t resultBitset = 0;

	for (int i = 0; i < iterationsPerThread; i++) {
		long long int number = offset + idx * iterationsPerThread + i;
		int steps = -1;

		while (true) {
			steps++;

			// If odd
			if (number % 2 == 1) {
				number = number * 3 + 1;
			}
			// If even
			else {
				number = number / 2;
				if (number < 2) {
					if (steps > largest) {
						//printf("\n%llu\n", offset + idx * 64 + i);
						//atomicOr(&result[idx], 1ULL << i);
						resultBitset |= 1ULL << i;
					}
					break;
				}
			}
		}
	}

	result[idx] = resultBitset;
}

int collatzDistance(int64_t number) {
	int steps = 0;
	while (true) {
		if (number < 2) {
			return steps;
		}

		if (number % 2 == 0) {
			number = number / 2;
		}
		else {
			number = number * 3 + 1;
		}
		steps++;
	}	
}


int main(int argc, char *argv[])
{
	struct separate_thousands : std::numpunct<char> {
		char_type do_thousands_sep() const override { return ' '; }  // separate with space
		string_type do_grouping() const override { return "\3"; } // groups of 3 digit
	};

	auto thousands = std::make_unique<separate_thousands>();
	std::cout.imbue(std::locale(std::cout.getloc(), thousands.release()));

	std::chrono::duration<long long> totalTime = std::chrono::seconds(0);
	auto startTime = std::chrono::steady_clock::now();

	uint64_t *steps;
	const int blocks = 1024;
	const int threads = 512;
	const int segmentSize = blocks * threads;
	const int iterationsPerThread = 64;
	const int64_t maxNumber = 10e17;
	int64_t startingNumber = 0;
	int largest = 0;
	int64_t largestRecord = 0;
	bool resumeComputation = false;

	cudaError_t cudaStatus = cudaMallocManaged(&steps, segmentSize * sizeof(uint64_t));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		exit(1);
	}

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
		while (getline(inputFile, str)) {
			if (str.find("end") == std::string::npos) {
				if (!str.empty()) {
					largest = std::stol(str.substr(str.find(" ")));
					largestRecord = std::stoll(str.substr(0, str.find(" ")));
					tmpFile << str << "\n";
				}
			}
			else {
				startingNumber = std::stoll(str.substr(0, str.find(" ")));
				totalTime = std::chrono::seconds(std::stoll(str.substr(str.find("-")+1)));
				resumeComputation = true; 
				
				std::time_t t = std::time(nullptr);
				std::tm tm = *std::localtime(&t);

				std::cout << "Previous total execution time: " << totalTime.count() << " seconds" << std::endl;
				std::cout << "Largest previous record: " << largestRecord << " : " << largest << std::endl;
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


	if (!resumeComputation) {
		// Start by computing some distance on CPU so that the kernels have a decently high cutoff
		for (int i = 0; i < 10000000; i++) {
			int dist = collatzDistance(i);
			if (dist > largest) {
				largest = dist;

				std::time_t t = std::time(nullptr);
				std::tm tm = *std::localtime(&t);

				std::cout << std::put_time(&tm, "[%T] ") << i << " : " << largest << std::endl;
				resultFile << i << " " << largest << "\n";
			}
		}
	}

	int segment = 0;
	while (true) {
		uint64_t offset = (uint64_t)segment * (uint64_t)segmentSize * iterationsPerThread + startingNumber;

		if (_kbhit()) {
			std::chrono::time_point endTime = std::chrono::steady_clock::now();
			std::chrono::duration<long long> time = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
			totalTime += time;

			std::cout << "\n\nSaving on: " << offset << std::endl;
			resultFile << offset << " end-" << totalTime.count() << std::endl;


			resultFile.close();
			
			std::cout << "Session execution time: " << time.count() << " seconds" << std::endl;
			std::cout << "Total execution time: " << totalTime.count() << " seconds" << std::endl;
			exit(0);
		}

		collatzBitSet <<< blocks, threads >>> (steps, largest, offset, iterationsPerThread);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			exit(1);
		}

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			exit(1);
		}

		for (int i = 0; i < segmentSize; i++) {
			uint64_t number = steps[i];
			if (number) {
				std::bitset<64> bits(number);
				for (int j = 0; j < iterationsPerThread; j++) {
					if (bits[j]) {
						int distance = collatzDistance(offset + i * iterationsPerThread + j);
						if (distance > largest) {
							largest = distance;

							std::time_t t = std::time(nullptr);
							std::tm tm = *std::localtime(&t);

							std::cout << "\r                                                                     ";
							std::cout << "\r" << std::put_time(&tm, "[%T] ") << offset + i * iterationsPerThread + j << " : " << largest << std::endl;
							std::cout << "\rCurrently on : " << offset;
							resultFile << offset + i * iterationsPerThread + j << " " << largest << "\n";
						}
					}
				}
			}
		}

		std::cout << "\rCurrently on : " << offset;

		segment++;
	}
}
