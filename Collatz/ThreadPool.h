#include <array>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <vector>

template <typename Functor>
class ThreadPool {
	std::vector<std::thread> workers;
	std::queue<Functor> workQueue;

	std::mutex mux;
	std::mutex doneMux;
	std::condition_variable cv;
	std::condition_variable doneCv;

	int workLeft = 0;
	bool threadsShouldStop = false;

	void workerLoop() {
		while (true) {
			// Creates unique lock and locks the mutex
			std::unique_lock<std::mutex> lock(mux);

			// Unlocks the mutex, waits until awoken either by notify or spuriously
			// When woken locks the mutex and checks the predicate, unlocks mutex and keeps sleeping if predicate is false
			cv.wait(lock, [&] { return (threadsShouldStop || !workQueue.empty()); });

			if (threadsShouldStop) { return; }// Lock is unlocked on destruction

			// If shouldExit is not true then there has to be work waiting in the queue
			// Take the work from the queue while mutex is held
			Functor work = std::move(workQueue.front());
			workQueue.pop();

			// Unlock mutex and do the work
			lock.unlock();
			work();

			// Decrement workLeft and notify in case main thread is waiting
			{
				std::scoped_lock lock(mux, doneMux);
				workLeft--;
			}
			doneCv.notify_one();
		}
	}

public:
	explicit ThreadPool(int numWorkers) {
		// Create numWorkers threads in-place in the workers vector
		for (int i = 0; i < numWorkers; i++) {
			// Run the member function workerLoop and access member variables through the pointer this in each thread
			workers.emplace_back(std::thread(&ThreadPool::workerLoop, this));
		}

		std::cout << "Created " << numWorkers << " threads" << std::endl;
	}

	~ThreadPool() {
		{
			std::lock_guard<std::mutex> lock(mux);
			threadsShouldStop = true;
		}
		cv.notify_all();

		for (std::thread &worker : workers) {
			worker.join();
		}

		std::cout << "Terminated " << workers.size() << " threads" << std::endl;
	}

	ThreadPool(const ThreadPool &) = delete;
	ThreadPool(ThreadPool &&) = delete;
	ThreadPool &operator=(const ThreadPool &) = delete;
	ThreadPool &operator=(ThreadPool &&) = delete;

	void addWork(Functor f) {
		{
			std::lock_guard<std::mutex> lock(mux);
			workQueue.push(std::move(f));
			workLeft++;
		}
		cv.notify_one();
	}

	void waitForThreads() {
		//std::cout << "Waiting for threads to finish\n";

		std::unique_lock<std::mutex> lock(doneMux);
		doneCv.wait(lock, [&] { 
			/*if (workLeft % 10 == 0) {
				std::cout << workLeft << " blocks left\n";
			}*/
			return workLeft == 0; 
		});
	}
};