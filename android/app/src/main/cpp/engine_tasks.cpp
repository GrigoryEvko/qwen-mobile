#include "engine_tasks.h"

#include <sys/stat.h>

#include <exception>
#include <utility>

TaskThread::TaskThread() : thread_([this] { run(); }) {}

TaskThread::~TaskThread() {
    stop();
}

bool TaskThread::post(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return false;
        }
        tasks_.push_back(std::move(task));
    }
    cv_.notify_all();
    return true;
}

void TaskThread::drain() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return tasks_.empty() && !busy_; });
}

void TaskThread::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stopped_) {
            return;
        }
        stopped_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) {
        thread_.join();
    }
}

void TaskThread::run() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        cv_.wait(lock, [this] { return stopped_ || !tasks_.empty(); });
        if (tasks_.empty()) {
            // stop() came and no task is left.
            return;
        }
        std::function<void()> task = std::move(tasks_.front());
        tasks_.pop_front();
        busy_ = true;
        lock.unlock();
        try {
            task();
        } catch (const std::exception &) {
            // A failed task stops only itself. The task writes its own log line.
        } catch (...) {
        }
        lock.lock();
        busy_ = false;
        cv_.notify_all();
    }
}

std::string draft_head_file(const std::string & path) {
    static const std::string kExt    = ".gguf";
    static const std::string kSuffix = "-draft32k";
    if (path.size() <= kExt.size() || path.compare(path.size() - kExt.size(), kExt.size(), kExt) != 0) {
        return {};
    }
    const std::string stem = path.substr(0, path.size() - kExt.size());
    if (stem.size() >= kSuffix.size() && stem.compare(stem.size() - kSuffix.size(), kSuffix.size(), kSuffix) == 0) {
        return {};
    }
    const std::string candidate = stem + kSuffix + kExt;
    struct stat st;
    if (stat(candidate.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
        return {};
    }
    return candidate;
}
