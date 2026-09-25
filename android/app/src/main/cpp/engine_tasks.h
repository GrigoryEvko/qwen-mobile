/**
 * The background work of an engine: a thread that runs tasks in order. Also
 * the file with the reduced draft head of a model.
 *
 * No llama.cpp and no Android dependency.
 */
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>

/**
 * A thread that runs tasks one at a time, in the order of the requests. A
 * task that throws an exception stops only itself.
 *
 * The owner calls stop() before it destroys what the tasks use: stop() runs
 * the queued tasks and joins the thread. The destructor calls stop().
 */
class TaskThread {
public:
    TaskThread();
    ~TaskThread();

    TaskThread(const TaskThread &)             = delete;
    TaskThread & operator=(const TaskThread &) = delete;

    /** Queue a task. After stop() the call does nothing and returns false. */
    bool post(std::function<void()> task);

    /** Wait until the queue is empty and no task runs. */
    void drain();

    /** Run the queued tasks, then join the thread. A second call does nothing. */
    void stop();

private:
    void run();

    // The thread is the last member: a member starts before the members that
    // follow it, thus a thread declared first runs while the mutex, the queue
    // and the flags of this object are still raw memory.
    std::mutex                        mutex_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> tasks_;
    bool                              busy_    = false;
    bool                              stopped_ = false;
    std::thread                       thread_;
};

/**
 * The file of the same model with the reduced MTP draft head, when it is next
 * to the file: "<stem>-draft32k.gguf" for "<stem>.gguf". Returns the path of
 * the file when it is a regular file, else an empty string. A path that is
 * already such a file gives an empty string.
 */
std::string draft_head_file(const std::string & path);
