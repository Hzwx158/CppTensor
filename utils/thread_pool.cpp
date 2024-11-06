#include "thread_pool.hpp"

namespace numcpp::thread{
void *ThreadPool::thread_loop(void *args){
    ThreadPool *self = static_cast<ThreadPool*>(args);
    Runable *task;
    while(true){
        {
            // printf("running:%d, empty:%d\n", self->running, self->task_queue.empty());
            std::unique_lock<std::mutex> lock(self->mtx);
            self->task_cv.wait(lock, [self](){
                return (!self->running)||(!self->task_queue.empty());
            });
            if(self->task_queue.empty()&&(!self->running)){
                // printf("exit thread\n");
                //如果不再执行了、也没有任务了，就退了
                return nullptr;
            }
            task = self->task_queue.front();
            self->task_queue.pop();
            self->thread_free --;
        }
        task->run();
        delete task;
        {
            std::lock_guard<std::mutex> lock(self->mtx);
            self->thread_free++;
        }
    }
}
void ThreadPool::initThreads(){
    for(size_t i=0; i<thread_count; ++i){
        if(pthread_create(tids+i, nullptr, &ThreadPool::thread_loop, this)){
            i--;
            continue;
        }
    }
}
ThreadPool::~ThreadPool(){
    // printf("in destroyer\n");
    running = false;
    task_cv.notify_all();
    for(size_t i=0;i<thread_count;++i){
        pthread_join(tids[i], nullptr);
    }
}
}