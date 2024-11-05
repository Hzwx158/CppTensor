#ifndef NUMCPP_UTILS_THREADS_POOL_HPP
#define NUMCPP_UTILS_THREADS_POOL_HPP
#include <mutex>
#include <queue>
#include <thread>
#include <tuple>
#include <utility>
namespace numcpp::thread{

struct Runable{virtual void run() const=0;};

template<class Functor, class ...Args>
class Task:public Runable{
private:
    Functor &&f;
    std::tuple<Args...> args;
public:
    Task(Functor &&f_, Args&&... args_)
        :f((Functor&&)f_)
        ,args{std::forward<Args>(args_)...}{}
    ~Task(){}
    void run() const override final{std::apply(f, args);}
};

template<size_t N>
class ThreadPool{
private:
    using Thread=std::thread;
    using TaskQueue = std::queue<Runable*>;
    TaskQueue task_queue;
    std::condition_variable task_cv;
    size_t thread_free;
    std::mutex mtx;
    Thread thread_array[N];
    bool running;
    void initThreads(){
        for(size_t i=0; i<N; ++i){
            thread_array[i] = Thread([this](){
            while(true){ //每个线程循环执行：
                // printf("i=%d, empty=%d, free=%d\n",i, this->task_queue.empty(), this->thread_free);
                Runable *task;
                {
                    std::unique_lock<std::mutex> lock(this->mtx);
                    task_cv.wait(lock, [this](){
                        return !this->running||!this->task_queue.empty();
                    }); //wait直到队列非空，或者不再执行
                    if(this->task_queue.empty()&&(!this->running))
                        return; //如果不再执行了、也没有任务了，就退了
                    task = this->task_queue.front();
                    this->task_queue.pop();
                    this->thread_free --;
                }
                task->run();
                {
                    std::lock_guard<std::mutex> lock(this->mtx);
                    this->thread_free++;
                }
            }});
        }
    }
public:
    ThreadPool():task_queue(), task_cv(), thread_free{N},running{true}{
        initThreads();
    }
    ~ThreadPool(){
        // printf("in destroyer\n");
        running = false;
        task_cv.notify_all();
        for(Thread &t:thread_array){
            if(t.joinable())
                t.join();
        }
    }
    template<class Functor, class ...Args>
    void post(Functor &&f, Args &&... args){
        // printf("post\n");
        Runable *task = new Task<Functor, Args...>(std::forward<Functor>(f), std::forward<Args>(args)...);
        {
            std::lock_guard lg(mtx);
            task_queue.push(task);
        }
        task_cv.notify_one();//唤起一个线程来执行
    }
    
};




}

#endif