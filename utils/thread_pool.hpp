#ifndef NUMCPP_UTILS_THREADS_POOL_HPP
#define NUMCPP_UTILS_THREADS_POOL_HPP
#include <mutex>
#include <queue>
#include <tuple>
#include <pthread.h>
#include <sstream>
namespace numcpp::thread{
/**
 * @brief A base class of runable task templates
 */
struct Runable{
    virtual void run() const=0;
    virtual ~Runable(){}
};

/**
 * @brief Task class
 * @tparam Functor type of the function invoked by task
 * @tparam Args type of arguments for the function
 */
template<class Functor, class ...Args>
class Task:public Runable{
private:
    Functor &&f;
    std::tuple<Args...> args;
public:
    /**
     * @brief constructor
     * @param f_ function to be invoked
     * @param args_ arguments for function
     */
    Task(Functor &&f_, Args&&... args_)
        :f((Functor&&)f_)
        ,args{std::forward<Args>(args_)...}{}
    ~Task(){}
    /**
     * @brief run the function
     */
    void run() const override {std::apply(f, args);}
};
/**
 * @brief a simple thread pool with mutex
 */
class ThreadPool{
private:
    std::queue<Runable*> task_queue; // queue of task
    std::mutex mtx; // mutex for task queue/condition_variable
    std::condition_variable task_cv; // condition_variable for task
    size_t thread_count; // total number of threads
    size_t thread_free; // number of free threads
    pthread_t *tids; // set of threads' id
    bool running; //whether the thread pool is still running
    /**
     * @brief function running in thread
     * @param args take `this` as argument
     * @return `nullptr`
     */
    static void *thread_loop(void *args);
    /**
     * @brief initialize each thread, called in constructor
     */
    void initThreads();
public:
    /**
     * @brief constructor
     * @param thread_count_ total count of threads
     */
    ThreadPool(size_t thread_count_)
        :task_queue{},mtx(),task_cv()
        ,thread_count{thread_count_},thread_free{thread_count_}
        ,tids{new pthread_t[thread_count_]}
        ,running{true}
    {
        initThreads();
    }
    ~ThreadPool();
    /**
     * @brief post a task into pool
     * @param f task function to be invoked
     * @param args arguments of `f`
     */
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
    /**
     * @brief get total number of free threads
     * @return an unsigned long long
     */
    size_t countThreadFree() const {return thread_free;}
    
};
/**
 * @brief A simple class for async output
 */
class OStream{
private:
    // cache
    std::ostringstream oss;
    // the std::ostream binded
    std::ostream &osm_ref;
    // mutex for output
    std::mutex mtx;
public:
    /**
     * @brief constructor, bind this with a std::ostream
     * @param osm the ostream to be bind
     */
    explicit OStream(std::ostream &osm):oss(), osm_ref(osm), mtx(){}
    OStream(const OStream &) = delete;
    OStream(OStream &&)=delete;
    ~OStream(){
        if(oss.str().length())
            osm_ref << oss.str();
    }
    OStream &operator<<(std::ostream &(*f)(std::ostream &)){
        {
            std::lock_guard lg(mtx);
            osm_ref << oss.str() << f;
            oss.str("");
        }
        return *this;
    }
    OStream &operator<<(OStream &(*f)(OStream &)){
        return f(*this);;
    }
    template<class T>
    OStream &operator<<(T &&obj){
        {    
            std::lock_guard lg(mtx);
            oss << obj;
            std::string const &s = oss.str();
            if(s.length()>10){
                osm_ref << s;
                oss.str("");
            }
        }
        return *this;
    }
    // friend OStream &endl(OStream &async_osm);
};
// OStream &endl(OStream &async_osm);

}

#endif