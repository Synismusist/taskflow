#include <taskflow/taskflow.hpp>

// require c++20 compiler
#include <taskflow/coroutine/coroutine.hpp>

int main() {

  tf::Taskflow taskflow;
  tf::Executor executor;
  
  // create a coroutine static task
  tf::Task task = taskflow.emplace([]() -> tf::StaticCoroutine {
    
    // do something first and then multitask via co-yield
    co_await executor.suspend();
    //co_yield;
    
    // finish the task
    //co_return;
  });

  //// create another async task
  //std::future<void> fu = executor.async([]() -> tf::AsyncCoroutine {
  //});
  
  // create another dependent-async task
  tf::AsyncTask task2 = executor.silent_dependent_async([](){
  });

  // resume the task
  executor.resume(task);
  executor.resume(task2);

  return 0;
}





