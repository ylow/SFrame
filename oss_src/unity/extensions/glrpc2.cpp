/**
 * Copyright (C) 2015 Dato, Inc.
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */
#include <string>
#include <vector>
#include <parallel/pthread_tools.hpp>
#include <rpc/dc.hpp>
#include <rpc/dc_init_from_env.hpp>
#include <logger/logger.hpp>
#include <unity/lib/toolkit_function_macros.hpp>
#include <unity/lib/toolkit_class_macros.hpp>
#include <parallel/mutex.hpp>
using namespace graphlab;

distributed_control* dc = nullptr;
void bootstrap_distributed_control() {
  global_logger().set_log_level(LOG_INFO);
  dc = new distributed_control();
}

void stop_distributed_control() {
  delete dc;
  dc = nullptr;
}


void dc_barrier() {
  dc->barrier();
}

flexible_type reduce(flexible_type ret) {
  dc->all_reduce2(ret, [](flexible_type& a, const flexible_type& b) { a += b; });
  return ret;
}

int numprocs() {
  return dc->numprocs();
}
int procid() {
  return dc->procid();
}

class param_server : public graphlab::toolkit_class_base {
 public:
  param_server() {
 }
  ~param_server() {
  }

  void init() {
    rmi.reset(new dc_dist_object<param_server>(*dc, this));
    for (procid_t p = 0; p < rmi->numprocs(); ++p) {
      if (p != rmi->procid()) allprocs_except_me.push_back(p);
    }
    
  }

  void set_gradient(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    std::vector<float> adata(data, data + numel);
    if (elemid % rmi->numprocs() == rmi->procid()) {
      set_gradient_remote(elemid, adata);
    } else {
      rmi->RPC_CALL(remote_call, param_server::set_gradient_remote)(elemid % rmi->numprocs(), elemid, adata); 
    }
  }

  void accumulate_gradient(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    std::vector<float> adata(data, data + numel);
    if (elemid % rmi->numprocs() == rmi->procid()) {
      accumulate_gradient_remote(elemid, adata);
    } else {
      rmi->RPC_CALL(remote_call, param_server::accumulate_gradient_remote)(elemid % rmi->numprocs(), elemid, adata); 
    }
  }

  void sync() {
    rmi->full_barrier();
  }

  void fetch_all() {
    std::vector<request_future<std::vector<float>>> futures(all_elements.size()); 
    for (size_t i = 0;i < all_elements.size(); ++i) {
      if (i % rmi->numprocs() != rmi->procid()) {
        futures[i] = rmi->RPC_CALL(future_remote_request, param_server::get_value_remote)(i % rmi->numprocs(), i);
      }
    }

    for (size_t i = 0;i < all_elements.size(); ++i) {
      if (i % rmi->numprocs() != rmi->procid()) {
        futures[i]();
        locks[i].lock();
        all_elements[i] = std::move(futures[i]());
        locks[i].unlock();
      }
    }
    
  }

  void get_elem_to_ptr(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    memcpy((void*)(data), 
           reinterpret_cast<void*>(&(all_elements[elemid][0])), 
           all_elements[elemid].size() * sizeof(float));
  }

  void clear(size_t numel) {
    all_elements.resize(numel);
    ctr.resize(numel);
    for (size_t i = 0;i < numel; ++i) all_elements[i].resize(rmi->numprocs());
    locks.resize(numel);
  }

  void set_gradient_remote(size_t elemid, const std::vector<float>& data) {
    locks[elemid].lock();
    all_elements[elemid] = data;
    locks[elemid].unlock();
    if (elemid % rmi->numprocs() == rmi->procid()) {
      rmi->RPC_CALL(broadcast_call, param_server::set_gradient_remote)(allprocs_except_me.begin(), allprocs_except_me.end(), elemid, data); 
    }
  }

  void accumulate_gradient_remote(size_t elemid, const std::vector<float>& data) {
    for (size_t i = 0;i < data.size(); ++i) {
      all_elements[elemid][i] += data[i];
    }
    ctr[elemid]++;
    bool tosync = (ctr[elemid] % rmi->numprocs() == 0);
    if (tosync && elemid % rmi->numprocs() == rmi->procid()) {
      rmi->RPC_CALL(broadcast_call, param_server::set_gradient_remote)(allprocs_except_me.begin(), allprocs_except_me.end(), elemid, all_elements[elemid]); 
    }
  }
  std::vector<float> get_value_remote(size_t elemid) {
    auto ret = all_elements[elemid];
    return ret;
  }  

  BEGIN_CLASS_MEMBER_REGISTRATION("param_server")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::clear, "numelem")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::get_elem_to_ptr, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::sync)
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::fetch_all)
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::set_gradient, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::accumulate_gradient, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::init)
  END_CLASS_MEMBER_REGISTRATION

 private:
   std::unique_ptr<dc_dist_object<param_server>> rmi;
   std::vector<std::vector<float>> all_elements;
   std::vector<size_t> ctr;
   std::vector<mutex> locks;
   std::vector<procid_t> allprocs_except_me;
   
};

BEGIN_FUNCTION_REGISTRATION
REGISTER_FUNCTION(bootstrap_distributed_control);
REGISTER_FUNCTION(stop_distributed_control);
REGISTER_NAMED_FUNCTION("barrier", dc_barrier);
REGISTER_FUNCTION(reduce, "val");
REGISTER_FUNCTION(procid);
REGISTER_FUNCTION(numprocs);
END_FUNCTION_REGISTRATION


BEGIN_CLASS_REGISTRATION
REGISTER_CLASS(param_server)
END_CLASS_REGISTRATION
