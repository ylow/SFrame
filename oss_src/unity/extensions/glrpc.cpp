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
#include <util/dense_bitset.hpp>
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
   quiting = false;
 }
  ~param_server() {
    synclock.lock();
    quiting = true;
    cvar.signal();
    synclock.unlock();
    thr.join();
  }

  void init() {
    rmi.reset(new dc_dist_object<param_server>(*dc, this));
    for (procid_t p = 0; p < rmi->numprocs(); ++p) {
      if (p != rmi->procid()) allprocs_except_me.push_back(p);
    }
    
  }

  void add_gradient(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    std::vector<float> adata(data, data + numel);
    locks[elemid].lock();
    all_elements[elemid][rmi->procid()] = adata;
    locks[elemid].unlock();
    synclock.lock();
    bitset.set_bit(elemid);
    cvar.signal(); 
    synclock.unlock();
  }

  void accumulate_gradient(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    std::vector<float> adata(data, data + numel);
    locks[elemid].lock();
    auto& targ = all_elements[elemid][rmi->procid()];
    if (targ.size() != adata.size()) {
      targ = adata;
    } else {
      for (size_t i = 0;i < targ.size(); ++i) targ[i] += adata[i];
    }
    locks[elemid].unlock();
    synclock.lock();
    bitset.set_bit(elemid);
    cvar.signal(); 
    synclock.unlock();
  }

  void sync() {
    rmi->full_barrier();
  }

  void get_elem_to_ptr(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    memcpy((void*)(data), 
           reinterpret_cast<void*>(&(all_elements[elemid][rmi->procid()][0])), 
           all_elements[elemid][rmi->procid()].size() * sizeof(float));

    for (size_t i = 0;i < all_elements[elemid].size(); ++i) {
      if (i == rmi->procid()) continue;
      const auto& e = all_elements[elemid][i];
      if (numel == e.size()) for (size_t j = 0;j < e.size(); ++j) data[j] += e[j];
    }
  }

  void get_elem_to_ptr_average(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    memcpy((void*)(data), 
           reinterpret_cast<void*>(&(all_elements[elemid][rmi->procid()][0])), 
           all_elements[elemid][rmi->procid()].size() * sizeof(float));
    int denom = 1;
    for (size_t i = 0;i < all_elements[elemid].size(); ++i) {
      if (i == rmi->procid()) continue;
      const auto& e = all_elements[elemid][i];
      if (numel == e.size()) {
        ++denom;
        for (size_t j = 0;j < e.size(); ++j) data[j] += e[j];
      }
    }
    for (size_t i = 0;i < numel; ++i) {
      data[i] /= denom;
    }
  }

  std::vector<float> get_elem(size_t elemid) {
    std::vector<float> ret = all_elements[elemid][rmi->procid()];
    for (size_t i = 0;i < all_elements[elemid].size(); ++i) {
      if (i == rmi->procid()) continue;
      const auto& e = all_elements[elemid][i];
      if (ret.size() == e.size()) for (size_t j = 0;j < ret.size(); ++j) ret[j] += e[j];
    }
    return ret;
  }

  void clear(size_t numel) {
    all_elements.resize(numel);
    for (size_t i = 0;i < numel; ++i) all_elements[i].resize(rmi->numprocs());
    locks.resize(numel);
    bitset.resize(numel); bitset.clear();
    thr.launch([&]() {
      synclock.lock();
      while(!quiting) {
        while(!quiting && bitset.popcount() == 0) {
          cvar.wait(synclock);
        }
        if (quiting) break;
        dense_bitset bscopy = bitset;
        synclock.unlock();
        for (size_t elemid : bscopy) {
          bitset.clear_bit(elemid);
          rmi->RPC_CALL(broadcast_call, param_server::set_gradient_remote)
               (allprocs_except_me.begin(), allprocs_except_me.end(), rmi->procid(), elemid, all_elements[elemid][rmi->procid()]);
        }
        synclock.lock();
      }
      synclock.unlock();
    });
  }

  void set_gradient_remote(procid_t source, size_t elemid, const std::vector<float>& data) {
    locks[elemid].lock();
    all_elements[elemid][source] = data;
    locks[elemid].unlock();
  }

  BEGIN_CLASS_MEMBER_REGISTRATION("param_server")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::clear, "numelem")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::get_elem, "elemid")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::get_elem_to_ptr, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::get_elem_to_ptr_average, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::sync)
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::add_gradient, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::accumulate_gradient, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::init)
  END_CLASS_MEMBER_REGISTRATION

 private:
   std::unique_ptr<dc_dist_object<param_server>> rmi;
   std::vector<std::vector<std::vector<float>>> all_elements;
   std::vector<mutex> locks;
   std::vector<procid_t> allprocs_except_me;
   thread thr;
   dense_bitset bitset;
   mutex synclock;
   conditional cvar;
   bool quiting;
   
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
