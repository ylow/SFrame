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
  param_server() { }

  void init() {
    rmi.reset(new dc_dist_object<param_server>(*dc, this));
  }

  void add_gradient(size_t elemid, size_t ptr_to_data, size_t numel) {
    float* data = reinterpret_cast<float*>(ptr_to_data);
    std::vector<float> data_copy(data, data + numel);
    procid_t target_machine = elemid % rmi->numprocs();

    std::vector<float> delta_gradient(data_copy.size());
    if (local_contrib[elemid].size() == 0) {
      delta_gradient = data_copy;
    } else {
      for (size_t i = 0;i < data_copy.size(); ++i) {
        delta_gradient[i] = data_copy[i] - local_contrib[elemid][i];
      }
    }
    local_contrib[elemid] = data_copy;

    if (target_machine == rmi->procid()) {
      accumulate_gradient(elemid, delta_gradient);
    } else {
      rmi->RPC_CALL(remote_call, param_server::accumulate_gradient)(target_machine, elemid, delta_gradient);
    }
  }

  void sync() {
    rmi->full_barrier();
    clear(all_elements.size());
  }

  std::vector<float> get_elem(size_t elemid) {
    return all_elements[elemid];
  }

  void clear(size_t numel) {
    all_elements.clear();
    all_elements.resize(numel);
    num_updates.resize(numel);
    local_contrib.resize(numel);
  }
  void set_gradient(size_t elemid, const std::vector<float>& data) {
    all_elements[elemid] = data;
  }

  void accumulate_gradient(size_t elemid, const std::vector<float>& data) {
    if (elemid >= all_elements.size()) {
      std::cerr << "BAD no such elem!" << std::endl;
      return;
    }
    std::vector<float>& localelem = all_elements[elemid];
    if (localelem.size() > 0 && localelem.size() != data.size()) {
      std::cerr << "BAD. Size mismatch!" << std::endl;
      return;
    } else if (localelem.size() == 0) {
      localelem = data;
    } else {
      for (size_t i = 0;i < data.size(); ++i) localelem[i] += data[i];
    }

    int n = __sync_fetch_and_add(&(num_updates[elemid]), 1);
    // async broadcast
    if (n % rmi->numprocs() == 0) {
      for (size_t p = 0; p < rmi->numprocs(); ++p) {
        if (p != rmi->procid()) rmi->RPC_CALL(remote_call, param_server::set_gradient)(p, elemid, localelem);
      }
    }
  }

  BEGIN_CLASS_MEMBER_REGISTRATION("param_server")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::clear, "numelem")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::get_elem, "elemid")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::sync)
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::add_gradient, "elemid", "ptr", "len")
  REGISTER_CLASS_MEMBER_FUNCTION(param_server::init)
  END_CLASS_MEMBER_REGISTRATION

 private:
   std::unique_ptr<dc_dist_object<param_server>> rmi;
   std::vector<std::vector<float>> all_elements;
   std::vector<int> num_updates;
   std::vector<std::vector<float>> local_contrib;
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
