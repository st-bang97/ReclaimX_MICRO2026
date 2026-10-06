// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, George L. Yuan,
// Ali Bakhoda, Andrew Turner, Ivan Sham, Vijay Kandiah, Nikos Hardavellas,
// Mahmoud Khairy, Junrui Pan, Timothy G. Rogers
// The University of British Columbia, Northwestern University, Purdue
// University All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "gpu-sim.h"

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include "zlib.h"

#include "dram.h"
#include "mem_fetch.h"
#include "shader.h"
#include "shader_trace.h"
#include <unordered_map>
#include <algorithm>
#include <time.h>
#include "addrdec.h"
#include "delayqueue.h"
#include "dram.h"
#include "gpu-cache.h"
#include "gpu-misc.h"
#include "icnt_wrapper.h"
#include "l2cache.h"
#include "shader.h"
#include "stat-tool.h"

#include "../../libcuda/gpgpu_context.h"
#include "../abstract_hardware_model.h"
#include "../cuda-sim/cuda-sim.h"
#include "../cuda-sim/cuda_device_runtime.h"
#include "../cuda-sim/ptx-stats.h"
#include "../cuda-sim/ptx_ir.h"
#include "../debug.h"
#include "../gpgpusim_entrypoint.h"
#include "../statwrapper.h"
#include "../trace.h"
#include "assert.h"
#include "mem_latency_stat.h"
#include "power_stat.h"
#include "stats.h"
#include "visualizer.h"
#ifdef GPGPUSIM_POWER_MODEL
#include "power_interface.h"
#else
class gpgpu_sim_wrapper {};
#endif

#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <sstream>

void print_caller_info() {
  const int max_frames = 10;
  void *buffer[max_frames];
  int nptrs = backtrace(buffer, max_frames);
  char **symbols = backtrace_symbols(buffer, nptrs);

  if (symbols != nullptr && nptrs >= 2) {
    std::cout << "Caller function: " << symbols[2] << std::endl;
  }

  free(symbols);
}
bool g_interactive_debugger_enabled = false;

bool sim_prof_enable = false;
bool util_prof = false;
void bst_debug_print(const char *format, ...) {
#ifdef BST_DEBUG
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
  fflush(stdout);
#endif
}
std::list<shd_warp_t *> all_warps;
std::list<shd_warp_t *> fail_warps;
bool skip_cycles = false;
bool skip_cycles_enable = false;
uint64_t skipped_cycles = 0;
uint64_t gpu_tot_skipped_cycle = 0;

std::map<unsigned long long, std::list<event_stats *>> sim_prof;

void print_sim_prof(FILE *fout, float freq) {
  freq /= 1000;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->print(fout, freq);
    }
  }
}

unsigned long long kernel_time = 0;
unsigned long long memory_copy_time_h2d = 0;
unsigned long long memory_copy_time_d2h = 0;
unsigned long long prefetch_time = 0;
unsigned long long devicesync_time = 0;
unsigned long long writeback_time = 0;
unsigned long long dma_time = 0;
unsigned long long comp_time = 0;
unsigned long long decomp_time = 0;

// unsigned long long gpu_sim_cycle = 0;
// unsigned long long gpu_tot_sim_cycle = 0;

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu) {
  memory_copy_time_h2d =
      0;  // BST edited for sim prof calculation duplicated add
  memory_copy_time_d2h =
      0;                // BST edited for sim prof calculation duplicated add
  prefetch_time = 0;    // BST edited for sim prof calculation duplicated add
  devicesync_time = 0;  // BST edited for sim prof calculation duplicated add
  writeback_time = 0;   // BST edited for sim prof calculation duplicated add
  dma_time = 0;         // BST edited for sim prof calculation duplicated add
  kernel_time = 0;      // BST edited for sim prof calculation duplicated add
  comp_time = 0;        // BST edited for sim prof calculation duplicated add
  decomp_time = 0;      // BST edited for sim prof calculation duplicated add
  float freq = gpu->shader_clock() / 1000.0;
  unsigned long long cur_cycle_cal =
      gpu->gpu_sim_cycle + gpu->gpu_tot_sim_cycle;
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      (*iter2)->calculate();
    }
  }

  unsigned long long page_fault_time = 0;
  if (!gpu->get_config().enable_accurate_simulation) {
    page_fault_time = gpu->m_new_stats->mf_page_fault_outstanding *
                      gpu->get_config().page_fault_latency;
  }

  fprintf(fout, "Tot_prefetch_time: %llu(cycle), %f(us)\n", prefetch_time,
          ((float)prefetch_time) / freq);
  fprintf(fout, "Tot_kernel_exec_time: %llu(cycle), %f(us)\n", kernel_time,
          ((float)kernel_time) / freq);

  if (!gpu->get_config().enable_accurate_simulation) {
    fprintf(fout, "Tot_kernel_exec_time_and_fault_time: %llu(cycle), %f(us)\n",
            kernel_time + page_fault_time,
            ((float)(kernel_time + page_fault_time)) / freq);
  }

  fprintf(fout, "Tot_memcpy_h2d_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d, ((float)memory_copy_time_h2d) / freq);
  fprintf(fout, "Tot_memcpy_d2h_time: %llu(cycle), %f(us)\n",
          memory_copy_time_d2h, ((float)memory_copy_time_d2h) / freq);
  fprintf(fout, "Tot_memcpy_time: %llu(cycle), %f(us)\n",
          memory_copy_time_h2d + memory_copy_time_d2h,
          ((float)(memory_copy_time_h2d + memory_copy_time_d2h)) / freq);
  fprintf(fout, "Tot_devicesync_time: %llu(cycle), %f(us)\n", devicesync_time,
          ((float)devicesync_time) / freq);
  fprintf(fout, "Tot_writeback_time: %llu(cycle), %f(us)\n", writeback_time,
          ((float)writeback_time) / freq);
  fprintf(fout, "Tot_dma_time: %llu(cycle), %f(us)\n", dma_time,
          ((float)dma_time) / freq);
  fprintf(fout, "Tot_decomp_time: %llu(cycle), %f(us)\n", decomp_time,
          ((float)decomp_time) / freq);
  fprintf(fout, "Tot_comp_time: %llu(cycle), %f(us)\n", comp_time,
          ((float)comp_time) / freq);
  fprintf(fout, "Tot_memcpy_d2h_sync_wb_time: %llu(cycle), %f(us)\n",
          writeback_time + devicesync_time + memory_copy_time_d2h,
          ((float)(writeback_time + devicesync_time + memory_copy_time_d2h) /
           freq));
}

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == kernel_launched &&
          ((kernel_stats *)(*iter2))->kernel_id == kernel_id) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == prefetch &&
          ((memory_stats *)(*iter2))->start_addr == start_addr &&
          ((memory_stats *)(*iter2))->size == size) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void update_sim_prof_prefetch_break_down(unsigned long long end_time) {
  for (std::map<unsigned long long, std::list<event_stats *>>::reverse_iterator
           iter = sim_prof.rbegin();
       iter != sim_prof.rend(); iter++) {
    for (std::list<event_stats *>::reverse_iterator iter2 =
             iter->second.rbegin();
         iter2 != iter->second.rend(); iter2++) {
      if ((*iter2)->type == prefetch_breakdown) {
        (*iter2)->end_time = end_time;
        return;
      }
    }
  }
}

void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout) {
  new_stats->print(stdout);
  if (gpu->get_config().collect_access_trace) {
    FILE *f4 = fopen("access.txt", "w");

    new_stats->print_time_and_access(f4);

  fclose(f4);
  }
  if (sim_prof_enable) {
    print_sim_prof(stdout, gpu->shader_clock());
    calculate_sim_prof(stdout, gpu);
  }
}

tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

/* Clock Domains */

#define CORE 0x01
#define L2 0x02
#define DRAM 0x04
#define ICNT 0x08
#define GMMU 0x10

#define MEM_LATENCY_STAT_IMPL

#include "mem_latency_stat.h"

void power_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-accelwattch_xml_file", OPT_CSTR,
                         &g_power_config_name, "AccelWattch XML file",
                         "accelwattch_sass_sim.xml");

  option_parser_register(opp, "-power_simulation_enabled", OPT_BOOL,
                         &g_power_simulation_enabled,
                         "Turn on power simulator (1=On, 0=Off)", "0");

  option_parser_register(opp, "-power_per_cycle_dump", OPT_BOOL,
                         &g_power_per_cycle_dump,
                         "Dump detailed power output each cycle", "0");

  option_parser_register(opp, "-hw_perf_file_name", OPT_CSTR,
                         &g_hw_perf_file_name,
                         "Hardware Performance Statistics file", "hw_perf.csv");

  option_parser_register(
      opp, "-hw_perf_bench_name", OPT_CSTR, &g_hw_perf_bench_name,
      "Kernel Name in Hardware Performance Statistics file", "");

  option_parser_register(opp, "-power_simulation_mode", OPT_INT32,
                         &g_power_simulation_mode,
                         "Switch performance counter input for power "
                         "simulation (0=Sim, 1=HW, 2=HW-Sim Hybrid)",
                         "0");

  option_parser_register(opp, "-dvfs_enabled", OPT_BOOL, &g_dvfs_enabled,
                         "Turn on DVFS for power model", "0");
  option_parser_register(opp, "-aggregate_power_stats", OPT_BOOL,
                         &g_aggregate_power_stats,
                         "Accumulate power across all kernels", "0");

  // Accelwattch Hyrbid Configuration

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RH],
      "Get L1 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_RM],
      "Get L1 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WH],
      "Get L1 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L1_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L1_WM],
      "Get L1 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RH],
      "Get L2 Read Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_RM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_RM],
      "Get L2 Read Misses for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WH", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WH],
      "Get L2 Write Hits for Accelwattch-Hybrid from Accel-Sim", "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_L2_WM", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_L2_WM],
      "Get L2 Write Misses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CC_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CC_ACC],
      "Get Constant Cache Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_SHARED_ACC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_SHRD_ACC],
      "Get Shared Memory Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(opp, "-accelwattch_hybrid_perfsim_DRAM_RD", OPT_BOOL,
                         &accelwattch_hybrid_configuration[HW_DRAM_RD],
                         "Get DRAM Reads for Accelwattch-Hybrid from Accel-Sim",
                         "0");
  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_DRAM_WR", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_DRAM_WR],
      "Get DRAM Writes for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NOC", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NOC],
      "Get Interconnect Acesses for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_PIPE_DUTY", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_PIPE_DUTY],
      "Get Pipeline Duty Cycle Acesses for Accelwattch-Hybrid from Accel-Sim",
      "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_NUM_SM_IDLE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_NUM_SM_IDLE],
      "Get Number of Idle SMs for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_CYCLES", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_CYCLES],
      "Get Executed Cycles for Accelwattch-Hybrid from Accel-Sim", "0");

  option_parser_register(
      opp, "-accelwattch_hybrid_perfsim_VOLTAGE", OPT_BOOL,
      &accelwattch_hybrid_configuration[HW_VOLTAGE],
      "Get Chip Voltage for Accelwattch-Hybrid from Accel-Sim", "0");

  // Output Data Formats
  option_parser_register(
      opp, "-power_trace_enabled", OPT_BOOL, &g_power_trace_enabled,
      "produce a file for the power trace (1=On, 0=Off)", "0");

  option_parser_register(
      opp, "-power_trace_zlevel", OPT_INT32, &g_power_trace_zlevel,
      "Compression level of the power trace output log (0=no comp, 9=highest)",
      "6");

  option_parser_register(
      opp, "-steady_power_levels_enabled", OPT_BOOL,
      &g_steady_power_levels_enabled,
      "produce a file for the steady power levels (1=On, 0=Off)", "0");

  option_parser_register(opp, "-steady_state_definition", OPT_CSTR,
                         &gpu_steady_state_definition,
                         "allowed deviation:number of samples", "8:4");
}

void memory_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_perf_sim_memcpy", OPT_BOOL,
                         &m_perf_sim_memcpy, "Fill the L2 cache on memcpy",
                         "1");
  option_parser_register(opp, "-gpgpu_simple_dram_model", OPT_BOOL,
                         &simple_dram_model,
                         "simple_dram_model with fixed latency and BW", "0");
  option_parser_register(opp, "-gpgpu_dram_scheduler", OPT_INT32,
                         &scheduler_type, "0 = fifo, 1 = FR-FCFS (defaul)",
                         "1");
  option_parser_register(opp, "-gpgpu_dram_partition_queues", OPT_CSTR,
                         &gpgpu_L2_queue_config, "i2$:$2d:d2$:$2i", "8:8:8:8");

  option_parser_register(opp, "-l2_ideal", OPT_BOOL, &l2_ideal,
                         "Use a ideal L2 cache that always hit", "0");
  option_parser_register(opp, "-gpgpu_cache:dl2", OPT_CSTR,
                         &m_L2_config.m_config_string,
                         "unified banked L2 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>}",
                         "64:128:8,L:B:m:N,A:16:4,4");
  option_parser_register(opp, "-gpgpu_cache:dl2_texture_only", OPT_BOOL,
                         &m_L2_texure_only, "L2 cache used for texture only",
                         "1");
  option_parser_register(
      opp, "-gpgpu_n_mem", OPT_UINT32, &m_n_mem,
      "number of memory modules (e.g. memory controllers) in gpu", "8");
  option_parser_register(opp, "-gpgpu_n_sub_partition_per_mchannel", OPT_UINT32,
                         &m_n_sub_partition_per_memory_channel,
                         "number of memory subpartition in each memory module",
                         "1");
  option_parser_register(opp, "-gpgpu_n_mem_per_ctrlr", OPT_UINT32,
                         &gpu_n_mem_per_ctrlr,
                         "number of memory chips per memory controller", "1");
  option_parser_register(opp, "-gpgpu_memlatency_stat", OPT_INT32,
                         &gpgpu_memlatency_stat,
                         "track and display latency statistics 0x2 enables MC, "
                         "0x4 enables queue logs",
                         "0");
  option_parser_register(opp, "-gpgpu_frfcfs_dram_sched_queue_size", OPT_INT32,
                         &gpgpu_frfcfs_dram_sched_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_return_queue_size", OPT_INT32,
                         &gpgpu_dram_return_queue_size,
                         "0 = unlimited (default); # entries per chip", "0");
  option_parser_register(opp, "-gpgpu_dram_buswidth", OPT_UINT32, &busW,
                         "default = 4 bytes (8 bytes per cycle at DDR)", "4");
  option_parser_register(
      opp, "-gpgpu_dram_burst_length", OPT_UINT32, &BL,
      "Burst length of each DRAM request (default = 4 data bus cycle)", "4");
  option_parser_register(opp, "-dram_data_command_freq_ratio", OPT_UINT32,
                         &data_command_freq_ratio,
                         "Frequency ratio between DRAM data bus and command "
                         "bus (default = 2 times, i.e. DDR)",
                         "2");
  option_parser_register(
      opp, "-gpgpu_dram_timing_opt", OPT_CSTR, &gpgpu_dram_timing_opt,
      "DRAM timing parameters = "
      "{nbk:tCCD:tRRD:tRCD:tRAS:tRP:tRC:CL:WL:tCDLR:tWR:nbkgrp:tCCDL:tRTPL}",
      "4:2:8:12:21:13:34:9:4:5:13:1:0:0");
  option_parser_register(opp, "-gpgpu_l2_rop_latency", OPT_UINT32, &rop_latency,
                         "ROP queue latency (default 85)", "85");
  option_parser_register(opp, "-dram_latency", OPT_UINT32, &dram_latency,
                         "DRAM latency (default 30)", "30");
  option_parser_register(opp, "-dram_dual_bus_interface", OPT_UINT32,
                         &dual_bus_interface,
                         "dual_bus_interface (default = 0) ", "0");
  option_parser_register(opp, "-dram_bnk_indexing_policy", OPT_UINT32,
                         &dram_bnk_indexing_policy,
                         "dram_bnk_indexing_policy (0 = normal indexing, 1 = "
                         "Xoring with the higher bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_bnkgrp_indexing_policy", OPT_UINT32,
                         &dram_bnkgrp_indexing_policy,
                         "dram_bnkgrp_indexing_policy (0 = take higher bits, 1 "
                         "= take lower bits) (Default = 0)",
                         "0");
  option_parser_register(opp, "-dram_seperate_write_queue_enable", OPT_BOOL,
                         &seperate_write_queue_enabled,
                         "Seperate_Write_Queue_Enable", "0");
  option_parser_register(opp, "-dram_write_queue_size", OPT_CSTR,
                         &write_queue_size_opt, "Write_Queue_Size", "32:28:16");
  option_parser_register(
      opp, "-dram_elimnate_rw_turnaround", OPT_BOOL, &elimnate_rw_turnaround,
      "elimnate_rw_turnaround i.e set tWTR and tRTW = 0", "0");
  option_parser_register(opp, "-icnt_flit_size", OPT_UINT32, &icnt_flit_size,
                         "icnt_flit_size", "32");
  m_address_mapping.addrdec_setoption(opp);
}

void shader_core_config::reg_options(class OptionParser *opp) {
  option_parser_register(opp, "-gpgpu_simd_model", OPT_INT32, &model,
                         "1 = post-dominator", "1");
  option_parser_register(
      opp, "-gpgpu_shader_core_pipeline", OPT_CSTR,
      &gpgpu_shader_core_pipeline_opt,
      "shader core pipeline config, i.e., {<nthread>:<warpsize>}", "1024:32");
  option_parser_register(opp, "-gpgpu_tex_cache:l1", OPT_CSTR,
                         &m_L1T_config.m_config_string,
                         "per-shader L1 texture cache  (READ-ONLY) config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>:<rf>}",
                         "8:128:5,L:R:m:N,F:128:4,128:2");
  option_parser_register(
      opp, "-gpgpu_const_cache:l1", OPT_CSTR, &m_L1C_config.m_config_string,
      "per-shader L1 constant memory cache  (READ-ONLY) config "
      " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_alloc>,<mshr>:<N>:<"
      "merge>,<mq>} ",
      "64:64:2,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:il1", OPT_CSTR,
                         &m_L1I_config.m_config_string,
                         "shader L1 instruction cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq>} ",
                         "4:256:4,L:R:f:N,A:2:32,4");
  option_parser_register(opp, "-gpgpu_cache:dl1", OPT_CSTR,
                         &m_L1D_config.m_config_string,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_l1_cache_write_ratio", OPT_UINT32,
                         &m_L1D_config.m_wr_percent, "L1D write ratio", "0");
  option_parser_register(opp, "-gpgpu_l1_banks", OPT_UINT32,
                         &m_L1D_config.l1_banks, "The number of L1 cache banks",
                         "1");
  option_parser_register(opp, "-gpgpu_l1_banks_byte_interleaving", OPT_UINT32,
                         &m_L1D_config.l1_banks_byte_interleaving,
                         "l1 banks byte interleaving granularity", "32");
  option_parser_register(opp, "-gpgpu_l1_banks_hashing_function", OPT_UINT32,
                         &m_L1D_config.l1_banks_hashing_function,
                         "l1 banks hashing function", "0");
  option_parser_register(opp, "-gpgpu_l1_latency", OPT_UINT32,
                         &m_L1D_config.l1_latency, "L1 Hit Latency", "1");
  option_parser_register(opp, "-gpgpu_smem_latency", OPT_UINT32, &smem_latency,
                         "smem Latency", "3");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefL1", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefL1,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_cache:dl1PrefShared", OPT_CSTR,
                         &m_L1D_config.m_config_stringPrefShared,
                         "per-shader L1 data cache config "
                         " {<nsets>:<bsize>:<assoc>,<rep>:<wr>:<alloc>:<wr_"
                         "alloc>,<mshr>:<N>:<merge>,<mq> | none}",
                         "none");
  option_parser_register(opp, "-gpgpu_gmem_skip_L1D", OPT_BOOL, &gmem_skip_L1D,
                         "global memory access skip L1D cache (implements "
                         "-Xptxas -dlcm=cg, default=no skip)",
                         "0");

  option_parser_register(opp, "-gpgpu_perfect_mem", OPT_BOOL,
                         &gpgpu_perfect_mem,
                         "enable perfect memory mode (no cache miss)", "0");
  option_parser_register(
      opp, "-n_regfile_gating_group", OPT_UINT32, &n_regfile_gating_group,
      "group of lanes that should be read/written together)", "4");
  option_parser_register(
      opp, "-gpgpu_clock_gated_reg_file", OPT_BOOL, &gpgpu_clock_gated_reg_file,
      "enable clock gated reg file for power calculations", "0");
  option_parser_register(
      opp, "-gpgpu_clock_gated_lanes", OPT_BOOL, &gpgpu_clock_gated_lanes,
      "enable clock gated lanes for power calculations", "0");
  option_parser_register(opp, "-gpgpu_shader_registers", OPT_UINT32,
                         &gpgpu_shader_registers,
                         "Number of registers per shader core. Limits number "
                         "of concurrent CTAs. (default 8192)",
                         "8192");
  option_parser_register(
      opp, "-gpgpu_registers_per_block", OPT_UINT32, &gpgpu_registers_per_block,
      "Maximum number of registers per CTA. (default 8192)", "8192");
  option_parser_register(opp, "-gpgpu_ignore_resources_limitation", OPT_BOOL,
                         &gpgpu_ignore_resources_limitation,
                         "gpgpu_ignore_resources_limitation (default 0)", "0");
  option_parser_register(
      opp, "-gpgpu_shader_cta", OPT_UINT32, &max_cta_per_core,
      "Maximum number of concurrent CTAs in shader (default 8)", "8");
  option_parser_register(
      opp, "-gpgpu_num_cta_barriers", OPT_UINT32, &max_barriers_per_cta,
      "Maximum number of named barriers per CTA (default 16)", "16");
  option_parser_register(opp, "-gpgpu_n_clusters", OPT_UINT32, &n_simt_clusters,
                         "number of processing clusters", "10");
  option_parser_register(opp, "-ETC_throttle_SM", OPT_BOOL, &ETC_throttle_SM,
                         "ETC_throttle_SM", "0");
  option_parser_register(opp, "-gpgpu_n_cores_per_cluster", OPT_UINT32,
                         &n_simt_cores_per_cluster,
                         "number of simd cores per cluster", "3");
  option_parser_register(opp, "-CURE_throttle_SM", OPT_BOOL, &CURE_throttle_SM,
                         "CURE_throttle_SM", "0");
  option_parser_register(opp, "-gpgpu_n_cluster_ejection_buffer_size",
                         OPT_UINT32, &n_simt_ejection_buffer_size,
                         "number of packets in ejection buffer", "8");
  option_parser_register(
      opp, "-gpgpu_n_ldst_response_buffer_size", OPT_UINT32,
      &ldst_unit_response_queue_size,
      "number of response packets in ld/st unit ejection buffer", "2");
  option_parser_register(
      opp, "-gpgpu_shmem_per_block", OPT_UINT32, &gpgpu_shmem_per_block,
      "Size of shared memory per thread block or CTA (default 48kB)", "49152");
  option_parser_register(
      opp, "-gpgpu_shmem_size", OPT_UINT32, &gpgpu_shmem_size,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_option", OPT_CSTR,
                         &gpgpu_shmem_option,
                         "Option list of shared memory sizes", "0");
  option_parser_register(
      opp, "-gpgpu_unified_l1d_size", OPT_UINT32,
      &m_L1D_config.m_unified_cache_size,
      "Size of unified data cache(L1D + shared memory) in KB", "0");
  option_parser_register(opp, "-gpgpu_adaptive_cache_config", OPT_BOOL,
                         &adaptive_cache_config, "adaptive_cache_config", "0");
  option_parser_register(
      opp, "-gpgpu_shmem_sizeDefault", OPT_UINT32, &gpgpu_shmem_sizeDefault,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_size_PrefL1", OPT_UINT32, &gpgpu_shmem_sizePrefL1,
      "Size of shared memory per shader core (default 16kB)", "16384");
  option_parser_register(opp, "-gpgpu_shmem_size_PrefShared", OPT_UINT32,
                         &gpgpu_shmem_sizePrefShared,
                         "Size of shared memory per shader core (default 16kB)",
                         "16384");
  option_parser_register(
      opp, "-gpgpu_shmem_num_banks", OPT_UINT32, &num_shmem_bank,
      "Number of banks in the shared memory in each shader core (default 16)",
      "16");
  option_parser_register(
      opp, "-gpgpu_shmem_limited_broadcast", OPT_BOOL, &shmem_limited_broadcast,
      "Limit shared memory to do one broadcast per cycle (default on)", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_mem_unit_ports", OPT_INT32, &mem_unit_ports,
      "The number of memory transactions allowed per core cycle", "1");
  option_parser_register(opp, "-gpgpu_shmem_warp_parts", OPT_INT32,
                         &mem_warp_parts,
                         "Number of portions a warp is divided into for shared "
                         "memory bank conflict check ",
                         "2");
  option_parser_register(
      opp, "-gpgpu_warpdistro_shader", OPT_INT32, &gpgpu_warpdistro_shader,
      "Specify which shader core to collect the warp size distribution from",
      "-1");
  option_parser_register(
      opp, "-gpgpu_warp_issue_shader", OPT_INT32, &gpgpu_warp_issue_shader,
      "Specify which shader core to collect the warp issue distribution from",
      "0");
  option_parser_register(opp, "-gpgpu_local_mem_map", OPT_BOOL,
                         &gpgpu_local_mem_map,
                         "Mapping from local memory space address to simulated "
                         "GPU physical address space (default = enabled)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_reg_banks", OPT_INT32,
                         &gpgpu_num_reg_banks,
                         "Number of register banks (default = 8)", "8");
  option_parser_register(
      opp, "-gpgpu_reg_bank_use_warp_id", OPT_BOOL, &gpgpu_reg_bank_use_warp_id,
      "Use warp ID in mapping registers to banks (default = off)", "0");
  option_parser_register(opp, "-gpgpu_sub_core_model", OPT_BOOL,
                         &sub_core_model,
                         "Sub Core Volta/Pascal model (default = off)", "0");
  option_parser_register(opp, "-gpgpu_enable_specialized_operand_collector",
                         OPT_BOOL, &enable_specialized_operand_collector,
                         "enable_specialized_operand_collector", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sp,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_units_dp,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_units_sfu,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_int",
                         OPT_INT32, &gpgpu_operand_collector_num_units_int,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_tensor_core",
                         OPT_INT32,
                         &gpgpu_operand_collector_num_units_tensor_core,
                         "number of collector units (default = 4)", "4");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_units_mem,
                         "number of collector units (default = 2)", "2");
  option_parser_register(opp, "-gpgpu_operand_collector_num_units_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_units_gen,
                         "number of collector units (default = 0)", "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_in_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_in_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_in_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_in_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sp,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_dp",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_dp,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_sfu",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_sfu,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_int",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_int,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_operand_collector_num_out_ports_tensor_core", OPT_INT32,
      &gpgpu_operand_collector_num_out_ports_tensor_core,
      "number of collector unit in ports (default = 1)", "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_mem",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_mem,
                         "number of collector unit in ports (default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_operand_collector_num_out_ports_gen",
                         OPT_INT32, &gpgpu_operand_collector_num_out_ports_gen,
                         "number of collector unit in ports (default = 0)",
                         "0");
  option_parser_register(opp, "-gpgpu_coalesce_arch", OPT_INT32,
                         &gpgpu_coalesce_arch,
                         "Coalescing arch (GT200 = 13, Fermi = 20)", "13");
  option_parser_register(opp, "-gpgpu_num_sched_per_core", OPT_INT32,
                         &gpgpu_num_sched_per_core,
                         "Number of warp schedulers per core", "1");
  option_parser_register(opp, "-gpgpu_max_insn_issue_per_warp", OPT_INT32,
                         &gpgpu_max_insn_issue_per_warp,
                         "Max number of instructions that can be issued per "
                         "warp in one cycle by scheduler (either 1 or 2)",
                         "2");
  option_parser_register(opp, "-gpgpu_dual_issue_diff_exec_units", OPT_BOOL,
                         &gpgpu_dual_issue_diff_exec_units,
                         "should dual issue use two different execution unit "
                         "resources (Default = 1)",
                         "1");
  option_parser_register(opp, "-gpgpu_simt_core_sim_order", OPT_INT32,
                         &simt_core_sim_order,
                         "Select the simulation order of cores in a cluster "
                         "(0=Fix, 1=Round-Robin)",
                         "1");
  option_parser_register(
      opp, "-gpgpu_pipeline_widths", OPT_CSTR, &pipeline_widths_string,
      "Pipeline widths "
      "ID_OC_SP,ID_OC_DP,ID_OC_INT,ID_OC_SFU,ID_OC_MEM,OC_EX_SP,OC_EX_DP,OC_EX_"
      "INT,OC_EX_SFU,OC_EX_MEM,EX_WB,ID_OC_TENSOR_CORE,OC_EX_TENSOR_CORE",
      "1,1,1,1,1,1,1,1,1,1,1,1,1");
  option_parser_register(opp, "-gpgpu_tensor_core_avail", OPT_INT32,
                         &gpgpu_tensor_core_avail,
                         "Tensor Core Available (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sp_units", OPT_INT32,
                         &gpgpu_num_sp_units, "Number of SP units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_dp_units", OPT_INT32,
                         &gpgpu_num_dp_units, "Number of DP units (default=0)",
                         "0");
  option_parser_register(opp, "-gpgpu_num_int_units", OPT_INT32,
                         &gpgpu_num_int_units,
                         "Number of INT units (default=0)", "0");
  option_parser_register(opp, "-gpgpu_num_sfu_units", OPT_INT32,
                         &gpgpu_num_sfu_units, "Number of SF units (default=1)",
                         "1");
  option_parser_register(opp, "-gpgpu_num_tensor_core_units", OPT_INT32,
                         &gpgpu_num_tensor_core_units,
                         "Number of tensor_core units (default=1)", "0");
  option_parser_register(
      opp, "-gpgpu_num_mem_units", OPT_INT32, &gpgpu_num_mem_units,
      "Number if ldst units (default=1) WARNING: not hooked up to anything",
      "1");
  option_parser_register(
      opp, "-gpgpu_scheduler", OPT_CSTR, &gpgpu_scheduler_string,
      "Scheduler configuration: < lrr | gto | two_level_active > "
      "If "
      "two_level_active:<num_active_warps>:<inner_prioritization>:<outer_"
      "prioritization>"
      "For complete list of prioritization values see shader.h enum "
      "scheduler_prioritization_type"
      "Default: gto",
      "gto");

  option_parser_register(
      opp, "-gpgpu_concurrent_kernel_sm", OPT_BOOL, &gpgpu_concurrent_kernel_sm,
      "Support concurrent kernels on a SM (default = disabled)", "0");
  option_parser_register(opp, "-gpgpu_perfect_inst_const_cache", OPT_BOOL,
                         &perfect_inst_const_cache,
                         "perfect inst and const cache mode, so all inst and "
                         "const hits in the cache(default = disabled)",
                         "0");
  option_parser_register(
      opp, "-gpgpu_inst_fetch_throughput", OPT_INT32, &inst_fetch_throughput,
      "the number of fetched intruction per warp each cycle", "1");
  option_parser_register(opp, "-gpgpu_reg_file_port_throughput", OPT_INT32,
                         &reg_file_port_throughput,
                         "the number ports of the register file", "1");
  option_parser_register(opp, "-tlb_size", OPT_INT32, &tlb_size,
                       "Number of tlb entries per SM.", "4096");
  for (unsigned j = 0; j < SPECIALIZED_UNIT_NUM; ++j) {
    std::stringstream ss;
    ss << "-specialized_unit_" << j + 1;
    option_parser_register(opp, ss.str().c_str(), OPT_CSTR,
                           &specialized_unit_string[j],
                           "specialized unit config"
                           " {<enabled>,<num_units>:<latency>:<initiation>,<ID_"
                           "OC_SPEC>:<OC_EX_SPEC>,<NAME>}",
                           "0,4,4,4,4,BRA");
  }
}

void gpgpu_sim_config::reg_options(option_parser_t opp) {
  gpgpu_functional_sim_config::reg_options(opp);
  m_shader_config.reg_options(opp);
  m_memory_config.reg_options(opp);
  power_config::reg_options(opp);
  option_parser_register(opp, "-gpgpu_max_cycle", OPT_INT64, &gpu_max_cycle_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_insn", OPT_INT64, &gpu_max_insn_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_cta", OPT_INT32, &gpu_max_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(opp, "-gpgpu_max_completed_cta", OPT_INT32,
                         &gpu_max_completed_cta_opt,
                         "terminates gpu simulation early (0 = no limit)", "0");
  option_parser_register(
      opp, "-gpgpu_runtime_stat", OPT_CSTR, &gpgpu_runtime_stat,
      "display runtime statistics such as dram utilization {<freq>:<flag>}",
      "10000:0");
  option_parser_register(opp, "-liveness_message_freq", OPT_INT64,
                         &liveness_message_freq,
                         "Minimum number of seconds between simulation "
                         "liveness messages (0 = always print)",
                         "1");
  option_parser_register(opp, "-gpgpu_compute_capability_major", OPT_UINT32,
                         &gpgpu_compute_capability_major,
                         "Major compute capability version number", "7");
  option_parser_register(opp, "-gpgpu_compute_capability_minor", OPT_UINT32,
                         &gpgpu_compute_capability_minor,
                         "Minor compute capability version number", "0");
  option_parser_register(opp, "-gpgpu_flush_l1_cache", OPT_BOOL,
                         &gpgpu_flush_l1_cache,
                         "Flush L1 cache at the end of each kernel call", "0");
  option_parser_register(opp, "-gpgpu_flush_l2_cache", OPT_BOOL,
                         &gpgpu_flush_l2_cache,
                         "Flush L2 cache at the end of each kernel call", "0");
  option_parser_register(
      opp, "-gpgpu_deadlock_detect", OPT_BOOL, &gpu_deadlock_detect,
      "Stop the simulation at deadlock (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-progress_print", OPT_BOOL, &progress_print,
      "print medium result in progress (1=on (default), 0=off)", "1");
  option_parser_register(
      opp, "-gpgpu_ptx_instruction_classification", OPT_INT32,
      &(gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification),
      "if enabled will classify ptx instruction types per kernel (Max 255 "
      "kernels now)",
      "0");
  option_parser_register(
      opp, "-gpgpu_ptx_sim_mode", OPT_INT32,
      &(gpgpu_ctx->func_sim->g_ptx_sim_mode),
      "Select between Performance (default) or Functional simulation (1)", "0");
  option_parser_register(opp, "-gpgpu_clock_domains", OPT_CSTR,
                         &gpgpu_clock_domains,
                         "Clock Domain Frequencies in MhZ {<Core Clock>:<ICNT "
                         "Clock>:<L2 Clock>:<DRAM Clock>}",
                         "500.0:2000.0:2000.0:2000.0");
  option_parser_register(
      opp, "-gpgpu_max_concurrent_kernel", OPT_INT32, &max_concurrent_kernel,
      "maximum kernels that can run concurrently on GPU, set this value "
      "according to max resident grids for your compute capability",
      "32");
  option_parser_register(
      opp, "-gpgpu_cflog_interval", OPT_INT32, &gpgpu_cflog_interval,
      "Interval between each snapshot in control flow logger", "0");
  option_parser_register(opp, "-visualizer_enabled", OPT_BOOL,
                         &g_visualizer_enabled,
                         "Turn on visualizer output (1=On, 0=Off)", "1");
  option_parser_register(opp, "-visualizer_outputfile", OPT_CSTR,
                         &g_visualizer_filename,
                         "Specifies the output log file for visualizer", NULL);
  option_parser_register(
      opp, "-visualizer_zlevel", OPT_INT32, &g_visualizer_zlevel,
      "Compression level of the visualizer output log (0=no comp, 9=highest)",
      "6");
  option_parser_register(opp, "-gpgpu_stack_size_limit", OPT_INT32,
                         &stack_size_limit, "GPU thread stack size", "1024");
  option_parser_register(opp, "-gpgpu_heap_size_limit", OPT_INT32,
                         &heap_size_limit, "GPU malloc heap size ", "8388608");
  option_parser_register(opp, "-gpgpu_runtime_sync_depth_limit", OPT_INT32,
                         &runtime_sync_depth_limit,
                         "GPU device runtime synchronize depth", "2");
  option_parser_register(opp, "-gpgpu_runtime_pending_launch_count_limit",
                         OPT_INT32, &runtime_pending_launch_count_limit,
                         "GPU device runtime pending launch count", "2048");
  option_parser_register(opp, "-trace_enabled", OPT_BOOL, &Trace::enabled,
                         "Turn on traces", "0");
  option_parser_register(opp, "-trace_components", OPT_CSTR, &Trace::config_str,
                         "comma seperated list of traces to enable. "
                         "Complete list found in trace_streams.tup. "
                         "Default none",
                         "none");
  option_parser_register(
      opp, "-trace_sampling_core", OPT_INT32, &Trace::sampling_core,
      "The core which is printed using CORE_DPRINTF. Default 0", "0");
  option_parser_register(opp,
                         "-trace_samplinm_gpu->get_global_memory()_partition",
                         OPT_INT32, &Trace::sampling_memory_partition,
                         "The memory partition which is printed using "
                         "MEMPART_DPRINTF. Default -1 (i.e. all)",
                         "-1");
  gpgpu_ctx->stats->ptx_file_line_stats_options(opp);

  // Jin: kernel launch latency
  option_parser_register(opp, "-gpgpu_kernel_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_kernel_launch_latency),
                         "Kernel launch latency in cycles. Default: 0", "0");
  option_parser_register(opp, "-gpgpu_cdp_enabled", OPT_BOOL,
                         &(gpgpu_ctx->device_runtime->g_cdp_enabled),
                         "Turn on CDP", "0");

  option_parser_register(opp, "-gpgpu_TB_launch_latency", OPT_INT32,
                         &(gpgpu_ctx->device_runtime->g_TB_launch_latency),
                         "thread block launch latency in cycles. Default: 0",
                         "0");

  option_parser_register(
      opp, "-gddr_size", OPT_CSTR, &gddr_size_string,
      "Size of GDDR in MB/GB.(GLOBAL_HEAP_START, GLOBAL_HEAP_START + "
      "gddr_size) would be used for unmanged memory, (GLOBAL_HEAP_START + "
      "gddr_size, GLOBAL_HEAP_START + gddr_size*2) would be used for managed "
      "memory. ",
      "1GB");

  option_parser_register(
      opp, "-page_table_walk_latency", OPT_INT64, &page_table_walk_latency,
      "Average page table walk latency (in core cycle).", "100");

  option_parser_register(opp, "-eviction_policy", OPT_INT32, &eviction_policy,
                         "Select page eviction policy", "0");

  option_parser_register(opp, "-invalidate_clean", OPT_BOOL, &invalidate_clean,
                         "Should directly invalidate clean pages", "0");

  option_parser_register(
      opp, "-reserve_accessed_page_percent", OPT_FLOAT,
      &reserve_accessed_page_percent,
      "Percentage of accessed pages reserved from eviction in hope that they "
      "will be accessed in next iteration.",
      "5.0");

  option_parser_register(
      opp, "-percentage_of_free_page_buffer", OPT_FLOAT,
      &free_page_buffer_percentage,
      "Percentage of free page buffer to trigger the page eviction.", "0.0");

  option_parser_register(
      opp, "-percentage_of_comp_page_buffer", OPT_FLOAT,
      &comp_buffer_percentage,
      "Percentage of free page buffer to trigger the page compression.", "0.0");
option_parser_register(
      opp, "-percentage_of_comp_evict_start", OPT_FLOAT,
      &comp_evict_start_percentage,
      "Percentage of compression size to trigger the page compression.", "30.0");
  option_parser_register(opp, "-selective_compression_threshold", OPT_FLOAT,
                         &selective_compression_threshold,
                         "Threshold for selective compression.", "1.7");
  option_parser_register(opp, "-enable_LRU_compression", OPT_BOOL,
                         &enable_LRU_compression,
                         "Enable LRU compression", "0");
  option_parser_register(opp, "-enable_LRU_decompression", OPT_BOOL,
                         &enable_LRU_decompression,
                         "Enable LRU decompression", "0");
  option_parser_register(opp, "-page_size", OPT_CSTR, &page_size_string,
                         "GDDR page size, only 4KB/2MB avaliable.", "4KB");
  option_parser_register(opp, "-comp_band_scalar", OPT_FLOAT, &comp_band_scalar,
                         "Global compression bandwidth", "4.5");
  option_parser_register(opp, "-decomp_band_scalar", OPT_FLOAT, &decomp_band_scalar,
                         "Global decompression bandwidth", "4.5");
  option_parser_register(opp, "-comp_band_string", OPT_CSTR, &comp_band_string,
                         "compression bandwidth per 2SM and split by ,", "NO");
  option_parser_register(
      opp, "-decomp_band_string", OPT_CSTR, &decomp_band_string,
      "decompression bandwidth per 2SM and split by ,", "NO");
  option_parser_register(opp, "-device_num_tpc", OPT_INT32, &device_num_tpc,
                         "number of TPCs in the device", "54");
  option_parser_register(opp, "-comp_ratio_string", OPT_CSTR,
                         &comp_ratio_string,
                         "compression ratio per 2MB and split by ,", "NO");
  option_parser_register(opp, "-pcie_bandwidth", OPT_CSTR,
                         &pcie_bandwidth_string,
                         "PCI-e bandwidth per direction, in GB/s.", "16.0GB/s");

  option_parser_register(opp, "-enable_dma", OPT_INT32, &enable_dma,
                         "Enable direct access to CPU memory", "0");

  option_parser_register(
      opp, "-multiply_dma_penalty", OPT_INT32, &multiply_dma_penalty,
      "Oversubscription Multiplicative Penalty Factor for Adaptive DMA", "0");

  option_parser_register(
      opp, "-migrate_threshold", OPT_INT32, &migrate_threshold,
      "Access counter threshold for migrating the page from cpu to gpu", "10");
  option_parser_register(opp, "-sim_util_enable", OPT_BOOL, &util_prof,
                         "Enable pcie utilzation", "0");
  option_parser_register(opp, "-sim_prof_enable", OPT_BOOL, &sim_prof_enable,
                         "Enable gpgpu-sim profiler", "0");

  option_parser_register(opp, "-hardware_prefetch", OPT_INT32,
                         &hardware_prefetch,
                         "Select gpgpu-sim hardware prefetcher", "1");

  option_parser_register(
      opp, "-hwprefetch_oversub", OPT_INT32, &hwprefetch_oversub,
      "Select gpgpu-sim hardware prefetcher under over-subscription", "0");

  option_parser_register(opp, "-page_fault_latency", OPT_INT64,
                         &page_fault_latency,
                         "Average fault latency (in core cycle).", "66645");
  option_parser_register(
      opp, "-page_comp_fault_latency", OPT_INT64, &page_comp_fault_latency,
      "Serialized CRM handling latency per 64KB transition (core cycles).", "2820");//bst edit 2us
  option_parser_register(opp, "-page_compress_latency", OPT_INT64,
                         &compress_latency,
                         "Average compress latency (in dram cycle).",
                         "11");  // bst edit 128B/per 11cycle BPC paer 45nm
  option_parser_register(opp, "-enable_accurate_simulation", OPT_BOOL,
                         &enable_accurate_simulation,
                         "Enable page fault functional simulation.", "0");
  option_parser_register(opp, "-collect_access_trace", OPT_BOOL,
                         &collect_access_trace,
                         "Enable collecting page access trace in access.txt "
                         "file",
                         "0");
  option_parser_register(opp, "-collect_memory_access_stats", OPT_BOOL,
                         &collect_memory_access_stats,
                         "Enable collecting memory access stats in memory_access_stats.txt file",
                         "0");
  option_parser_register(
      opp, "-tbn_threshold", OPT_INT32, &tbn_threshold,
      "tbn_threshold 50 default 1 2MB 100 disable must use TBN prefetch", "50");
  option_parser_register(opp, "-compression_policy", OPT_INT32,
                         &compression_policy,
                         "compression_policy 0 none 1 partial 2 full.", "0");
  option_parser_register(opp, "-change_comp_ratio", OPT_BOOL, &change_comp_ratio,
                         "change_comp_ratio 0 off 1 on.", "0");
  option_parser_register(opp, "-static_comp_ratio", OPT_FLOAT, &static_comp_ratio,
                         "static_comp_ratio 0 off and othrer is static comp ratio.", "0.0");
  option_parser_register(opp, "-fault_based_LRU", OPT_BOOL, &fault_based_LRU,
                         "fault_based_LRU 0 off 1 on.", "0");
  option_parser_register(opp, "-forest_LRU", OPT_BOOL, &forest_LRU,
                         "forest_LRU 0 off 1 on.", "0");
  option_parser_register(opp, "-tree_type_string", OPT_CSTR, &tree_type_string,
                         "tree_type 0 default 1 LS 2 HCHI 3 HCLI 4 range", "NO");
  option_parser_register(opp, "-tree_type_kernel_num_string", OPT_CSTR, &tree_type_kernel_num_string,
                         "repeated kernel num", "NO");
  option_parser_register(opp, "-tree_logic", OPT_INT32, &tree_logic,
                         "tree logic 0 default 1 2 3 4", "0");//0 is TBN(CURE-MEM) 1 is TBN+2MB eviction 2 is Range 3 is forest 4 is SAFE
  option_parser_register(opp, "-MICRO_barrier_Range", OPT_BOOL, &MICRO_barrier_Range,
                         "with tree_logic==2: comp/decomp at 64KB, eviction at 2MB. 0 off 1 on.", "0");
  option_parser_register(opp, "-real_WB_delay", OPT_INT32, &real_WB_delay,
                         "real_WB_delay 0 is optimistic other is delay cycle", "0");//0 is no other is delay cycle
  option_parser_register(opp, "-proactive_WB", OPT_INT32, &proactive_WB,
                         "proactive_WB 0 off 1 UPCYCLE 2 is proactive.", "0");//originally check reserved pages but if this is on, race condition occured in pcie transfer so removed.
  option_parser_register(opp, "-ignore_reserved_pages", OPT_BOOL, &ignore_reserved_pages,
                         "ignore_reserved_pages 0 off 1 on.", "1");//originally check reserved pages but if this is on, race condition occured in pcie transfer so removed.
  option_parser_register(opp, "-selective_compression", OPT_BOOL,
                         &selective_compression,
                         "selective_compression with only compress effective "
                         "pages 0 none 1 selective.",
                         "0");
  option_parser_register(opp, "-evict_form", OPT_BOOL, &evict_form,
                         "evict form to host 0 raw 1 compressed .", "0");
  option_parser_register(opp, "-full_comp_policy", OPT_BOOL, &full_comp_policy,
                         "full_comp 0 none 1 full.",
                         "0");  // deprecated use compression_engine instead
  option_parser_register(opp, "-comp_evict_policy", OPT_BOOL,
                         &comp_evict_policy, "comp_evict 0 off 1 on.", "0");
  option_parser_register(opp, "-comp_ratio", OPT_FLOAT, &comp_ratio,
                         "average compression ratio.", "2.0");
  option_parser_register(opp, "-enable_smart_runtime", OPT_BOOL,
                         &enable_smart_runtime,
                         "Enable access pattern detection, policy engine, and "
                         "adaptive memory management.",
                         "0");
  option_parser_register(
      opp, "-skip_cycles_enable", OPT_BOOL, &skip_cycles_enable,
      "Enable skip cycles if all warps stall and wait for page fualt", "0");
  option_parser_register(
      opp, "-compression_engine", OPT_INT32, &compression_engine,
      "Select compression engine (0: none, 1: LCP_BPC, 2: LCP_BDI)", "0");
  option_parser_register(opp, "-two_level_LRU", OPT_BOOL, &two_level_LRU,
                         "two_level_LRU 0 off 1 on.", "0");
}

void gpgpu_sim_config::convert_byte_string() {
  if (!compression_policy) {
    comp_ratio_string = const_cast<char*>("NO");
  }
  if(!tree_type_string) {
    tree_type_string = const_cast<char*>("NO");
  }
  if(!tree_type_kernel_num_string) {
    tree_type_kernel_num_string = const_cast<char*>("NO");
  }
  
  gpgpu_functional_sim_config::convert_byte_string();
  if (strstr(pcie_bandwidth_string, "GB/s")) {
    pcie_bandwidth = strtof(pcie_bandwidth_string, NULL);
    if (pcie_bandwidth == 16.0) {
      curve_a = 12.0;
    } else if (pcie_bandwidth == 32.0) {
      curve_a = 24.0;
    } else if (pcie_bandwidth == 64.0) {
      curve_a = 48.0;
    } else {
      curve_a = pcie_bandwidth * 3 / 4;
    }

    curve_b = 0.07292;

  } else {
    printf("-pcie_bandwidth should be in GB/s\n");
    exit(1);
  }
}
void gpgpu_sim_config::calc_comp_lat() {
  if (compression_policy || full_comp_policy) {
    compress_latency = round(core_freq / dram_freq * compress_latency * 4);
    printf("bst compression latency %llu\n", compress_latency);
  } else {
    compress_latency = 0;
  }
}
/////////////////////////////////////////////////////////////////////////////

void increment_x_then_y_then_z(dim3 &i, const dim3 &bound) {
  i.x++;
  if (i.x >= bound.x) {
    i.x = 0;
    i.y++;
    if (i.y >= bound.y) {
      i.y = 0;
      if (i.z < bound.z) i.z++;
    }
  }
}

void gpgpu_sim::launch(kernel_info_t *kinfo) {
  unsigned cta_size = kinfo->threads_per_cta();
  if (cta_size > m_shader_config->n_thread_per_shader) {
    printf(
        "Execution error: Shader kernel CTA (block) size is too large for "
        "microarch config.\n");
    printf("                 CTA size (x*y*z) = %u, max supported = %u\n",
           cta_size, m_shader_config->n_thread_per_shader);
    printf(
        "                 => either change -gpgpu_shader argument in "
        "gpgpusim.config file or\n");
    printf(
        "                 modify the CUDA source to decrease the kernel block "
        "size.\n");
    abort();
  }
  unsigned n = 0;
  for (n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done()) {
      m_running_kernels[n] = kinfo;
      break;
    }
  }
  assert(n < m_running_kernels.size());
}

bool gpgpu_sim::can_start_kernel() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if ((NULL == m_running_kernels[n]) || m_running_kernels[n]->done())
      return true;
  }
  return false;
}

bool gpgpu_sim::hit_max_cta_count() const {
  if (m_config.gpu_max_cta_opt != 0) {
    if ((gpu_tot_issued_cta + m_total_cta_launched) >= m_config.gpu_max_cta_opt)
      return true;
  }
  return false;
}

bool gpgpu_sim::kernel_more_cta_left(kernel_info_t *kernel) const {
  if (hit_max_cta_count()) return false;

  if (kernel && !kernel->no_more_ctas_to_run()) return true;

  return false;
}

bool gpgpu_sim::get_more_cta_left() const {
  if (hit_max_cta_count()) return false;

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && !m_running_kernels[n]->no_more_ctas_to_run())
      return true;
  }
  return false;
}

void gpgpu_sim::decrement_kernel_latency() {
  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    if (m_running_kernels[n] && m_running_kernels[n]->m_kernel_TB_latency)
      m_running_kernels[n]->m_kernel_TB_latency--;
  }
}

kernel_info_t *gpgpu_sim::select_kernel() {
  if (m_running_kernels[m_last_issued_kernel] &&
      !m_running_kernels[m_last_issued_kernel]->no_more_ctas_to_run() &&
      !m_running_kernels[m_last_issued_kernel]->m_kernel_TB_latency) {
    unsigned launch_uid = m_running_kernels[m_last_issued_kernel]->get_uid();
    if (std::find(m_executed_kernel_uids.begin(), m_executed_kernel_uids.end(),
                  launch_uid) == m_executed_kernel_uids.end()) {
      m_running_kernels[m_last_issued_kernel]->start_cycle =
          gpu_sim_cycle + gpu_tot_sim_cycle;
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(
          m_running_kernels[m_last_issued_kernel]->name());
    }
    return m_running_kernels[m_last_issued_kernel];
  }

  for (unsigned n = 0; n < m_running_kernels.size(); n++) {
    unsigned idx =
        (n + m_last_issued_kernel + 1) % m_config.max_concurrent_kernel;
    if (kernel_more_cta_left(m_running_kernels[idx]) &&
        !m_running_kernels[idx]->m_kernel_TB_latency) {
      m_last_issued_kernel = idx;
      m_running_kernels[idx]->start_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      // record this kernel for stat print if it is the first time this kernel
      // is selected for execution
      unsigned launch_uid = m_running_kernels[idx]->get_uid();
      assert(std::find(m_executed_kernel_uids.begin(),
                       m_executed_kernel_uids.end(),
                       launch_uid) == m_executed_kernel_uids.end());
      m_executed_kernel_uids.push_back(launch_uid);
      m_executed_kernel_names.push_back(m_running_kernels[idx]->name());

      return m_running_kernels[idx];
    }
  }
  return NULL;
}

unsigned gpgpu_sim::finished_kernel() {
  if (m_finished_kernel.empty()) return 0;
  unsigned result = m_finished_kernel.front();
  m_finished_kernel.pop_front();
  return result;
}

void gpgpu_sim::set_kernel_done(kernel_info_t *kernel) {
  unsigned uid = kernel->get_uid();
  m_finished_kernel.push_back(uid);
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); k++) {
    if (*k == kernel) {
      kernel->end_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
      *k = NULL;
      break;
    }
  }
  assert(k != m_running_kernels.end());
}

void gpgpu_sim::stop_all_running_kernels() {
  std::vector<kernel_info_t *>::iterator k;
  for (k = m_running_kernels.begin(); k != m_running_kernels.end(); ++k) {
    if (*k != NULL) {       // If a kernel is active
      set_kernel_done(*k);  // Stop the kernel
      assert(*k == NULL);
    }
  }
}

void exec_gpgpu_sim::createSIMTCluster() {
  m_cluster = new simt_core_cluster *[m_shader_config->n_simt_clusters];
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i] =
        new exec_simt_core_cluster(this, i, m_shader_config, m_memory_config,
                                   m_shader_stats, m_memory_stats, m_new_stats);
}

gpgpu_sim::gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
    : gpgpu_t(config, ctx), m_config(config) {
  gpgpu_ctx = ctx;
  m_shader_config = &m_config.m_shader_config;
  m_memory_config = &m_config.m_memory_config;
  ctx->ptx_parser->set_ptx_warp_size(m_shader_config);
  ptx_file_line_stats_create_exposed_latency_tracker(m_config.num_shader());

#ifdef GPGPUSIM_POWER_MODEL
  m_gpgpusim_wrapper = new gpgpu_sim_wrapper(
      config.g_power_simulation_enabled, config.g_power_config_name,
      config.g_power_simulation_mode, config.g_dvfs_enabled);
#endif

  m_shader_stats = new shader_core_stats(m_shader_config);
  m_memory_stats = new memory_stats_t(m_config.num_shader(), m_shader_config,
                                      m_memory_config, this);
  average_pipeline_duty_cycle = (float *)malloc(sizeof(float));
  active_sms = (float *)malloc(sizeof(float));
  m_power_stats =
      new power_stat_t(m_shader_config, average_pipeline_duty_cycle, active_sms,
                       m_shader_stats, m_memory_config, m_memory_stats);

  m_new_stats = new gpgpu_new_stats(m_config);

  // Initialize public member variables
  gpu_sim_insn = 0;
  gpu_tot_sim_insn = 0;
  gpu_sim_insn_last_update = 0;
  gpu_sim_insn_last_update_sid = 0;
  
  // Initialize occupancy stats
  gpu_occupancy = occupancy_stats();
  gpu_tot_occupancy = occupancy_stats();
  
  gpu_tot_issued_cta = 0;
  gpu_completed_cta = 0;
  if (m_shader_config->ETC_throttle_SM) {
    num_throttled_SM = m_shader_config->n_simt_clusters - 2;
  } else if (m_shader_config->CURE_throttle_SM) {
    num_throttled_SM = m_shader_config->n_simt_clusters;
  } else {
    num_throttled_SM = m_shader_config->n_simt_clusters;
  }
  
  // Initialize CURE throttle vectors
  prev_active_warps_per_cluster.resize(m_shader_config->n_simt_clusters, 0);
  prev_total_warps_per_cluster.resize(m_shader_config->n_simt_clusters, 0);
  
  // throttle 조절 마지막 실행 시점 초기화
  last_throttle_adjustment_cycle = 0;
  print_throttle_count = 0;
  m_total_cta_launched = 0;
  
  // Initialize protected member variables
  m_last_issued_kernel = 0;
  last_gpu_sim_insn = 0;
  gpu_deadlock = false;
  printf("bst debug gpgpu_sim compression_engine = %d\n", m_config.compression_engine);
  m_gmmu = new gmmu_t(this, config, m_new_stats);

  gpu_stall_dramfull = 0;
  gpu_stall_icnt2sh = 0;
  partiton_reqs_in_parallel = 0;
  partiton_reqs_in_parallel_total = 0;
  partiton_reqs_in_parallel_util = 0;
  partiton_reqs_in_parallel_util_total = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_tot_sim_cycle_parition_util = 0;
  partiton_replys_in_parallel = 0;
  partiton_replys_in_parallel_total = 0;

  m_memory_partition_unit =
      new memory_partition_unit *[m_memory_config->m_n_mem];
  m_memory_sub_partition =
      new memory_sub_partition *[m_memory_config->m_n_mem_sub_partition];
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
    m_memory_partition_unit[i] =
        new memory_partition_unit(i, m_memory_config, m_memory_stats, this);
    for (unsigned p = 0;
         p < m_memory_config->m_n_sub_partition_per_memory_channel; p++) {
      unsigned submpid =
          i * m_memory_config->m_n_sub_partition_per_memory_channel + p;
      m_memory_sub_partition[submpid] =
          m_memory_partition_unit[i]->get_sub_partition(p);
    }
  }

  icnt_wrapper_init();
  icnt_create(m_shader_config->n_simt_clusters,
              m_memory_config->m_n_mem_sub_partition);

  time_vector_create(NUM_MEM_REQ_STAT);
  fprintf(stdout,
          "GPGPU-Sim uArch: performance model initialization complete.\n");

  m_running_kernels.resize(config.max_concurrent_kernel, NULL);
  m_last_issued_kernel = 0;
  m_last_cluster_issue = m_shader_config->n_simt_clusters -
                         1;  // this causes first launch to use simt cluster 0
  m_last_fetched_cluster = 0;
  *average_pipeline_duty_cycle = 0;
  *active_sms = 0;

  last_liveness_message_time = 0;

  // Initialize pointer members
  m_cuda_properties = NULL;
  
  // Jin: functional simulation for CDP
  m_functional_sim = false;
  m_functional_sim_kernel = NULL;
}

int gpgpu_sim::shared_mem_size() const {
  return m_shader_config->gpgpu_shmem_size;
}

int gpgpu_sim::shared_mem_per_block() const {
  return m_shader_config->gpgpu_shmem_per_block;
}

int gpgpu_sim::num_registers_per_core() const {
  return m_shader_config->gpgpu_shader_registers;
}

int gpgpu_sim::num_registers_per_block() const {
  return m_shader_config->gpgpu_registers_per_block;
}

int gpgpu_sim::wrp_size() const { return m_shader_config->warp_size; }

int gpgpu_sim::shader_clock() const { return m_config.core_freq / 1000; }

int gpgpu_sim::max_cta_per_core() const {
  return m_shader_config->max_cta_per_core;
}

int gpgpu_sim::get_max_cta(const kernel_info_t &k) const {
  return m_shader_config->max_cta(k);
}

void gpgpu_sim::set_prop(cudaDeviceProp *prop) { m_cuda_properties = prop; }

int gpgpu_sim::compute_capability_major() const {
  return m_config.gpgpu_compute_capability_major;
}

int gpgpu_sim::compute_capability_minor() const {
  return m_config.gpgpu_compute_capability_minor;
}

const struct cudaDeviceProp *gpgpu_sim::get_prop() const {
  return m_cuda_properties;
}

enum divergence_support_t gpgpu_sim::simd_model() const {
  return m_shader_config->model;
}

void gpgpu_sim_config::init_clock_domains(void) {
  sscanf(gpgpu_clock_domains, "%lf:%lf:%lf:%lf", &core_freq, &icnt_freq,
         &l2_freq, &dram_freq);
  core_freq = core_freq MhZ;
  icnt_freq = icnt_freq MhZ;
  l2_freq = l2_freq MhZ;
  dram_freq = dram_freq MhZ;
  core_period = 1 / core_freq;
  icnt_period = 1 / icnt_freq;
  dram_period = 1 / dram_freq;
  l2_period = 1 / l2_freq;
  printf("GPGPU-Sim uArch: clock freqs: %lf:%lf:%lf:%lf\n", core_freq,
         icnt_freq, l2_freq, dram_freq);
  printf("GPGPU-Sim uArch: clock periods: %.20lf:%.20lf:%.20lf:%.20lf\n",
         core_period, icnt_period, l2_period, dram_period);
}

void gpgpu_sim::reinit_clock_domains(void) {
  core_time = 0;
  dram_time = 0;
  icnt_time = 0;
  l2_time = 0;
  gmmu_time = 0;
}

bool gpgpu_sim::active() {
  if (m_config.gpu_max_cycle_opt &&
      (gpu_tot_sim_cycle + gpu_sim_cycle) >= m_config.gpu_max_cycle_opt)
    return false;
  if (m_config.gpu_max_insn_opt &&
      (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt)
    return false;
  if (m_config.gpu_max_cta_opt &&
      (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt))
    return false;
  if (m_config.gpu_max_completed_cta_opt &&
      (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt))
    return false;
  if (m_config.gpu_deadlock_detect && gpu_deadlock) return false;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    if (m_cluster[i]->get_not_completed() > 0) return true;
  ;
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    if (m_memory_partition_unit[i]->busy() > 0) return true;
  ;
  if (icnt_busy()) return true;
  if (get_more_cta_left()) return true;
  return false;
}

void gpgpu_sim::init() {
  // run a CUDA grid on the GPU microarchitecture simulator
  gpu_sim_cycle = 0;
  skipped_cycles = 0;
  gpu_sim_insn = 0;
  last_gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;

// McPAT initialization function. Called on first launch of GPU
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    init_mcpat(m_config, m_gpgpusim_wrapper, m_config.gpu_stat_sample_freq,
               gpu_tot_sim_insn, gpu_sim_insn);
  }
#endif

  reinit_clock_domains();
  gpgpu_ctx->func_sim->set_param_gpgpu_num_shaders(m_config.num_shader());
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
    m_cluster[i]->reinit();
  m_shader_stats->new_grid();
  // initialize the control-flow, memory access, memory latency logger
  if (m_config.g_visualizer_enabled) {
    create_thread_CFlogger(gpgpu_ctx, m_config.num_shader(),
                           m_shader_config->n_thread_per_shader, 0,
                           m_config.gpgpu_cflog_interval);
  }
  shader_CTA_count_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
  if (m_config.gpgpu_cflog_interval != 0) {
    insn_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size);
    shader_warp_occ_create(m_config.num_shader(), m_shader_config->warp_size,
                           m_config.gpgpu_cflog_interval);
    shader_mem_acc_create(m_config.num_shader(), m_memory_config->m_n_mem, 4,
                          m_config.gpgpu_cflog_interval);
    shader_mem_lat_create(m_config.num_shader(), m_config.gpgpu_cflog_interval);
    shader_cache_access_create(m_config.num_shader(), 3,
                               m_config.gpgpu_cflog_interval);
    set_spill_interval(m_config.gpgpu_cflog_interval * 40);
  }

  if (g_network_mode) icnt_init();
}

void gpgpu_sim::update_stats() {
  m_memory_stats->memlatstat_lat_pw();
  gpu_tot_sim_cycle += gpu_sim_cycle;
  gpu_tot_sim_insn += gpu_sim_insn;
  gpu_tot_skipped_cycle += skipped_cycles;
  gpu_tot_issued_cta += m_total_cta_launched;
  partiton_reqs_in_parallel_total += partiton_reqs_in_parallel;
  partiton_replys_in_parallel_total += partiton_replys_in_parallel;
  partiton_reqs_in_parallel_util_total += partiton_reqs_in_parallel_util;
  gpu_tot_sim_cycle_parition_util += gpu_sim_cycle_parition_util;
  gpu_tot_occupancy += gpu_occupancy;

  gpu_sim_cycle = 0;
  skipped_cycles = 0;
  partiton_reqs_in_parallel = 0;
  partiton_replys_in_parallel = 0;
  partiton_reqs_in_parallel_util = 0;
  gpu_sim_cycle_parition_util = 0;
  gpu_sim_insn = 0;
  m_total_cta_launched = 0;
  gpu_completed_cta = 0;
  gpu_occupancy = occupancy_stats();
}

PowerscalingCoefficients *gpgpu_sim::get_scaling_coeffs() {
  return m_gpgpusim_wrapper->get_scaling_coeffs();
}

void gpgpu_sim::print_stats() {
  gpgpu_ctx->stats->ptx_file_line_stats_write_file();
  gpu_print_stat();

  if (g_network_mode) {
    printf(
        "----------------------------Interconnect-DETAILS----------------------"
        "----------\n");
    icnt_display_stats();
    icnt_display_overall_stats();
    printf(
        "----------------------------END-of-Interconnect-DETAILS---------------"
        "----------\n");
  }
}
bool gpgpu_sim::is_progress_print_set() { return m_config.progress_print; }
void gpgpu_sim::deadlock_check() {
  if (m_config.gpu_deadlock_detect && gpu_deadlock) {
    fflush(stdout);
    printf(
        "\n\nGPGPU-Sim uArch: ERROR ** deadlock detected: last writeback core "
        "%u @ gpu_sim_cycle %u (+ gpu_tot_sim_cycle %u) (%u cycles ago)\n",
        gpu_sim_insn_last_update_sid, (unsigned)gpu_sim_insn_last_update,
        (unsigned)(gpu_tot_sim_cycle - gpu_sim_cycle),
        (unsigned)(gpu_sim_cycle - gpu_sim_insn_last_update));
    unsigned num_cores = 0;
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned not_completed = m_cluster[i]->get_not_completed();
      if (not_completed) {
        if (!num_cores) {
          printf(
              "GPGPU-Sim uArch: DEADLOCK  shader cores no longer committing "
              "instructions [core(# threads)]:\n");
          printf("GPGPU-Sim uArch: DEADLOCK  ");
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores < 8) {
          m_cluster[i]->print_not_completed(stdout);
        } else if (num_cores >= 8) {
          printf(" + others ... ");
        }
        num_cores += m_shader_config->n_simt_cores_per_cluster;
      }
    }
    printf("\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      bool busy = m_memory_partition_unit[i]->busy();
      if (busy)
        printf("GPGPU-Sim uArch DEADLOCK:  memory partition %u busy\n", i);
    }
    if (icnt_busy()) {
      printf("GPGPU-Sim uArch DEADLOCK:  iterconnect contains traffic\n");
      icnt_display_state(stdout);
    }
    printf(
        "\nRe-run the simulator in gdb and use debug routines in .gdbinit to "
        "debug this\n");
    fflush(stdout);
    abort();
  }
}

/// printing the names and uids of a set of executed kernels (usually there is
/// only one)
std::string gpgpu_sim::executed_kernel_info_string() {
  std::stringstream statout;

  statout << "kernel_name = ";
  for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
    statout << m_executed_kernel_names[k] << " ";
  }
  statout << std::endl;
  statout << "kernel_launch_uid = ";
  for (unsigned int k = 0; k < m_executed_kernel_uids.size(); k++) {
    statout << m_executed_kernel_uids[k] << " ";
  }
  statout << std::endl;

  return statout.str();
}

std::string gpgpu_sim::executed_kernel_name() {
  std::stringstream statout;
  if (m_executed_kernel_names.size() == 1)
    statout << m_executed_kernel_names[0];
  else {
    for (unsigned int k = 0; k < m_executed_kernel_names.size(); k++) {
      statout << m_executed_kernel_names[k] << " ";
    }
  }
  return statout.str();
}
void gpgpu_sim::set_cache_config(std::string kernel_name,
                                 FuncCache cacheConfig) {
  m_special_cache_config[kernel_name] = cacheConfig;
}

FuncCache gpgpu_sim::get_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return iter->second;
    }
  }
  return (FuncCache)0;
}

bool gpgpu_sim::has_special_cache_config(std::string kernel_name) {
  for (std::map<std::string, FuncCache>::iterator iter =
           m_special_cache_config.begin();
       iter != m_special_cache_config.end(); iter++) {
    std::string kernel = iter->first;
    if (kernel_name.compare(kernel) == 0) {
      return true;
    }
  }
  return false;
}

void gpgpu_sim::set_cache_config(std::string kernel_name) {
  if (has_special_cache_config(kernel_name)) {
    change_cache_config(get_cache_config(kernel_name));
  } else {
    change_cache_config(FuncCachePreferNone);
  }
}

void gpgpu_sim::change_cache_config(FuncCache cache_config) {
  if (cache_config != m_shader_config->m_L1D_config.get_cache_status()) {
    printf("FLUSH L1 Cache at configuration change between kernels\n");
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      m_cluster[i]->cache_invalidate();
    }
  }

  switch (cache_config) {
    case FuncCachePreferNone:
      m_shader_config->m_L1D_config.init(
          m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
      m_shader_config->gpgpu_shmem_size =
          m_shader_config->gpgpu_shmem_sizeDefault;
      break;
    case FuncCachePreferL1:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefL1 == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefL1 == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;

      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefL1,
            FuncCachePreferL1);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefL1;
      }
      break;
    case FuncCachePreferShared:
      if ((m_shader_config->m_L1D_config.m_config_stringPrefShared == NULL) ||
          (m_shader_config->gpgpu_shmem_sizePrefShared == (unsigned)-1)) {
        printf("WARNING: missing Preferred L1 configuration\n");
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_string, FuncCachePreferNone);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizeDefault;
      } else {
        m_shader_config->m_L1D_config.init(
            m_shader_config->m_L1D_config.m_config_stringPrefShared,
            FuncCachePreferShared);
        m_shader_config->gpgpu_shmem_size =
            m_shader_config->gpgpu_shmem_sizePrefShared;
      }
      break;
    default:
      break;
  }
}

void gpgpu_sim::clear_executed_kernel_info() {
  m_executed_kernel_names.clear();
  m_executed_kernel_uids.clear();
}
void gpgpu_sim::gpu_print_stat() {
  FILE *statfout = stdout;

  std::string kernel_info_str = executed_kernel_info_string();
  fprintf(statfout, "%s", kernel_info_str.c_str());

  printf("gpu_sim_cycle = %lld\n", gpu_sim_cycle);
  printf("gpu_sim_insn = %lld\n", gpu_sim_insn);
  printf("gpu_ipc = %12.4f\n", (float)gpu_sim_insn / gpu_sim_cycle);
  printf("gpu_tot_sim_cycle = %lld\n", gpu_tot_sim_cycle + gpu_sim_cycle);
  printf("gpu_tot_sim_insn = %lld\n", gpu_tot_sim_insn + gpu_sim_insn);
  printf("gpu_tot_ipc = %12.4f\n", (float)(gpu_tot_sim_insn + gpu_sim_insn) /
                                       (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("gpu_tot_issued_cta = %lld\n",
         gpu_tot_issued_cta + m_total_cta_launched);
  printf("gpu_occupancy = %.4f%% \n", gpu_occupancy.get_occ_fraction() * 100);
  printf("gpu_tot_occupancy = %.4f%% \n",
         (gpu_occupancy + gpu_tot_occupancy).get_occ_fraction() * 100);

  fprintf(statfout, "max_total_param_size = %llu\n",
          gpgpu_ctx->device_runtime->g_max_total_param_size);

  // performance counter for stalls due to congestion.
  printf("gpu_stall_dramfull = %d\n", gpu_stall_dramfull);
  printf("gpu_stall_icnt2sh    = %d\n", gpu_stall_icnt2sh);

  printf("partiton_level_parallism = %12.4f\n",
         (float)partiton_reqs_in_parallel / gpu_sim_cycle);
  printf("partiton_level_parallism_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel + partiton_reqs_in_parallel_total) /
             (gpu_tot_sim_cycle + gpu_sim_cycle));
  printf("partiton_level_parallism_util = %12.4f\n",
         (float)partiton_reqs_in_parallel_util / gpu_sim_cycle_parition_util);
  printf("partiton_level_parallism_util_total  = %12.4f\n",
         (float)(partiton_reqs_in_parallel_util +
                 partiton_reqs_in_parallel_util_total) /
             (gpu_sim_cycle_parition_util + gpu_tot_sim_cycle_parition_util));
  printf("L2_BW  = %12.4f GB/Sec\n",
         ((float)(partiton_replys_in_parallel * 32) /
          (gpu_sim_cycle * m_config.icnt_period)) /
             1000000000);
  printf("L2_BW_total  = %12.4f GB/Sec\n",
         ((float)((partiton_replys_in_parallel +
                   partiton_replys_in_parallel_total) *
                  32) /
          ((gpu_tot_sim_cycle + gpu_sim_cycle) * m_config.icnt_period)) /
             1000000000);

  time_t curr_time;
  time(&curr_time);
  unsigned long long elapsed_time =
      MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
  printf("gpu_total_sim_rate=%u\n",
         (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time));

  // shader_print_l1_miss_stat( stdout );
  shader_print_cache_stats(stdout);

  cache_stats core_cache_stats;
  core_cache_stats.clear();
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_cache_stats(core_cache_stats);
  }
  printf("\nTotal_core_cache_stats:\n");
  core_cache_stats.print_stats(stdout, "Total_core_cache_stats_breakdown");
  printf("\nTotal_core_cache_fail_stats:\n");
  core_cache_stats.print_fail_stats(stdout,
                                    "Total_core_cache_fail_stats_breakdown");
  shader_print_scheduler_stat(stdout, false);

  m_shader_stats->print(stdout);
#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    if (m_config.g_power_simulation_mode > 0) {
      // if(!m_config.g_aggregate_power_stats)
      mcpat_reset_perf_count(m_gpgpusim_wrapper);
      calculate_hw_mcpat(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                         m_power_stats, m_config.gpu_stat_sample_freq,
                         gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                         gpu_sim_insn, m_config.g_power_simulation_mode,
                         m_config.g_dvfs_enabled, m_config.g_hw_perf_file_name,
                         m_config.g_hw_perf_bench_name, executed_kernel_name(),
                         m_config.accelwattch_hybrid_configuration,
                         m_config.g_aggregate_power_stats);
    }
    m_gpgpusim_wrapper->print_power_kernel_stats(
        gpu_sim_cycle, gpu_tot_sim_cycle, gpu_tot_sim_insn + gpu_sim_insn,
        kernel_info_str, true);
    // if(!m_config.g_aggregate_power_stats)
    mcpat_reset_perf_count(m_gpgpusim_wrapper);
  }
#endif

  // performance counter that are not local to one shader
  m_memory_stats->memlatstat_print(m_memory_config->m_n_mem,
                                   m_memory_config->nbk);
  for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
    m_memory_partition_unit[i]->print(stdout);

  // L2 cache stats
  if (!m_memory_config->m_L2_config.disabled()) {
    cache_stats l2_stats;
    struct cache_sub_stats l2_css;
    struct cache_sub_stats total_l2_css;
    l2_stats.clear();
    l2_css.clear();
    total_l2_css.clear();

    printf("\n========= L2 cache stats =========\n");
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      m_memory_sub_partition[i]->accumulate_L2cache_stats(l2_stats);
      m_memory_sub_partition[i]->get_L2cache_sub_stats(l2_css);

      fprintf(stdout,
              "L2_cache_bank[%d]: Access = %llu, Miss = %llu, Miss_rate = "
              "%.3lf, Pending_hits = %llu, Reservation_fails = %llu\n",
              i, l2_css.accesses, l2_css.misses,
              (double)l2_css.misses / (double)l2_css.accesses,
              l2_css.pending_hits, l2_css.res_fails);

      total_l2_css += l2_css;
    }
    if (!m_memory_config->m_L2_config.disabled() &&
        m_memory_config->m_L2_config.get_num_lines()) {
      // L2c_print_cache_stat();
      printf("L2_total_cache_accesses = %llu\n", total_l2_css.accesses);
      printf("L2_total_cache_misses = %llu\n", total_l2_css.misses);
      if (total_l2_css.accesses > 0)
        printf("L2_total_cache_miss_rate = %.4lf\n",
               (double)total_l2_css.misses / (double)total_l2_css.accesses);
      printf("L2_total_cache_pending_hits = %llu\n", total_l2_css.pending_hits);
      printf("L2_total_cache_reservation_fails = %llu\n",
             total_l2_css.res_fails);
      printf("L2_total_cache_breakdown:\n");
      l2_stats.print_stats(stdout, "L2_cache_stats_breakdown");
      printf("L2_total_cache_reservation_fail_breakdown:\n");
      l2_stats.print_fail_stats(stdout, "L2_cache_stats_fail_breakdown");
      total_l2_css.print_port_stats(stdout, "L2_cache");
    }
  }

  if (m_config.gpgpu_cflog_interval != 0) {
    spill_log_to_file(stdout, 1, gpu_sim_cycle);
    insn_warp_occ_print(stdout);
  }
  if (gpgpu_ctx->func_sim->gpgpu_ptx_instruction_classification) {
    StatDisp(gpgpu_ctx->func_sim->g_inst_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
    StatDisp(gpgpu_ctx->func_sim->g_inst_op_classification_stat
                 [gpgpu_ctx->func_sim->g_ptx_kernel_count]);
  }

#ifdef GPGPUSIM_POWER_MODEL
  if (m_config.g_power_simulation_enabled) {
    m_gpgpusim_wrapper->detect_print_steady_state(
        1, gpu_tot_sim_insn + gpu_sim_insn);
  }
#endif

  // Interconnect power stat print
  long total_simt_to_mem = 0;
  long total_mem_to_simt = 0;
  long temp_stm = 0;
  long temp_mts = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    m_cluster[i]->get_icnt_stats(temp_stm, temp_mts);
    total_simt_to_mem += temp_stm;
    total_mem_to_simt += temp_mts;
  }
  printf("\nicnt_total_pkts_mem_to_simt=%ld\n", total_mem_to_simt);
  printf("icnt_total_pkts_simt_to_mem=%ld\n", total_simt_to_mem);

  time_vector_print();
  print_UVM_stats(m_new_stats, this, stdout);
  fflush(stdout);

  clear_executed_kernel_info();
}

// performance counter that are not local to one shader
unsigned gpgpu_sim::threads_per_core() const {
  return m_shader_config->n_thread_per_shader;
}

void shader_core_ctx::mem_instruction_stats(const warp_inst_t &inst) {
  unsigned active_count = inst.active_count();
  // this breaks some encapsulation: the is_[space] functions, if you change
  // those, change this.
  switch (inst.space.get_type()) {
    case undefined_space:
    case reg_space:
      break;
    case shared_space:
      m_stats->gpgpu_n_shmem_insn += active_count;
      break;
    case sstarr_space:
      m_stats->gpgpu_n_sstarr_insn += active_count;
      break;
    case const_space:
      m_stats->gpgpu_n_const_insn += active_count;
      break;
    case param_space_kernel:
    case param_space_local:
      m_stats->gpgpu_n_param_insn += active_count;
      break;
    case tex_space:
      m_stats->gpgpu_n_tex_insn += active_count;
      break;
    case global_space:
    case local_space:
      if (inst.is_store())
        m_stats->gpgpu_n_store_insn += active_count;
      else
        m_stats->gpgpu_n_load_insn += active_count;
      break;
    default:
      abort();
  }
}
bool shader_core_ctx::can_issue_1block(kernel_info_t &kernel) {
  // Jin: concurrent kernels on one SM
  if (m_config->gpgpu_concurrent_kernel_sm) {
    if (m_config->max_cta(kernel) < 1) return false;

    return occupy_shader_resource_1block(kernel, false);
  } else {
    return (get_n_active_cta() < m_config->max_cta(kernel));
  }
}

int shader_core_ctx::find_available_hwtid(unsigned int cta_size, bool occupy) {
  unsigned int step;
  for (step = 0; step < m_config->n_thread_per_shader; step += cta_size) {
    unsigned int hw_tid;
    for (hw_tid = step; hw_tid < step + cta_size; hw_tid++) {
      if (m_occupied_hwtid.test(hw_tid)) break;
    }
    if (hw_tid == step + cta_size)  // consecutive non-active
      break;
  }
  if (step >= m_config->n_thread_per_shader)  // didn't find
    return -1;
  else {
    if (occupy) {
      for (unsigned hw_tid = step; hw_tid < step + cta_size; hw_tid++)
        m_occupied_hwtid.set(hw_tid);
    }
    return step;
  }
}

bool shader_core_ctx::occupy_shader_resource_1block(kernel_info_t &k,
                                                    bool occupy) {
  unsigned threads_per_cta = k.threads_per_cta();
  const class function_info *kernel = k.entry();
  unsigned int padded_cta_size = threads_per_cta;
  unsigned int warp_size = m_config->warp_size;
  if (padded_cta_size % warp_size)
    padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

  if (m_occupied_n_threads + padded_cta_size > m_config->n_thread_per_shader)
    return false;

  if (find_available_hwtid(padded_cta_size, false) == -1) return false;

  const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

  if (m_occupied_shmem + kernel_info->smem > m_config->gpgpu_shmem_size)
    return false;

  unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
  if (m_occupied_regs + used_regs > m_config->gpgpu_shader_registers)
    return false;

  if (m_occupied_ctas + 1 > m_config->max_cta_per_core) return false;

  if (occupy) {
    m_occupied_n_threads += padded_cta_size;
    m_occupied_shmem += kernel_info->smem;
    m_occupied_regs += (padded_cta_size * ((kernel_info->regs + 3) & ~3));
    m_occupied_ctas++;

    SHADER_DPRINTF(LIVENESS,
                   "GPGPU-Sim uArch: Occupied %u threads, %u shared mem, %u "
                   "registers, %u ctas, on shader %d\n",
                   m_occupied_n_threads, m_occupied_shmem, m_occupied_regs,
                   m_occupied_ctas, m_sid);
  }

  return true;
}

void shader_core_ctx::release_shader_resource_1block(unsigned hw_ctaid,
                                                     kernel_info_t &k) {
  if (m_config->gpgpu_concurrent_kernel_sm) {
    unsigned threads_per_cta = k.threads_per_cta();
    const class function_info *kernel = k.entry();
    unsigned int padded_cta_size = threads_per_cta;
    unsigned int warp_size = m_config->warp_size;
    if (padded_cta_size % warp_size)
      padded_cta_size = ((padded_cta_size / warp_size) + 1) * (warp_size);

    assert(m_occupied_n_threads >= padded_cta_size);
    m_occupied_n_threads -= padded_cta_size;

    int start_thread = m_occupied_cta_to_hwtid[hw_ctaid];

    for (unsigned hwtid = start_thread; hwtid < start_thread + padded_cta_size;
         hwtid++)
      m_occupied_hwtid.reset(hwtid);
    m_occupied_cta_to_hwtid.erase(hw_ctaid);

    const struct gpgpu_ptx_sim_info *kernel_info = ptx_sim_kernel_info(kernel);

    assert(m_occupied_shmem >= (unsigned int)kernel_info->smem);
    m_occupied_shmem -= kernel_info->smem;

    unsigned int used_regs = padded_cta_size * ((kernel_info->regs + 3) & ~3);
    assert(m_occupied_regs >= used_regs);
    m_occupied_regs -= used_regs;

    assert(m_occupied_ctas >= 1);
    m_occupied_ctas--;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Launches a cooperative thread array (CTA).
 *
 * @param kernel
 *    object that tells us which kernel to ask for a CTA from
 */

unsigned exec_shader_core_ctx::sim_init_thread(
    kernel_info_t &kernel, ptx_thread_info **thread_info, int sid, unsigned tid,
    unsigned threads_left, unsigned num_threads, core_t *core,
    unsigned hw_cta_id, unsigned hw_warp_id, gpgpu_t *gpu) {
  return ptx_sim_init_thread(kernel, thread_info, sid, tid, threads_left,
                             num_threads, core, hw_cta_id, hw_warp_id, gpu);
}

void shader_core_ctx::issue_block2core(kernel_info_t &kernel) {
  if (!m_config->gpgpu_concurrent_kernel_sm)
    set_max_cta(kernel);
  else
    assert(occupy_shader_resource_1block(kernel, true));

  kernel.inc_running();

  // find a free CTA context
  unsigned free_cta_hw_id = (unsigned)-1;

  unsigned max_cta_per_core;
  if (!m_config->gpgpu_concurrent_kernel_sm)
    max_cta_per_core = kernel_max_cta_per_shader;
  else
    max_cta_per_core = m_config->max_cta_per_core;
  for (unsigned i = 0; i < max_cta_per_core; i++) {
    if (m_cta_status[i] == 0) {
      free_cta_hw_id = i;
      break;
    }
  }
  assert(free_cta_hw_id != (unsigned)-1);

  // determine hardware threads and warps that will be used for this CTA
  int cta_size = kernel.threads_per_cta();

  // hw warp id = hw thread id mod warp size, so we need to find a range
  // of hardware thread ids corresponding to an integral number of hardware
  // thread ids
  int padded_cta_size = cta_size;
  if (cta_size % m_config->warp_size)
    padded_cta_size =
        ((cta_size / m_config->warp_size) + 1) * (m_config->warp_size);

  unsigned int start_thread, end_thread;

  if (!m_config->gpgpu_concurrent_kernel_sm) {
    start_thread = free_cta_hw_id * padded_cta_size;
    end_thread = start_thread + cta_size;
  } else {
    start_thread = find_available_hwtid(padded_cta_size, true);
    assert((int)start_thread != -1);
    end_thread = start_thread + cta_size;
    assert(m_occupied_cta_to_hwtid.find(free_cta_hw_id) ==
           m_occupied_cta_to_hwtid.end());
    m_occupied_cta_to_hwtid[free_cta_hw_id] = start_thread;
  }

  // reset the microarchitecture state of the selected hardware thread and warp
  // contexts
  reinit(start_thread, end_thread, false);

  // initalize scalar threads and determine which hardware warps they are
  // allocated to bind functional simulation state of threads to hardware
  // resources (simulation)
  warp_set_t warps;
  unsigned nthreads_in_block = 0;
  function_info *kernel_func_info = kernel.entry();
  symbol_table *symtab = kernel_func_info->get_symtab();
  unsigned ctaid = kernel.get_next_cta_id_single();
  checkpoint *g_checkpoint = new checkpoint();
  for (unsigned i = start_thread; i < end_thread; i++) {
    m_threadState[i].m_cta_id = free_cta_hw_id;
    unsigned warp_id = i / m_config->warp_size;
    nthreads_in_block += sim_init_thread(
        kernel, &m_thread[i], m_sid, i, cta_size - (i - start_thread),
        m_config->n_thread_per_shader, this, free_cta_hw_id, warp_id,
        m_cluster->get_gpu());
    m_threadState[i].m_active = true;
    // load thread local memory and register file
    if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
        ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
      char fname[2048];
      snprintf(fname, 2048, "checkpoint_files/thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      m_thread[i]->resume_reg_thread(fname, symtab);
      char f1name[2048];
      snprintf(f1name, 2048, "checkpoint_files/local_mem_thread_%d_%d_reg.txt",
               i % cta_size, ctaid);
      g_checkpoint->load_global_mem(m_thread[i]->m_local_mem, f1name);
    }
    //
    warps.set(warp_id);
  }
  assert(nthreads_in_block > 0 &&
         nthreads_in_block <=
             m_config->n_thread_per_shader);  // should be at least one, but
                                              // less than max
  m_cta_status[free_cta_hw_id] = nthreads_in_block;

  if (m_gpu->resume_option == 1 && kernel.get_uid() == m_gpu->resume_kernel &&
      ctaid >= m_gpu->resume_CTA && ctaid < m_gpu->checkpoint_CTA_t) {
    char f1name[2048];
    snprintf(f1name, 2048, "checkpoint_files/shared_mem_%d.txt", ctaid);

    g_checkpoint->load_global_mem(m_thread[start_thread]->m_shared_mem, f1name);
  }
  // now that we know which warps are used in this CTA, we can allocate
  // resources for use in CTA-wide barrier operations
  m_barriers.allocate_barrier(free_cta_hw_id, warps);

  // initialize the SIMT stacks and fetch hardware
  init_warps(free_cta_hw_id, start_thread, end_thread, ctaid, cta_size, kernel);
  m_n_active_cta++;

  shader_CTA_count_log(m_sid, 1);
  SHADER_DPRINTF(LIVENESS,
                 "GPGPU-Sim uArch: cta:%2u, start_tid:%4u, end_tid:%4u, "
                 "initialized @(%lld,%lld), kernel_uid:%u, kernel_name:%s\n",
                 free_cta_hw_id, start_thread, end_thread, m_gpu->gpu_sim_cycle,
                 m_gpu->gpu_tot_sim_cycle, kernel.get_uid(),
                 kernel.get_name().c_str());
}

///////////////////////////////////////////////////////////////////////////////////////////

void dram_t::dram_log(int task) {
  if (task == SAMPLELOG) {
    StatAddSample(mrqq_Dist, que_length());
  } else if (task == DUMPLOG) {
    printf("Queue Length DRAM[%d] ", id);
    StatDisp(mrqq_Dist);
  }
}

// Find next clock domain and increment its time
int gpgpu_sim::next_clock_domain(void) {
  // to get the cycles spent for any cuda stream operation before and after the
  // kernel is launched monotonically increase the total simulation cycle
  if (!active()) {
    int mask = 0x00;
    mask |= GMMU;
    gpu_tot_sim_cycle++;
    return mask;
  }

  double smallest = min4(core_time, icnt_time, dram_time, gmmu_time);
  int mask = 0x00;
  if (l2_time <= smallest) {
    smallest = l2_time;
    mask |= L2;
    l2_time += m_config.l2_period;
  }
  if (icnt_time <= smallest) {
    mask |= ICNT;
    icnt_time += m_config.icnt_period;
  }
  if (dram_time <= smallest) {
    mask |= DRAM;
    dram_time += m_config.dram_period;
  }
  if (core_time <= smallest) {
    mask |= CORE;
    core_time += m_config.core_period;
  }
  if (gmmu_time <= smallest) {
    mask |= GMMU;
    gmmu_time += m_config.core_period;
  }
  return mask;
}
void gpgpu_sim::throttle_SM() {
 if (m_shader_config->ETC_throttle_SM) {  // ETC_throttle_SM option
    // page_fault_latency 간격으로만 throttle 조절
    unsigned long long current_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
    if (current_cycle - last_throttle_adjustment_cycle >= m_config.page_fault_latency) {
      last_throttle_adjustment_cycle = current_cycle;      // 실제 활성 SM 수 계산
      print_throttle_count++;
      if(print_throttle_count%1000==0){
        m_gmmu->process_print_on=true;
      }

      if (m_gmmu->need_throttle_SM&&m_gmmu->fault_occured) {
        if (num_throttled_SM == m_shader_config->n_simt_clusters) {
          num_throttled_SM = m_shader_config->n_simt_clusters / 2;
        } else if (num_throttled_SM > 1) {
          num_throttled_SM -= 1;
        }
        m_gmmu->need_throttle_SM = false;
        m_gmmu->fault_occured = false;
      } else {
        num_throttled_SM += 1;
      }
    }
  } else if (m_config.m_shader_config.CURE_throttle_SM) {
        // page_fault_latency 간격으로만 throttle 조절
    unsigned long long current_cycle = gpu_sim_cycle + gpu_tot_sim_cycle;
    if (current_cycle - last_throttle_adjustment_cycle >= m_config.page_fault_latency) {
      print_throttle_count++;
      if(print_throttle_count%1000==0){
        m_gmmu->process_print_on=true;
      }

      last_throttle_adjustment_cycle = current_cycle;
      // 실제 활성 SM 수 계산 (throttle 조절 전에 계산)
      m_gmmu->need_throttle_SM|=m_gmmu->get_comp_stage_queue_size();
      if (m_gmmu->need_throttle_SM) {
        if (m_gmmu->fault_occured && num_throttled_SM > 2) {
          num_throttled_SM -= 2;
        }
      } 
      else if(num_throttled_SM + m_gmmu->get_decomp_stage_queue_size() < m_shader_config->n_simt_clusters)
      {
        num_throttled_SM += 2;
      }
      m_gmmu->need_throttle_SM = 0;
      m_gmmu->fault_occured = false;
    unsigned after_issue_active_sms=0;
    for(unsigned i=0;i<m_shader_config->n_simt_clusters;i++)
    {
      if(m_cluster[i]->get_not_completed() || get_more_cta_left())
      {
        after_issue_active_sms+=m_cluster[i]->get_n_hardware_active_sms();
      }
    }
    unsigned idle_sms=m_shader_config->n_simt_clusters-(unsigned)((std::min(after_issue_active_sms,num_throttled_SM)));
    //잉여한 SM수는 전체 클러스터 수 - 현재 사용중인 SM수(Throttle된 것보다 작업이 있는 SM수가 더 적을 수 있음)
    if(print_throttle_count%1000==0){
    }
    m_gmmu->max_parallel_comp_requests = idle_sms/2; 
  }
}
}

void gpgpu_sim::issue_block2core() {  // SM들에게 작업을 할당
  unsigned last_issued = m_last_cluster_issue;
  
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
      unsigned idx = (i + last_issued + 1) % m_shader_config->n_simt_clusters;
      unsigned num = m_cluster[idx]->issue_block2core();

      if (num) {
        m_last_cluster_issue = idx;
        m_total_cta_launched += num;
      }
    }
}

unsigned long long g_single_step =
    0;  // set this in gdb to single step the pipeline

gpgpu_new_stats::gpgpu_new_stats(const gpgpu_sim_config &config)
    : m_config(config) {
  tlb_hit = new unsigned long long[m_config.num_cluster()];
  tlb_miss = new unsigned long long[m_config.num_cluster()];
  tlb_val = new unsigned long long[m_config.num_cluster()];
  tlb_evict = new unsigned long long[m_config.num_cluster()];
  tlb_page_evict = new unsigned long long[m_config.num_cluster()];

  mf_page_hit = new unsigned long long[m_config.num_cluster()];
  mf_page_miss = new unsigned long long[m_config.num_cluster()];
  mf_page_compressed = new unsigned long long[m_config.num_cluster()];

  mf_page_fault_outstanding = 0;
  mf_page_fault_pending = 0;
  mf_page_comp_pending = 0;

  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    tlb_hit[i] = 0;
    tlb_miss[i] = 0;
    tlb_val[i] = 0;
    tlb_evict[i] = 0;
    tlb_page_evict[i] = 0;
    mf_page_hit[i] = 0;
    mf_page_miss[i] = 0;
    mf_page_compressed[i] = 0;
  }

  pf_page_hit = 0;
  pf_page_miss = 0;

  page_evict_not_dirty = 0;
  page_evict_dirty = 0;
    collect_access_trace = m_config.collect_access_trace;
    collect_memory_access_stats = m_config.collect_memory_access_stats;
  num_dma = 0;
  inefficient_comp = 0;
  comp_evcit_num = 0;
  comp_evict_clean = 0;
  comp_fault_num = 0;
  comp_evict_dirty = 0;
  compress = 0;
  decompress = 0;
  dma_page_transfer_read = 0;
  dma_page_transfer_write = 0;
  mf_page_comp_pending = 0;

  tlb_thrashing =
      new std::map<mem_addr_t, std::vector<bool>>[m_config.num_cluster()];

  ma_latency =
      new std::map<unsigned,
                   std::pair<bool, unsigned long long>>[m_config.num_cluster()];

  page_access_times =
      new std::map<mem_addr_t, unsigned>[m_config.num_cluster()];
}

void gpgpu_new_stats::print_pcie(FILE *fout) const {
  fprintf(fout, "Read lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
  fprintf(fout, "Write lanes:\n");
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    fprintf(fout, "%llu %f\n", iter->first, iter->second);
  }
}

void gpgpu_new_stats::print_access_pattern_detail(FILE *fout) const {
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout, "Shader %u\n", i);
    for (std::map<mem_addr_t, unsigned>::const_iterator iter =
             page_access_times[i].begin();
         iter != page_access_times[i].end(); iter++) {
      fprintf(fout, "%llu %u\n", iter->first, iter->second);
    }
  }
}

void gpgpu_new_stats::print_access_pattern(FILE *fout) const {
  std::map<mem_addr_t, unsigned> tot_access;
  fprintf(fout, "Total page access pttern:\n");
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, unsigned>::const_iterator iter =
             page_access_times[i].begin();
         iter != page_access_times[i].end(); iter++) {
      tot_access[iter->first] += iter->second;
    }
  }
  for (std::map<mem_addr_t, unsigned>::const_iterator iter = tot_access.begin();
       iter != tot_access.end(); iter++) {
    fprintf(fout, "%llu %u\n", iter->first, iter->second);
  }
}

void gpgpu_new_stats::print_time_and_access(FILE *fout) const {
  for (std::list<access_info>::const_iterator iter =
           time_and_page_access.begin();
       iter != time_and_page_access.end(); iter++) {
    fprintf(fout, "%u 0x%llx %zu %llu %d %u %u\n", iter->page_no, iter->mem_addr,
            iter->size, iter->cycle, iter->is_read, iter->sm_id, iter->warp_id);
  }

  for (std::map<unsigned long long, std::list<event_stats *>>::iterator iter =
           sim_prof.begin();
       iter != sim_prof.end(); iter++) {
    for (std::list<event_stats *>::iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if ((*iter2)->type == kernel_launched) {
        fprintf(fout, "K: %llu %llu\n", (*iter2)->start_time,
                (*iter2)->end_time);
      }
    }
  }
}

void gpgpu_new_stats::print(FILE *fout) const {
  fprintf(fout,
          "========================================UVM "
          "statistics==============================\n");

  fprintf(fout,
          "========================================TLB "
          "statistics(access)==============================\n");
  unsigned long long tot_tlb_hit = 0;
  unsigned long long tot_tlb_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_access: %llu Tlb_hit: %llu Tlb_miss: %llu "
            "Tlb_hit_rate: %f\n",
            i, tlb_hit[i] + tlb_miss[i], tlb_hit[i], tlb_miss[i],
            ((float)tlb_hit[i]) / ((float)(tlb_hit[i] + tlb_miss[i])));
    tot_tlb_hit += tlb_hit[i];
    tot_tlb_miss += tlb_miss[i];
  }

  fprintf(fout,
          "Tlb_tot_access: %llu Tlb_tot_hit: %llu, Tlb_tot_miss: %llu, "
          "Tlb_tot_hit_rate: %f\n",
          tot_tlb_hit + tot_tlb_miss, tot_tlb_hit, tot_tlb_miss,
          ((float)tot_tlb_hit) / ((float)(tot_tlb_hit + tot_tlb_miss)));

  fprintf(fout,
          "========================================TLB "
          "statistics(validate)==============================\n");
  unsigned long long tot_tlb_val = 0;
  unsigned long long tot_tlb_inval_te = 0;
  unsigned long long tot_tlb_inval_pe = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(fout,
            "Shader%u: Tlb_validate: %llu Tlb_invalidate: %llu Tlb_evict: %llu "
            "Tlb_page_evict: %llu\n",
            i, tlb_val[i], tlb_evict[i] + tlb_page_evict[i], tlb_evict[i],
            tlb_page_evict[i]);
    tot_tlb_val += tlb_val[i];
    tot_tlb_inval_te += tlb_evict[i];
    tot_tlb_inval_pe += tlb_page_evict[i];
  }

  fprintf(fout,
          "Tlb_tot_valiate: %llu Tlb_invalidate: %llu, Tlb_tot_evict: %llu, "
          "Tlb_tot_evict page: %llu\n",
          tot_tlb_val, tot_tlb_inval_te + tot_tlb_inval_pe, tot_tlb_inval_te,
          tot_tlb_inval_pe);

  fprintf(fout,
          "========================================TLB "
          "statistics(thrashing)==============================\n");
  std::map<mem_addr_t, unsigned> tlb_thrash[m_config.num_cluster()];
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
             tlb_thrashing[i].begin();
         iter != tlb_thrashing[i].end(); iter++) {
      for (unsigned j = 0; j != iter->second.size(); j++) {
        if (j + 2 >= iter->second.size()) break;
        if (iter->second[j] == true && iter->second[j + 1] == false &&
            iter->second[j + 2] == true)
          tlb_thrash[i][iter->first]++;
      }
    }
  }

  unsigned tot_tlb_thrash = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    unsigned s_thrash = 0;
    fprintf(fout, "Shader%u: ", i);
    for (std::map<mem_addr_t, unsigned>::iterator iter = tlb_thrash[i].begin();
         iter != tlb_thrash[i].end(); iter++) {
      fprintf(fout, "Page: %llu Trashed: %u | ", iter->first, iter->second);
      s_thrash += iter->second;
    }
    fprintf(fout, "Total %u\n", s_thrash);
    tot_tlb_thrash += s_thrash;
  }
  fprintf(fout, "Tlb_tot_thrash: %u\n", tot_tlb_thrash);

  fprintf(fout,
          "========================================Page fault "
          "statistics==============================\n");

  unsigned long long tot_page_hit = 0;
  unsigned long long tot_page_miss = 0;
  for (unsigned i = 0; i < m_config.num_cluster(); i++) {
    fprintf(
        fout,
        "Shader%u: Page_table_access:%llu Page_hit: %llu Page_miss: %llu "
        "Page_hit_rate: %f\n",
        i, mf_page_hit[i] + mf_page_miss[i], mf_page_hit[i], mf_page_miss[i],
        ((float)mf_page_hit[i]) / ((float)(mf_page_hit[i] + mf_page_miss[i])));
    tot_page_hit += mf_page_hit[i];
    tot_page_miss += mf_page_miss[i];
  }

  fprintf(fout,
          "Page_table_tot_access: %llu Page_tot_hit: %llu, Page_tot_miss %llu, "
          "Page_tot_hit_rate: %f Page_tot_fault: %llu Page_tot_pending: %llu\n",
          tot_page_hit + tot_page_miss, tot_page_hit, tot_page_miss,
          ((float)tot_page_hit) / ((float)(tot_page_hit + tot_page_miss)),
          mf_page_fault_outstanding, mf_page_fault_pending);

  float avg_mf_latency = 0;
  unsigned long long tot_mf_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator
           iter = mf_page_fault_latency.begin();
       iter != mf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 =
             iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      avg_mf_latency =
          ((float)tot_mf_fault) / ((float)(tot_mf_fault + 1)) * avg_mf_latency +
          ((float)(*iter2)) / ((float)(tot_mf_fault + 1));
      tot_mf_fault++;
    }
  }
  fprintf(fout,
          "Total_memory_access_page_fault: %llu, Average_latency: %f, "
          "skipped cycles: %llu, Total_skipped_cycles: %llu \n",
          tot_mf_fault, avg_mf_latency, skipped_cycles, gpu_tot_skipped_cycle);

  fprintf(fout,
          "========================================Page thrashing "
          "statistics==============================\n");

  unsigned long long tot_validate = 0;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
           page_thrashing.begin();
       iter != page_thrashing.end(); iter++) {
    for (std::vector<bool>::const_iterator iter2 = iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      if (*iter2 == true) tot_validate++;
    }
  }
  size_t total_comp_size = 0;
  size_t comp_count = 0;
  fprintf(fout, "========================================Comp_size "
          "statistics==============================\n");
  fprintf(fout, "Comp_size_list: ");
  for (std::list<size_t>::const_iterator iter = comp_size_list.begin();
       iter != comp_size_list.end(); iter++) {
    fprintf(fout, "%zu, ", *iter);
    total_comp_size += *iter;
    comp_count++;
  }
  if (comp_count > 0) {
    fprintf(fout, "Average_comp_size: %f\n", (float)total_comp_size / comp_count);
  }
  fprintf(fout, "Inefficient_comp: %llu\n", inefficient_comp);
  fprintf(fout, "Comp_evcit_num: %llu\n", comp_evcit_num);
  fprintf(fout, "Comp_evict_clean: %llu\n", comp_evict_clean);
  fprintf(fout, "Comp_fault_num: %llu\n", comp_fault_num);
  fprintf(fout, "Comp_evict_dirty: %llu\n", comp_evict_dirty);
  fprintf(fout, "Compress: %llu\n", compress);
  fprintf(fout, "Decompress: %llu\n", decompress);
  fprintf(
      fout,
      "Page_validate: %llu Page_evict_dirty: %llu Page_evict_not_dirty: %llu\n",
      tot_validate, page_evict_dirty, page_evict_not_dirty);

  std::map<mem_addr_t, unsigned> page_thrash;
  for (std::map<mem_addr_t, std::vector<bool>>::const_iterator iter =
           page_thrashing.begin();
       iter != page_thrashing.end(); iter++) {
    for (unsigned j = 0; j != iter->second.size(); j++) {
      if (j + 2 >= iter->second.size()) break;
      if (iter->second[j] == true && iter->second[j + 1] == false &&
          iter->second[j + 2] == true)
        page_thrash[iter->first]++;
    }
  }

  unsigned tot_page_thrash = 0;
  for (std::map<mem_addr_t, unsigned>::iterator iter = page_thrash.begin();
       iter != page_thrash.end(); iter++) {
    fprintf(fout, "Page: %llu Thrashed: %u\n", iter->first, iter->second);
    tot_page_thrash += iter->second;
  }
  fprintf(fout, "Page_tot_thrash: %u\n", tot_page_thrash);

  fprintf(fout,
          "========================================Memory access "
          "statistics==============================\n");

  fprintf(fout,
          "========================================Prefetch "
          "statistics==============================\n");

  fprintf(fout,
          "Tot_page_hit: %llu, Tot_page_miss: %llu, Tot_page_fault: %lu\n",
          pf_page_hit, pf_page_miss, pf_fault_latency.size());

  float avg_pf_latency = 0;
  float avg_pref_size = 0;
  float avg_pref_latency = 0;

  unsigned long long tot_pf_fault = 0;
  unsigned long long tot_pref_fault = 0;
  for (std::map<mem_addr_t, std::list<unsigned long long>>::const_iterator
           iter = pf_page_fault_latency.begin();
       iter != pf_page_fault_latency.end(); iter++) {
    for (std::list<unsigned long long>::const_iterator iter2 =
             iter->second.begin();
         iter2 != iter->second.end(); iter2++) {
      avg_pf_latency =
          ((float)tot_pf_fault) / ((float)(tot_pf_fault + 1)) * avg_pf_latency +
          ((float)(*iter2)) / ((float)(tot_pf_fault + 1));
      tot_pf_fault++;
    }
  }

  for (std::vector<std::pair<unsigned long, unsigned long long>>::const_iterator
           iter = pf_fault_latency.begin();
       iter != pf_fault_latency.end(); iter++) {
    avg_pref_size += iter->first;
    avg_pref_latency = ((float)tot_pref_fault) / ((float)(tot_pref_fault + 1)) *
                           avg_pref_latency +
                       ((float)(iter->second)) / ((float)(tot_pref_fault + 1));
    tot_pref_fault++;
  }

  avg_pref_size /= ((float)pf_fault_latency.size());
  fprintf(
      fout,
      "Avg_page_latency: %f, Avg_prefetch_size: %f, Avg_prefetch_latency: %f\n",
      avg_pf_latency, avg_pref_size, avg_pref_latency);

  fprintf(fout,
          "========================================Rdma "
          "statistics==============================\n");
  fprintf(fout, "dma_read: %llu\n", num_dma);
  fprintf(fout, "dma_migration_read %llu\n", dma_page_transfer_read);
  fprintf(fout, "dma_migration_write %llu\n", dma_page_transfer_write);

  fprintf(fout,
          "========================================PCI-e "
          "statistics==============================\n");
  float avg_r = 0;
  unsigned long long r_0 = 0;
  unsigned long long r_25 = 0;
  unsigned long long r_50 = 0;
  unsigned long long r_75 = 0;
  unsigned long long r_tot = 0;
  float avg_w = 0;
  unsigned long long w_0 = 0;
  unsigned long long w_25 = 0;
  unsigned long long w_50 = 0;
  unsigned long long w_75 = 0;
  unsigned long long w_tot = 0;
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_read_utilization.begin();
       iter != pcie_read_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      r_0++;
    } else if (iter->second <= 0.5) {
      r_25++;
    } else if (iter->second <= 0.75) {
      r_50++;
    } else {
      r_75++;
    }
    avg_r = (avg_r * ((float)r_tot) + iter->second) / ((float)(r_tot + 1));
    r_tot++;
  }
  for (std::list<std::pair<unsigned long long, float>>::const_iterator iter =
           pcie_write_utilization.begin();
       iter != pcie_write_utilization.end(); iter++) {
    if (iter->second <= 0.25) {
      w_0++;
    } else if (iter->second <= 0.5) {
      w_25++;
    } else if (iter->second <= 0.75) {
      w_50++;
    } else {
      w_75++;
    }
    avg_w = (avg_w * ((float)w_tot) + iter->second) / ((float)(w_tot + 1));
    w_tot++;
  }

  fprintf(fout, "Pcie_read_utilization: %f\n", avg_r);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)r_0) / ((float)r_tot), ((float)r_25) / ((float)r_tot),
          ((float)r_50) / ((float)r_tot), ((float)r_75) / ((float)r_tot));
  fprintf(fout, "Pcie_write_utilization: %f\n", avg_w);
  fprintf(fout, "[0-25]: %f, [26-50]: %f, [51-75]: %f, [76-100]: %f\n",
          ((float)w_0) / ((float)w_tot), ((float)w_25) / ((float)w_tot),
          ((float)w_50) / ((float)w_tot), ((float)w_75) / ((float)w_tot));
}

gpgpu_new_stats::~gpgpu_new_stats() {
  delete[] tlb_hit;
  delete[] tlb_miss;
  delete[] tlb_val;
  delete[] tlb_evict;
  delete[] tlb_page_evict;
  delete[] mf_page_hit;
  delete[] mf_page_miss;
  delete[] page_access_times;
  delete[] tlb_thrashing;
  delete[] ma_latency;
}

gmmu_t::gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
               class gpgpu_new_stats *new_stats)
    : m_gpu(gpu), m_config(config), m_new_stats(new_stats) {
  m_shader_config = &m_config.m_shader_config;
  g_memory = m_gpu->get_global_memory();

  if (m_config.enable_dma == 0) {
    dma_mode = dma_type::DISABLED;
  } else if (m_config.enable_dma == 1) {
    dma_mode = dma_type::ADAPTIVE;
  } else if (m_config.enable_dma == 2) {
    dma_mode = dma_type::ALWAYS;
  } else if (m_config.enable_dma == 3) {
    dma_mode = dma_type::OVERSUB;
  } else {
    printf("Unknown DMA mode\n");
    exit(1);
  }

  if (m_config.eviction_policy == 0) {
    evict_policy = eviction_policy::LRU;
  } else if (m_config.eviction_policy == 1) {
    evict_policy = eviction_policy::TBN;
  } else if (m_config.eviction_policy == 2) {
    evict_policy = eviction_policy::SEQUENTIAL_LOCAL;
  } else if (m_config.eviction_policy == 3) {
    evict_policy = eviction_policy::RANDOM;
  } else if (m_config.eviction_policy == 4) {
    evict_policy = eviction_policy::LFU;
  } else if (m_config.eviction_policy == 5) {
    evict_policy = eviction_policy::LRU4K;
  } else {
    printf("Unknown eviction policy");
    exit(1);
  }
  tbn_threshold_float = (float)100 / (float)m_config.tbn_threshold;
  
  if (m_config.compression_policy == 0) {
    compression_policy = compression_policy_struct::NONE;
    printf("BST debug: eviction policy ==0\n");
  } else if (m_config.compression_policy == 2) {
    printf("BST debug :eviction policy ==2\n");
    compression_policy = compression_policy_struct::FULL;
  } else if (m_config.compression_policy == 1) {
    printf("BST debug: eviction policy ==1\n");
    compression_policy = compression_policy_struct::PARTIAL;
  }
  if (m_config.ignore_reserved_pages) {
    m_ignore_reserved_pages = true;
  }
  if (m_config.selective_compression) {
    selective_compression = true;
  }
  if (m_config.full_comp_policy) {
    full_comp_policy = true;
  }
  if (m_config.comp_evict_policy) {
    comp_evict_policy = true;
  }
  if (m_config.MICRO_barrier_Range) {
    // MICRO_barrier_Range decouples comp/decomp (64KB) from eviction (2MB) and only
    // makes sense on the range tree with partial compression + compressed eviction.
    // Other policies (e.g. page_eviction_procedure -> evict_whole_tree) assume uniform
    // 2MB-leaf state and are not adapted for mixed-state leaves.
    assert(m_config.tree_logic == 2 &&
           "MICRO_barrier_Range requires -tree_logic 2 (range tree)");
    assert(m_config.compression_policy == 1 &&
           "MICRO_barrier_Range requires -compression_policy 1 (PARTIAL)");
    assert(m_config.comp_evict_policy &&
           "MICRO_barrier_Range requires -comp_evict_policy 1");
    printf("BST debug: MICRO_barrier_Range enabled (comp/decomp 64KB, eviction 2MB)\n");
  }
  comp_ratio = m_config.comp_ratio;
  if (m_config.hardware_prefetch == 0) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (m_config.hardware_prefetch == 1) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (m_config.hardware_prefetch == 2) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (m_config.hardware_prefetch == 3) {
    prefetcher = hwardware_prefetcher::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy");
    exit(1);
  }

  if (m_config.hwprefetch_oversub == 0) {
    oversub_prefetcher = hwardware_prefetcher_oversub::DISBALED;
  } else if (m_config.hwprefetch_oversub == 1) {
    oversub_prefetcher = hwardware_prefetcher_oversub::TBN;
  } else if (m_config.hwprefetch_oversub == 2) {
    oversub_prefetcher = hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL;
  } else if (m_config.hwprefetch_oversub == 3) {
    oversub_prefetcher = hwardware_prefetcher_oversub::RANDOM;
  } else {
    printf("Unknown hardware prefeching policy under over-subscription");
    exit(1);
  }

  pcie_read_latency_queue = NULL;
  pcie_write_latency_queue = NULL;
  printf("bst debug gmmu_t::gmmu_t() - compression_engine = %d\n", m_config.compression_engine);
  if (m_config.compression_engine == 2) {
    compressor = new BDICompressor("BDI");
  } else if (m_config.compression_engine == 1 ||
             m_config.compression_policy) {  // LCP_BPC or CURE_MEM
    compressor = new BPCompressor("BPC");
  }
  skip_compression_selection = false;
  num_write_stage_queue = 0;
  num_read_stage_queue = 0;
  try_error = 0;
  num_comp_stage_queue = 0;
  num_decomp_stage_queue = 0;
  num_comp_evict_stage_queue = 0;
  total_allocation_size = 0;

  over_sub = false;
  max_parallel_comp_requests = 1;
  current_stalled_SM=0;
  printf("bst debug max_parallel_comp_requests: %zu\n",
         max_parallel_comp_requests);
}

unsigned long long gmmu_t::calculate_transfer_time(size_t data_size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));

  if (data_size >= 2 * 1024 * 1024) {
    speed /= 2;
  }

  return (unsigned long long)((float)(data_size)*m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}
unsigned long long gmmu_t::calculate_decompres_time(size_t num_pages,
                                                    float comp_band) {
  float speed = 2.0 * comp_band / M_PI * m_config.curve_b *
                ((float)(num_pages * m_config.page_size) / 1024.0);

  return (unsigned long long)((float)(num_pages * m_config.page_size) *
                              m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}
void gmmu_t::calculate_devicesync_time(size_t data_size) {
  unsigned cur_turn = 0;
  unsigned cur_size = 0;

  float speed;

  while (data_size != 0) {
    unsigned long long cur_cycle =
        m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
    unsigned long long cur_time = 0;

    if (cur_turn == 0) {
      cur_size = MIN_PREFETCH_SIZE;
    } else {
      cur_size = MIN_PREFETCH_SIZE * pow(2, cur_turn - 1);
    }

    if (data_size < 4096) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
      cur_cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      return;
    } else {
      cur_size -= 4096;
      data_size -= 4096;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(4096) / (float)(1024)));
      cur_time = (unsigned long long)((float)(4096) * m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, 4096, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
      cur_cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    }

    if (data_size < cur_size) {
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(data_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(data_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, data_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
      cur_cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      return;
    } else {
      data_size -= cur_size;
      speed = 2.0 * m_config.curve_a / M_PI *
              atan(m_config.curve_b * ((float)(cur_size) / (float)(1024)));
      cur_time = (unsigned long long)((float)(cur_size)*m_config.core_freq /
                                      speed / (1024.0 * 1024.0 * 1024.0));

      if (sim_prof_enable) {
        event_stats *d_sync = new memory_stats(
            device_sync, cur_cycle, cur_cycle + cur_time, 0, cur_size, 0);
        sim_prof[cur_cycle].push_back(d_sync);
      }

      m_gpu->gpu_tot_sim_cycle += cur_time;
      cur_cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    }

    cur_turn++;
    if (cur_turn == 6) {
      cur_turn = 0;
    }
  }
  return;
}

bool gmmu_t::pcie_transfers_completed() {
  if (m_config.compression_policy == 0) {
    return pcie_write_stage_queue.empty() &&
           pcie_write_latency_queue == nullptr && pcie_read_stage_queue.empty() &&
           pcie_read_latency_queue == nullptr;  //&& comp_latency_queue.empty()
  } else {
    return pcie_read_stage_queue.empty() &&
           pcie_read_latency_queue == nullptr ;  //&& comp_latency_queue.empty()
  }
}
bool gmmu_t::compression_completed() {
  return comp_stage_queue.empty() && comp_latency_queue.empty();
}
void gmmu_t::register_tlbflush_callback(
    std::function<void(mem_addr_t)> cb_tlb) {
  callback_tlb_flush.push_back(cb_tlb);
}

void gmmu_t::tlb_flush(mem_addr_t page_num) {
  for (list<std::function<void(mem_addr_t)>>::iterator iter =
           callback_tlb_flush.begin();
       iter != callback_tlb_flush.end(); iter++) {
    (*iter)(page_num);
  }
}
bool gmmu_t::check_write_latency_queue(mem_addr_t page_num, bool refresh = true) {
  bst_debug_print("[DEBUG] check_write_latency_queue 시작: page_num=%lx\n",
                  page_num);
  if(pcie_write_latency_queue != NULL){
    for(auto iter = pcie_write_latency_queue->page_list.begin(); iter != pcie_write_latency_queue->page_list.end(); iter++){
      if(*iter == page_num){
        return true;
      }
    }
  }
  return false;
}
//new 2MB check
bool gmmu_t::check_write_stage_queue_2MB(mem_addr_t page_num, bool refresh = true) {
  auto global_memory = m_gpu->get_global_memory();
  mem_addr_t target_page_addr = global_memory->get_mem_addr(page_num);
  auto target_node = get_lp_node(target_page_addr);
  bool ret = false;

  // 큐가 비어있으면 빠르게 리턴
  if (pcie_write_stage_queue.empty()) {
    return false;
  }

  // 반복자를 사용하여 erase() 호출 가능하도록 수정
  auto iter = pcie_write_stage_queue.begin();
  while (iter != pcie_write_stage_queue.end()) {
    auto queue_entry = *iter;
    mem_addr_t front_addr = global_memory->get_mem_addr(queue_entry->page_list.front());
    auto front_node = get_lp_node(front_addr);

    if (target_node == front_node) {
      // 페이지 상태 일괄 업데이트
      for(auto pg_iter = queue_entry->page_list.begin(); pg_iter != queue_entry->page_list.end(); pg_iter++) {
        global_memory->set_page_access(*pg_iter);
        global_memory->set_page_dirty(*pg_iter);
        mem_addr_t page_addr = global_memory->get_mem_addr(*pg_iter);
        update_basic_block(front_node, page_addr, m_config.page_size, true);
        cancel_evicting_pages(*pg_iter, front_node->leaf_size);
        if(refresh) {
          refresh_valid_pages(page_addr,false);
        }
      }
      num_write_stage_queue -= queue_entry->page_list.size();
      iter = pcie_write_stage_queue.erase(iter);
      ret = true;
    } else {
      ++iter;
    }
  }
  return ret;
}
void gmmu_t::set_decomp_mf_wait_for_comp(mem_addr_t addr){
  for(auto iter = decomp_stage_queue.begin(); iter != decomp_stage_queue.end(); iter++){
    if((*iter)->start_addr <= addr && addr < (*iter)->start_addr + (*iter)->size){
      (*iter)->mf_wait_for_comp = true;
      break;
    }
  }
}
bool gmmu_t::check_write_stage_queue(mem_addr_t page_num, bool refresh = true) {
  bst_debug_print("[DEBUG] check_write_stage_queue 시작: page_num=%lx\n", page_num);

  if(m_config.tree_logic==1||m_config.fault_based_LRU) {
    return check_write_stage_queue_2MB(page_num,refresh);
  }

  auto global_memory = m_gpu->get_global_memory();
  bool check_cont = true;

  // 현재 page_num이 속하는 트리의 leaf_size 범위 계산
  // MICRO 모드에서는 접근한 64KB 블록만 cancel/recover 대상으로 삼는다.
  mem_addr_t check_page_addr = global_memory->get_mem_addr(page_num);
  auto root = get_lp_node(check_page_addr);
  bool micro = (m_config.MICRO_barrier_Range && m_config.tree_logic == 2);
  mem_addr_t bb_addr = micro ? (check_page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE
                             : get_basic_block(root, check_page_addr);
  size_t range_size = micro ? (size_t)MIN_PREFETCH_SIZE : (size_t)root->leaf_size;
  mem_addr_t bb_start_page = global_memory->get_page_num(bb_addr);
  mem_addr_t bb_end_page = global_memory->get_page_num(bb_addr + range_size-1);

  // 같은 leaf_size 범위에 속하는지 체크하는 람다 함수
  auto is_in_same_leaf_range = [&](mem_addr_t check_page_num) -> bool {//페이지 단위로 확인
    return check_page_num >= bb_start_page && check_page_num <= bb_end_page;
  };

  // pcie_write_stage_queue 처리
  for(auto iter = pcie_write_stage_queue.begin(); iter != pcie_write_stage_queue.end();) {
    auto& queue_entry = *iter;
    bool found = false;

    // 큐 엔트리의 page_list에서 같은 leaf_size 범위에 속하는 페이지가 있는지 체크
    for(const auto& pg : queue_entry->page_list) {
      if(is_in_same_leaf_range(pg)) {
        found = true;
        break;
      }
    }

    if(found) {
      if(m_config.tree_logic == 1 && !get_lp_node(global_memory->get_mem_addr(queue_entry->page_list.front()))->large_pages_reserved){
        //if TBN+2MB eviction, if large page is not reserved, restore allocation and reserve large page
        global_memory->alloc_pages(512);
        get_lp_node(global_memory->get_mem_addr(queue_entry->page_list.front()))->large_pages_reserved = true;
      }
      if(queue_entry->type == latency_type::COMP_EVICT || queue_entry->type == latency_type::COMP_INV_EVICT) {
        // 압축 제거 처리
        for(const auto& pg : queue_entry->page_list) {
          global_memory->set_page_access(pg);
          global_memory->set_page_dirty(pg);
          
          mem_addr_t page_addr = global_memory->get_mem_addr(pg);
          auto root = get_lp_node(page_addr);
          comp_evict_erase_recover(pg, micro_leaf(root));
        }
        update_basic_block_comp_evict(root, global_memory->get_mem_addr(queue_entry->page_list.front()), m_config.page_size*queue_entry->page_list.size(), true);
        num_write_stage_queue -= queue_entry->comp_size / m_config.page_size;
        num_comp_evict_stage_queue -= queue_entry->page_list.size();
        
      } else {
        // 일반 제거 처리
        for(const auto& pg : queue_entry->page_list) {
          global_memory->set_page_access(pg);
          global_memory->set_page_dirty(pg);
        }
        mem_addr_t chunk_start_addr = queue_entry->start_addr;
        unsigned int chunk_size = queue_entry->size;
        auto root = get_lp_node(chunk_start_addr);
        if(chunk_size < root->leaf_size){
          // leaf_size보다 작은 경우도 처리
          update_basic_block(root, chunk_start_addr, chunk_size, true);
          mem_addr_t chunk_page_num = global_memory->get_page_num(chunk_start_addr);
          cancel_evicting_pages(chunk_page_num, root->leaf_size);
          if(refresh) refresh_valid_pages(chunk_start_addr, true);
        } else {
          int num_leaf_size_blocks = chunk_size / root->leaf_size;
          for(int i = 0; i < num_leaf_size_blocks; i++){
            mem_addr_t page_addr = chunk_start_addr + i * root->leaf_size;
            update_basic_block(root, page_addr, root->leaf_size, true);
            mem_addr_t page_page_num = global_memory->get_page_num(page_addr);
            cancel_evicting_pages(page_page_num, root->leaf_size);
            if(refresh) refresh_valid_pages(page_addr, true);
          }
          // leaf_size로 나누어 떨어지지 않는 나머지 처리
          mem_addr_t remain = chunk_size % root->leaf_size;
          if(remain > 0){
            mem_addr_t page_addr = chunk_start_addr + num_leaf_size_blocks * root->leaf_size;
            update_basic_block(root, page_addr, remain, true);
            mem_addr_t page_page_num = global_memory->get_page_num(page_addr);
            cancel_evicting_pages(page_page_num, root->leaf_size);
            if(refresh) refresh_valid_pages(page_addr, true);
          }
        }
        
        num_write_stage_queue -= queue_entry->page_list.size();
      }
    }
    iter = found ? pcie_write_stage_queue.erase(iter) : std::next(iter);
  }
  
  // comp_stage_queue 처리 
  std::set<mem_addr_t> processed_comp_stage_leaf_ranges; // comp_stage_queue에서 처리된 leaf 범위 추적 (bb_start_page 기준, cancel_compressing_pages가 true를 반환한 경우만)
  for(auto iter = comp_stage_queue.begin(); iter != comp_stage_queue.end();) {
    auto& queue_entry = *iter;
    bool found = false;

    // 큐 엔트리의 page_list에서 같은 leaf_size 범위에 속하는 페이지가 있는지 체크
    for(const auto& pg : queue_entry->page_list) {
      if(is_in_same_leaf_range(pg)) {
        found = true;
        break;
      }
    }

    if(found) {
      for(const auto& pg : queue_entry->page_list) {
        global_memory->set_page_access(pg);
        global_memory->set_page_dirty(pg);
        
        mem_addr_t page_addr = global_memory->get_mem_addr(pg);
        auto root = get_lp_node(page_addr);
        bool cancel_changed_size = cancel_compressing_pages(pg, micro_leaf(root));

        // cancel_compressing_pages가 true를 반환한 경우에만 processed_comp_stage_leaf_ranges에 추가
        // MICRO 모드에서는 dedup 키를 64KB 블록 단위로 잡아 서로 다른 64KB 블록이 섞이지 않게 한다.
        if(cancel_changed_size) {
          mem_addr_t pg_bb_addr = micro ? (page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE
                                        : get_basic_block(root, page_addr);
          mem_addr_t pg_bb_start_page = global_memory->get_page_num(pg_bb_addr);
          processed_comp_stage_leaf_ranges.insert(pg_bb_start_page);
        }
        
        update_basic_block_comp(root, page_addr, m_config.page_size, true);
        if(refresh) refresh_valid_pages(page_addr, true);

      }
      num_comp_stage_queue -= (queue_entry->size - queue_entry->comp_size) / m_config.page_size;
    }
    iter = found ? comp_stage_queue.erase(iter) : std::next(iter);
  }

  // comp_latency_queue 처리
  if(!comp_latency_queue.empty()) {
    for(auto iter = comp_latency_queue.begin(); iter != comp_latency_queue.end();) {
      auto& queue_entry = *iter;
      bool found = false;
      bool already_processed_in_comp_stage = false;
      
      // 큐 엔트리의 page_list에서 같은 leaf_size 범위에 속하는 페이지가 있는지 체크
      for(const auto& pg : queue_entry->page_list) {
        if(is_in_same_leaf_range(pg)) {
          found = true;
          // comp_stage_queue에서 같은 leaf 범위가 이미 처리되었는지 확인
          mem_addr_t pg_addr = global_memory->get_mem_addr(pg);
          auto pg_root = get_lp_node(pg_addr);
          mem_addr_t pg_bb_addr = micro ? (pg_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE
                                        : get_basic_block(pg_root, pg_addr);
          mem_addr_t pg_bb_start_page = global_memory->get_page_num(pg_bb_addr);
          if(processed_comp_stage_leaf_ranges.find(pg_bb_start_page) != processed_comp_stage_leaf_ranges.end()) {
            already_processed_in_comp_stage = true;
            break; // 이미 처리된 leaf 범위를 찾았으므로 더 이상 확인할 필요 없음
          }
        }
      }

      if(found) {
        for(const auto& pg : queue_entry->page_list) {
          global_memory->set_page_access(pg);
          global_memory->set_page_dirty(pg);
          
          mem_addr_t page_addr = global_memory->get_mem_addr(pg);
          auto root = get_lp_node(page_addr);
          cancel_compressing_pages(pg, micro_leaf(root),true);
          update_basic_block_comp(root, page_addr, m_config.page_size, true);
          if(refresh) refresh_valid_pages(page_addr, true);
        }

        // comp_stage_queue에서 같은 leaf 범위가 이미 처리된 경우 change_size_comp_with_comp_size 호출하지 않음
        // cancel_compressing_pages가 false를 반환한 경우에만 change_size_comp_with_comp_size 호출
        if(!already_processed_in_comp_stage) {
          if(!global_memory->change_size_comp_with_comp_size(queue_entry->size - queue_entry->comp_size, false)) {
            printf("bst debug check_write_stage_queue comp_latency_queue change_size_comp_with_comp_size failed\n");
            print_debug_all(true);
          }
        } 


        if(sim_prof_enable) {
          for(auto stats_iter = compress_stats.begin(); stats_iter != compress_stats.end(); stats_iter++) {
            if(((memory_stats*)(*stats_iter))->start_addr == global_memory->get_mem_addr(queue_entry->page_list.front())) {
              compress_stats.erase(stats_iter);
              break;
            }
          }
        }
      }
      iter = found ? comp_latency_queue.erase(iter) : std::next(iter);
    }
  }

  bst_debug_print("[DEBUG] check_write_stage_queue 종료: 결과=%d\n", check_cont);
  return check_cont;
}
bool gmmu_t::is_block_precomp(mem_addr_t addr, size_t size) {
  for (auto iter = decomp_stage_queue.begin(); iter != decomp_stage_queue.end();
       ++iter) {
    if ((addr >= (*iter)->start_addr) &&
        (addr < (*iter)->start_addr + (*iter)->size)) {
      return true;
    }
  }
  return false;
}
// check if the block is already scheduled for eviction or is not valid at all
// decomp and pcie write will not be consider if framework work well
// check leaf node can be valid or not
bool gmmu_t::is_block_evictable(mem_addr_t addr, size_t size,
                                bool print = false,bool large_page_check = false) {
  // Early return if no valid pages
  if(!large_page_check) {
    bool has_valid_page = false;
    for (mem_addr_t start = addr; start < addr + size; start += m_config.page_size) {
      if (m_gpu->get_global_memory()->is_valid(m_gpu->get_global_memory()->get_page_num(start)) &&
          !m_gpu->get_global_memory()->is_iter_compressed(m_gpu->get_global_memory()->get_page_num(start))) {
        has_valid_page = true;
        break;
      }
    }
    if(!has_valid_page) return false;
  }

  // Check write stage queue
  for (const auto& entry : pcie_write_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) printf("[DEBUG] %lx 주소의 블록이 pcie_write_stage_queue에 있습니다.\n", check_addr);
        return false;
      }
    }
  }

  // Check comp stage queue  
  for (const auto& entry : comp_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) printf("[DEBUG] %lx 주소의 블록이 comp_stage_queue에 있습니다.\n", check_addr);
        return false;
      }
    }
  }

  // Check decomp stage queue
  for (const auto& entry : decomp_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) printf("[DEBUG] %lx 주소의 블록이 decomp_stage_queue에 있습니다.\n", check_addr);
        return false;
      }
    }
  }

  // Check latency queues
  if (!comp_latency_queue.empty()) {
    for (const auto& entry : comp_latency_queue) {
      for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
        if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) return false;
      }
    }
  }

  if (!decomp_latency_queue.empty()) {
    for (const auto& entry : decomp_latency_queue) {
      for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
        if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) return false;
      }
    }
  }

  // Check reserved pages
  if (!m_ignore_reserved_pages) {
  for (mem_addr_t start = addr; start < addr + size; start += m_config.page_size) {
    if (!reserve_pages_check(start)) {
      if (print) printf("[DEBUG] %lx 주소의 페이지 예약 확인에 실패했습니다.\n", start);
      return false;
    }
  }
  }
  // Check read queues
  for (const auto& entry : pcie_read_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) return false;
    }
  }

  if (pcie_read_latency_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= pcie_read_latency_queue->start_addr && 
          check_addr < pcie_read_latency_queue->start_addr + pcie_read_latency_queue->size) {
        return false;
      }
    }
  }

  if (print) printf("[DEBUG] %lx 주소의 블록이 이빅션가능.\n", addr);
  return true;
}
bool gmmu_t::is_block_comp_evict(mem_addr_t addr, size_t size, bool print) {
  // Check for compressed pages
  auto global_memory = m_gpu->get_global_memory();
  bool found_compressed = false;
  for (mem_addr_t start = addr; start < addr + size; start += m_config.page_size) {
    auto page_num = global_memory->get_page_num(start);
    if (!global_memory->is_valid(page_num) &&
        global_memory->is_iter_compressed(page_num)) {
      found_compressed = true;
      break;
    }
  }
  if (!found_compressed) return false;

  // Check various queues
  for (const auto& entry : pcie_write_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) {
          printf("bst debug is_block_comp_evict pcie_write_stage_queue addr %llu size %llu\n", addr, size);
          print_debug_all(true);
        }
        return false;
      }
    }
  }

  for (const auto& entry : comp_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) {
          printf("bst debug is_block_comp_evict comp_stage_queue addr %llu size %llu\n", addr, size);
          print_debug_all(true);
        }
        return false;
      }
    }
  }

  for (const auto& entry : decomp_stage_queue) {
    for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
      if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
        if (print) {
          printf("bst debug is_block_comp_evict decomp_stage_queue addr %llu size %llu\n", addr, size);
          print_debug_all(true);
        }
        return false;
      }
    }
  }

  if (!decomp_latency_queue.empty()) {
    for (const auto& entry : decomp_latency_queue) {
      for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
        if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
          if (print) {
            printf("bst debug is_block_comp_evict decomp_latency_queue addr %llu size %llu\n", addr, size);
            print_debug_all(true);
          }
          return false;
        }
      }
    }
  }

  if (!comp_latency_queue.empty()) {
    for (const auto& entry : comp_latency_queue) {
      for (mem_addr_t check_addr = addr; check_addr < addr + size; check_addr += m_config.page_size) {
        if (check_addr >= entry->start_addr && check_addr < entry->start_addr + entry->size) {
          if (print) {
            printf("bst debug is_block_comp_evict comp_latency_queue addr %llu size %llu\n", addr, size);
            print_debug_all(true);
          }
          return false;
        }
      }
    }
  }

  // Check reserved pages
  if (!m_ignore_reserved_pages) {
  for (mem_addr_t start = addr; start < addr + size; start += m_config.page_size) {
    if (!reserve_pages_check(start)) {
      if (print) {
        printf("bst debug is_block_comp_evict reserve_pages_check addr %llu size %llu\n", addr, size);
      }
      return false;
    }
  }
}

  return true;
}
// LCP_CompressedPage gmmu_t::find_LCP_compressed_page(mem_addr_t page_no) {
//   auto it = LCP_compressed_pages_map.find(page_no);
//   if (it != LCP_compressed_pages_map.end()) {
//     return it->second;  // 주소 반환 (원본 직접 접근 가능)
//   }
//   return NULL;
// }

void gmmu_t::print_debug_all(bool asser = true) {
  printf("BST debug: assert print cycle %llu\n", cur_cycle);
  printf("BST debug skip comp %d skip evict %d skip comp_evict %d\n", is_comp_enabled(), is_evict_enabled(), is_comp_evict_enabled());
  printf(
      "bst free page %f, comp free page %f, num read stage %f, num write stage "
      "%f, num decomp stage %f, num comp stage %f, num comp evict stage %f\n",
      m_gpu->get_global_memory()->get_free_pages(),
      m_gpu->get_global_memory()->get_comp_free_pages(), num_read_stage_queue,
      num_write_stage_queue, num_decomp_stage_queue, num_comp_stage_queue, num_comp_evict_stage_queue);
  printf("bst gddr page %f init %f\n",
         m_gpu->get_global_memory()->get_total_pages(),
         m_gpu->get_global_memory()->get_init_pages());
  for (auto iter = pcie_write_stage_queue.begin();
       iter != pcie_write_stage_queue.end(); iter++) {
    printf("BST debug PCIE WRITE stage %llu size %llu start_cycle %llu !!!\n",
           (*iter)->start_addr, (*iter)->size, (*iter)->start_cycle);
  }
  if (pcie_write_latency_queue != NULL) {
    printf("BST debug PCIE WRITE latency %llu page_num %llu start_cycle %llu ready_cycle %llu !!!\n", pcie_write_latency_queue->start_addr, m_gpu->get_global_memory()->get_page_num(pcie_write_latency_queue->start_addr), pcie_write_latency_queue->start_cycle, pcie_write_latency_queue->ready_cycle);
  }
  for (auto iter = comp_stage_queue.begin(); iter != comp_stage_queue.end();
       iter++) {
    printf("BST debug comp stage %llu size %llu start_cycle %llu !!!\n",
           m_gpu->get_global_memory()->get_page_num((*iter)->start_addr),
           (*iter)->size, (*iter)->start_cycle);
  }
  if (!comp_latency_queue.empty()) {
    for (auto current_check_comp_latency = comp_latency_queue.begin();
         current_check_comp_latency != comp_latency_queue.end();) {
      printf("BST debug comp latency %llu page_num %llu start_cycle %llu ready_cycle %llu !!!\n", (*current_check_comp_latency)->start_addr, m_gpu->get_global_memory()->get_page_num((*current_check_comp_latency)->start_addr), (*current_check_comp_latency)->start_cycle, (*current_check_comp_latency)->ready_cycle);
      current_check_comp_latency++;
    }
  }
  for (auto iter = pcie_read_stage_queue.begin();
       iter != pcie_read_stage_queue.end(); iter++) {
    printf("BST debug PCIE READ stage %llu start_cycle %llu size %llu\n",
           (*iter)->start_addr, (*iter)->start_cycle, (*iter)->size);
  }
  if (pcie_read_latency_queue != NULL) {
    printf("BST debug PCIE read latency %llu page_num %llu start_cycle %llu ready_cycle %llu\n", pcie_read_latency_queue->start_addr, m_gpu->get_global_memory()->get_page_num(pcie_read_latency_queue->start_addr), pcie_read_latency_queue->start_cycle, pcie_read_latency_queue->ready_cycle);
  }
  for (auto iter = decomp_stage_queue.begin(); iter != decomp_stage_queue.end();
       iter++) {
    printf("BST debug decomp stage %llu page_num %llu start_cycle %llu\n", (*iter)->start_addr, m_gpu->get_global_memory()->get_page_num((*iter)->start_addr), (*iter)->start_cycle);
  }
  if(decomp_latency_queue.size() > 0){
    for(auto iter = decomp_latency_queue.begin(); iter != decomp_latency_queue.end(); iter++){
      printf("BST debug decomp latency %llu page_num %llu start_cycle %llu ready_cycle %llu\n", (*iter)->start_addr, m_gpu->get_global_memory()->get_page_num((*iter)->start_addr), (*iter)->start_cycle, (*iter)->ready_cycle);
    }
  }  
  for (std::list<page_table_walk_latency_t>::iterator iter =
           page_table_walk_queue.begin();
       iter != page_table_walk_queue.end(); iter++) {
    printf("BST debug page table walk %llu page_num %llu ready_cycle %llu !!!\n", iter->mf->get_addr(), m_gpu->get_global_memory()->get_page_num(iter->mf->get_addr()), iter->ready_cycle);
  }
  int valid_page_num=0;
  for (std::list<eviction_t *>::iterator bst_iter = valid_pages.begin();
       bst_iter != valid_pages.end(); bst_iter++) {
    valid_page_num+=(*bst_iter)->size/4096;
    is_block_evictable((*bst_iter)->addr, (*bst_iter)->size, true);
    mem_addr_t bst_pg_num =
        m_gpu->get_global_memory()->get_page_num((*bst_iter)->addr);
    printf("bst debug assert valid debug %llu page_num %llu page_size=%lu cycle=%llu\n",
           (*bst_iter)->addr, bst_pg_num, (unsigned long)(*bst_iter)->size, (*bst_iter)->cycle);
    printf("large page check %d\n", is_block_evictable((*bst_iter)->addr, (*bst_iter)->size, true));
  }
  printf("bst debug valid page num %d\n", valid_page_num);
  size_t decompressing_page_size=0;
  size_t decompressing_page_comp_size=0;
  for(std::list<eviction_t *>::iterator bst_iter = decompressing_pages.begin();
      bst_iter != decompressing_pages.end(); bst_iter++){
    decompressing_page_size+=(*bst_iter)->size;
    decompressing_page_comp_size+=(*bst_iter)->comp_size;
    mem_addr_t bst_pg_num =
        m_gpu->get_global_memory()->get_page_num((*bst_iter)->addr);
    printf("bst debug decompressing page num %llu compressed = %d valid = %d compressed_size = %lu cycle = %llu\n", (*bst_iter)->addr, m_gpu->get_global_memory()->is_iter_compressed(bst_pg_num), m_gpu->get_global_memory()->is_valid(bst_pg_num), (unsigned long)(*bst_iter)->comp_size, (*bst_iter)->cycle);
  }
  printf("bst debug decompressing page size %llu comp size %llu\n", decompressing_page_size, decompressing_page_comp_size);
  size_t compressed_page_size=0;
  size_t compressed_page_comp_size=0;
  for (std::list<eviction_t *>::iterator bst_iter = compressed_pages.begin();
       bst_iter != compressed_pages.end(); bst_iter++) {
        compressed_page_size+=(*bst_iter)->size;
        compressed_page_comp_size+=(*bst_iter)->comp_size;
    mem_addr_t bst_pg_num =
        m_gpu->get_global_memory()->get_page_num((*bst_iter)->addr);
    printf("bst debug assert comp debug %llu page_num %llu comp_size=%llu cycle = %llu\n",
           (*bst_iter)->addr, bst_pg_num, (*bst_iter)->comp_size, (*bst_iter)->cycle);
    is_block_comp_evict((*bst_iter)->addr, (*bst_iter)->size,false);
  }
  printf("bst debug compressed page size %llu comp size %llu\n", compressed_page_size, compressed_page_comp_size);
  size_t compressing_page_size=0;
  size_t compressing_page_comp_size=0;
  for(std::list<eviction_t *>::iterator bst_iter = compressing_pages.begin();
      bst_iter != compressing_pages.end(); bst_iter++){
    compressing_page_size+=(*bst_iter)->size;
    compressing_page_comp_size+=(*bst_iter)->comp_size;
    mem_addr_t bst_pg_num =
        m_gpu->get_global_memory()->get_page_num((*bst_iter)->addr);
    printf("bst debug compressing page num %llu compressed = %d valid = %d compressed_size = %llu cycle = %llu\n", (*bst_iter)->addr, m_gpu->get_global_memory()->is_iter_compressed(bst_pg_num), m_gpu->get_global_memory()->is_valid(bst_pg_num), (*bst_iter)->comp_size, (*bst_iter)->cycle);
  }
  printf("bst debug compressing page size %llu comp size %llu\n", compressing_page_size, compressing_page_comp_size);
  size_t comp_evict_page_size=0;
  size_t comp_evict_page_comp_size=0;
  for (std::list<eviction_t *>::iterator iter = comp_evict_pages.begin();
       iter != comp_evict_pages.end(); iter++) {
    comp_evict_page_size+=(*iter)->size;
    comp_evict_page_comp_size+=(*iter)->comp_size;
    mem_addr_t bst_pg_num =
        m_gpu->get_global_memory()->get_page_num((*iter)->addr);
    printf("bst debug comp_evict_pages page_num %llu\n", (*iter)->addr);
  }
  printf("bst debug comp_evict_page size %llu comp size %llu\n", comp_evict_page_size, comp_evict_page_comp_size);

  fflush(stdout);
  if (asser) {
    sleep(100);
    assert(0);
  }
}
eviction_t *gmmu_t::two_level_LRU_find_candidate(int type, bool comp_only) {//in current version type 1 is not used only type 0 is used
  eviction_t *LRU_leaf = nullptr;
  if (large_page_info.empty()) return nullptr;

  size_t n = large_page_info.size();
  size_t checked = 0;
  auto iter = large_page_info.begin();
  
  // Clock_LRU_index가 범위를 벗어나지 않도록 보장
  Clock_LRU_index = Clock_LRU_index % n;
  std::advance(iter, Clock_LRU_index);

  // 전체를 한 바퀴 돌 때까지 반복
  while (checked < n) {
    // iterator가 유효한지 확인
    if (iter == large_page_info.end()) {
      iter = large_page_info.begin();
      continue;
    }
    
    // iterator가 가리키는 포인터가 유효한지 확인
    if (*iter == nullptr) {
      printf("bst debug two_level_LRU_find_candidate iter is nullptr\n");
      print_debug_all(true);
      assert(0);
    }
    
    if ((*iter)->accessed) {
      (*iter)->accessed = false;
      // 다음 인덱스로 이동
      ++checked;
      ++iter;
    } else {
      // accessed가 0인 노드를 찾음
      if(comp_only){
        if(!(*iter)->inefficient_compression){
          LRU_leaf = get_LRU_leaf_node(*iter,0,comp_only);
          if(LRU_leaf!=nullptr){
            Clock_LRU_index = (std::distance(large_page_info.begin(), iter) + 1) % n;
            return LRU_leaf;
          }
        }
      }
      else{
        LRU_leaf = get_LRU_leaf_node(*iter,0,comp_only);
        if(LRU_leaf!=nullptr){
          Clock_LRU_index = (std::distance(large_page_info.begin(), iter) + 1) % n;
          return LRU_leaf;
        }
      }
      // Clock_LRU_index를 다음 위치로 업데이트
      ++iter;
      ++checked;
    }
  }
  
  // 모든 노드가 accessed=true였던 경우, Clock_LRU_index를 다음 위치로 업데이트
  if (LRU_leaf == nullptr) {
    Clock_LRU_index = (Clock_LRU_index + 1) % n;
  }
  
  return LRU_leaf;
}
eviction_t *gmmu_t::get_LRU_leaf_node(lp_tree_node *root, int type,bool comp_only) {
    // root에 속하는 valid_pages 또는 comp_list에서 is_block_evictable이 가능한 가장 오래된(LRU) 원소를 반환
    if (type == 0) { // valid_pages에서 LRU
        for (auto it = valid_pages.begin(); it != valid_pages.end(); ++it) {
            mem_addr_t page_addr = (*it)->addr;
            mem_addr_t eviction_bb_addr;
            unsigned int eviction_size;
            
            // page_compression_procedure와 동일한 로직 적용
            if ((*it)->size < root->leaf_size) {
                // leaf 사이즈보다 작은경우 기존에거가 더 이상 안쓰이는거라 처리
                eviction_bb_addr = (*it)->addr;
                eviction_size = (*it)->size;
            } else {
                eviction_bb_addr = get_basic_block(root, page_addr);
                eviction_size = root->leaf_size;
            }
            // comp_only가 true인 경우, 압축 불가능한 페이지는 건너뜀
            if(comp_only && get_lp_node(page_addr)->inefficient_compression){
              continue; // 이 페이지는 건너뛰고 다음 페이지 확인
            }
            // root의 범위 내에 있고, is_block_evictable이 가능한지 확인
            if (page_addr >= root->addr && page_addr < root->addr + root->size) {
                if (is_block_evictable(eviction_bb_addr, eviction_size, false)) {
                    return *it;
                }
            }
        }
    } else if (type == 1) { // comp_list에서 LRU
        for (auto it = compressed_pages.begin(); it != compressed_pages.end(); ++it) {
            mem_addr_t page_addr = (*it)->addr;
            mem_addr_t eviction_bb_addr;
            unsigned int eviction_size;
            
            // page_compression_procedure와 동일한 로직 적용 
            if ((*it)->size < root->leaf_size) {
                // leaf 사이즈보다 작은경우 기존에거가 더 이상 안쓰이는거라 처리
                eviction_bb_addr = (*it)->addr;
                eviction_size = (*it)->size;
            } else {
                eviction_bb_addr = get_basic_block(root, page_addr);
                eviction_size = root->leaf_size;
            }
            
            if (page_addr >= root->addr && page_addr < root->addr + root->size) {
                if (is_block_comp_evict(eviction_bb_addr, eviction_size, false)) {
                    return *it;
                }
            }
        }
    }
    return nullptr;
}
bool gmmu_t::page_compression_procedure(//if return true, compression or eviction is success
    bool evict_enabled = false) {  // we only process 64KB compression
    oversubscription_on=true;
    if(evict_enabled){
      if(!is_comp_enabled()&& !is_evict_enabled()){
        return false;
      }
    }
    else{
      if(!is_comp_enabled()){
        return false;
      }
    }
    /*if(m_config.real_WB_delay>10000){//if real_WB_delay is on, only 2time page eviction is allowed
      if(cur_cycle-last_page_eviction_cycle<m_config.real_WB_delay){
        return false;
      }
      last_page_eviction_cycle=cur_cycle;
    }*/

  std::list<std::pair<mem_addr_t, size_t>> pages_to_compress;
  std::vector<mem_addr_t> all_basic_blocks;
  sort_valid_pages();
  bool scheduling_compression = false;
  bool compressible = true;
  float sampled_comp_ratio = 0;

  if(m_config.two_level_LRU){
    eviction_t *LRU_leaf = two_level_LRU_find_candidate(0,!evict_enabled);
    if(LRU_leaf!=nullptr){
      mem_addr_t two_level_LRU_page_addr;
      mem_addr_t two_level_LRU_page_size;
      // LRU_leaf의 노드 정보를 한 번만 가져와서 재사용
      struct lp_tree_node *lru_node = get_lp_node(LRU_leaf->addr);
      if(lru_node == nullptr) {
        // 노드를 찾을 수 없는 경우 안전하게 처리
        printf("bst debug two_level_LRU_find_candidate lru_node is nullptr\n");
        print_debug_all(true);
        return false;
      }
      
      if(LRU_leaf->size < lru_node->leaf_size){
        two_level_LRU_page_addr = LRU_leaf->addr;
        two_level_LRU_page_size = LRU_leaf->size;
      }
      else{
        two_level_LRU_page_addr = get_basic_block(lru_node, LRU_leaf->addr);
        two_level_LRU_page_size = lru_node->leaf_size;
      }

      // 원래 로직과 일치하도록 수정: inefficient_compression이 false일 때 압축, true일 때 제거
      if(!lru_node->inefficient_compression){//압축 가능
        update_basic_block_comp(lru_node,two_level_LRU_page_addr,two_level_LRU_page_size,false);
        if(two_level_LRU_page_size==PREFETCH_16K_SIZE){
          all_basic_blocks.push_back(two_level_LRU_page_addr);
        }
        else{
          for(int i=0;i<two_level_LRU_page_size/PREFETCH_16K_SIZE;i++){
              all_basic_blocks.push_back(two_level_LRU_page_addr+i*PREFETCH_16K_SIZE);
          }
        }
        if(m_config.enable_LRU_compression){
          traverse_and_remove_lp_tree_comp(lru_node, all_basic_blocks);
        }
        sampled_comp_ratio = 1;
        compressible = true;
      }
      else if(evict_enabled){//제거 가능한 경우에만 제거 수행
        update_basic_block(lru_node,two_level_LRU_page_addr,two_level_LRU_page_size,false);
        if(two_level_LRU_page_size==PREFETCH_16K_SIZE){
          all_basic_blocks.push_back(two_level_LRU_page_addr);
        }
        else{
          for(int i=0;i<two_level_LRU_page_size/PREFETCH_16K_SIZE;i++){
            all_basic_blocks.push_back(two_level_LRU_page_addr+i*PREFETCH_16K_SIZE);
          }
        }
        traverse_and_remove_lp_tree(lru_node, all_basic_blocks);
        compressible = false;
      }
      else{
        // 압축도 제거도 불가능한 경우
        skip_compression_selection = true;
      }
    }
    else{
      skip_compression_selection = true;
    }
  }else{
  // 압축 대상 페이지 선택
  int reserve_start =
      (int)(valid_pages.size() *
            (1 - m_config.reserve_accessed_page_percent / 100.0f));
  std::list<eviction_t *>::iterator compression_end = valid_pages.begin();
  std::advance(compression_end, reserve_start);
  // 압축 조건: 압축 정책이 활성화되어 있고, 압축 비율이 충분히 높은 페이지
  std::list<eviction_t *>::iterator iter = valid_pages.begin();
  // 압축 가능한 페이지 찾기
  mem_addr_t page_addr, bb_addr;
  struct lp_tree_node *root = nullptr;
  while (iter != compression_end) {
    page_addr = (*iter)->addr;
    root = get_lp_node(page_addr);//각 노드에 맞는 루트찾기
    mem_addr_t eviction_bb_addr;
    unsigned int eviction_size;
    if((*iter)->size < root->leaf_size){//leaf 사이즈보다 작은경우 기존에거가 더 이상 안쓰이는거라 처리
      eviction_bb_addr = (*iter)->addr;
      eviction_size = (*iter)->size;
    }
    else{
      eviction_bb_addr = get_basic_block(root, page_addr);
      eviction_size = root->leaf_size;
    }
    // 선택적 압축이 켜져 있고 압축 비율이 2 이상인 경우 또는 항상 압축하는 경우
    if (!selective_compression || root->inefficient_compression == false) {//선택압축이고 comp_size가 충분히 작고, inefficient_compression이 아니면
      // ommit this|| get_tree_comp_size(page_addr) 
      //<= MIN_PREFETCH_SIZE/m_config.selective_compression_threshold)
        if (is_block_evictable(eviction_bb_addr, eviction_size,false)) {  // find real valid page
          update_basic_block_comp(root, eviction_bb_addr, eviction_size, false);
          if(eviction_size==PREFETCH_16K_SIZE){
            all_basic_blocks.push_back(page_addr);
          }
          else{
            for(int i=0;i<eviction_size/PREFETCH_16K_SIZE;i++){
              if(is_block_evictable(page_addr+i*PREFETCH_16K_SIZE,PREFETCH_16K_SIZE)){
                all_basic_blocks.push_back(page_addr+i*PREFETCH_16K_SIZE);
              }
            }
          }
          if(m_config.enable_LRU_compression){
            traverse_and_remove_lp_tree_comp(root, all_basic_blocks);
          }          
          sampled_comp_ratio = 1;
          break;
        }
    } else if (evict_enabled && is_block_evictable(eviction_bb_addr, eviction_size,false)) {
        update_basic_block(root, eviction_bb_addr, eviction_size, false);
        if(eviction_size==PREFETCH_16K_SIZE){
          all_basic_blocks.push_back(page_addr);
        }
        else{
          for(int i=0;i<eviction_size/PREFETCH_16K_SIZE;i++){
            if(is_block_evictable(page_addr+i*PREFETCH_16K_SIZE,PREFETCH_16K_SIZE)){
              all_basic_blocks.push_back(page_addr+i*PREFETCH_16K_SIZE);
            }
          }
        }
        traverse_and_remove_lp_tree(root, all_basic_blocks);
        compressible = false;
        break;
    }

    iter++;
  }
  
  if(iter==compression_end){
    skip_compression_selection = true;
  }
}
  // 중복된 basic block 제거 (앞에 있는 것만 남기고 뒤의 중복은 삭제)
  std::vector<mem_addr_t> unique_basic_blocks;
  for (const auto& block : all_basic_blocks) {
    if (std::find(unique_basic_blocks.begin(), unique_basic_blocks.end(), block) == unique_basic_blocks.end()) {
      unique_basic_blocks.push_back(block);
    }
  }
  all_basic_blocks = unique_basic_blocks;
  
  // all_basic_blocks가 비어있지 않을 때만 처리
  if (!all_basic_blocks.empty()) {
    std::vector<mem_addr_t>::iterator bb = all_basic_blocks.begin();
    // 64K까지 연속적인 블록을 하나의 page_to_compress로 묶고, 연속적이지 않으면 해당 수만큼 16K 단위로 계산
    while (bb != all_basic_blocks.end()) {
      std::vector<mem_addr_t>::iterator next_bb = bb;
      size_t cur_num = 0;
      
      if (compressible) {
        // 압축 가능한 경우: 64K(=4*16K)까지 연속된 블록을 묶음
        do {
          next_bb++;
          cur_num++;
        } while (next_bb != all_basic_blocks.end() &&
                 cur_num < 4 &&
                 *next_bb / MIN_PREFETCH_SIZE == *bb / MIN_PREFETCH_SIZE &&
                 ((*next_bb) == ((*bb) + cur_num * PREFETCH_16K_SIZE)));
      } else {
        // 압축 불가능한 경우: leaf_size 단위로 연속된 블록을 묶음
        // 첫 번째 basic block에서 root를 가져옴
        struct lp_tree_node *current_root = get_lp_node(*bb);
        size_t max_chunks = current_root->leaf_size / PREFETCH_16K_SIZE;
        do {
          next_bb++;
          cur_num++;
        } while (next_bb != all_basic_blocks.end() &&
                 cur_num < max_chunks &&
                 ((*next_bb) == ((*bb) + cur_num * PREFETCH_16K_SIZE)));
      }
      
      // 연속된 블록을 하나의 page_to_compress로 추가
      struct lp_tree_node *bb_root = get_lp_node(*bb);
      pages_to_compress.push_back(
          std::make_pair((*bb), cur_num * PREFETCH_16K_SIZE));
      bb = next_bb;
    }
  }
  // 선택된 페이지 압축 처리
  for (const auto& iter : pages_to_compress) {
    scheduling_compression = true;
    std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
    mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(iter.first);
    p_t->start_addr = iter.first;
    size_t total_comp_size = 0;
    // iter.second / MIN_PREFETCH_SIZE가 0인 경우(MIN_PREFETCH_SIZE보다 작은 경우) 처리 추가
    if (iter.second / MIN_PREFETCH_SIZE == 0) {
      size_t cur_compressed_size = 0;
      if (get_tree_comp_size(iter.first) == 0||(sampled_comp_ratio > 0 && !is_sampled(iter.first))) {
        for (int page_index = 0; page_index < 16; page_index++) {
          CACHELINE_DATA line[32];
          mem_addr_t cur_page_addr = (iter.first / MIN_PREFETCH_SIZE) *
                                         MIN_PREFETCH_SIZE + page_index * m_config.page_size;
          mem_addr_t cur_page_num = m_gpu->get_global_memory()->get_page_num(cur_page_addr);
          m_gpu->get_global_memory()->read_single_page(cur_page_num, line);
          size_t cur_comp_size = CUREMEM_compress(compressor, line);
          cur_compressed_size += cur_comp_size;
        }
        // 기본값으로 set_tree_comp_size 호출
        set_tree_comp_size(iter.first, ((cur_compressed_size + 4095) / 4096) * 4096);
        set_sampled(iter.first);
      }
      // get_tree_comp_size에서는 MIN_PREFETCH_SIZE/iter.second로 반환하도록 가정
      // (get_tree_comp_size 함수 내부에서 처리 필요)
      total_comp_size += get_tree_comp_size(iter.first)*iter.second/MIN_PREFETCH_SIZE;//MIN_PREFETCH_SIZE로 나온 압축크기를 iter.size에 맞게 변환
    } else {
      for (int chunk_index = 0; chunk_index < iter.second / MIN_PREFETCH_SIZE; chunk_index++) {
        size_t cur_compressed_size = 0;
        if (get_tree_comp_size(iter.first+chunk_index*MIN_PREFETCH_SIZE) == 0 || (sampled_comp_ratio > 0 && !is_sampled(iter.first+chunk_index*MIN_PREFETCH_SIZE))) {
          for (int page_index = 0; page_index < 16; page_index++) {
            CACHELINE_DATA line[32];
            mem_addr_t cur_page_addr = iter.first + chunk_index*MIN_PREFETCH_SIZE + page_index * m_config.page_size;
            mem_addr_t cur_page_num = m_gpu->get_global_memory()->get_page_num(cur_page_addr);
            m_gpu->get_global_memory()->read_single_page(cur_page_num, line);
            size_t cur_comp_size = CUREMEM_compress(compressor, line);
            cur_compressed_size += cur_comp_size;
          }
          set_tree_comp_size(iter.first+chunk_index*MIN_PREFETCH_SIZE, ((cur_compressed_size + 4095) / 4096) * 4096);
          set_sampled(iter.first+chunk_index*MIN_PREFETCH_SIZE);
        }
        total_comp_size += get_tree_comp_size(iter.first + chunk_index * MIN_PREFETCH_SIZE);
      }
    }
    p_t->comp_size = total_comp_size;
    p_t->comp_band = m_config.comp_band_scalar;
    p_t->decomp_band = m_config.decomp_band_scalar;
    p_t->size = iter.second;
    if(iter.second<=p_t->comp_size*m_config.selective_compression_threshold){//compression_ratio가 낮은경우 압축 효율이 떨어질 수 있음(압축효율비율*comp_size가 원래 크기보다 큰경우는 압축x)
      lp_tree_node *root_for_inefficient = get_lp_node(iter.first);
      root_for_inefficient->leaf_will_be_inefficient = true;
    }
    // MICRO mode: valid_pages entries are 64KB (refresh_valid_pages), so the selection
    // above already picks 64KB blocks; make the LRU-list granularity match so
    // valid_pages_compressing / valid_pages_erase move exactly this 64KB.
    p_t->leaf_size = micro_leaf(get_lp_node(iter.first));
    p_t->start_cycle = cur_cycle;
    if (compressible) {
      p_t->type = latency_type::COMP;
    } else {
      p_t->type = latency_type::PCIE_WRITE_BACK;
      if (m_config.invalidate_clean) {
        for (std::list<eviction_t *>::iterator it = valid_pages.begin();
             it != valid_pages.end(); it++) {
          if ((*it)->addr <= iter.first &&
              iter.first < (*it)->addr + (*it)->size) {
            if ((*it)->RW == 1) {  // if read
              p_t->type = latency_type::INVALIDATE;
              break;
            }
          }
        }
      }
    }
      mem_addr_t chunk_page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);
      for (int i = 0; i < (int)(iter.second / m_config.page_size); i++) {
        p_t->page_list.push_back(chunk_page_num + i);
        assert(m_gpu->get_global_memory()->is_valid(chunk_page_num + i));
        if (compressible) {
          valid_pages_compressing(chunk_page_num + i, get_tree_comp_size(iter.first + i * m_config.page_size), p_t->leaf_size);
        } else {
          valid_pages_erase(chunk_page_num + i, p_t->leaf_size);
        }
      }

    // 압축 큐에 추가
    if (compressible) {
      p_t->ready_cycle = cur_cycle + m_config.real_WB_delay;
      reserve_crm_handling(p_t);
      comp_stage_queue.push_back(p_t);
      num_comp_stage_queue += (p_t->size-p_t->comp_size)/m_config.page_size;
    } else {
      reserve_crm_handling(p_t);
      pcie_write_stage_queue.push_back(p_t);
      num_write_stage_queue += p_t->page_list.size();
    }
  }
  if(evict_enabled){//if evict is enabled, we need to check compression and evict
    if(compressible){//if compression is enabled we dont need to check evict
      enable_comp();
      enable_evict();
    }else{//not compressible, we need to check evict
      disable_comp();//disable compression
      if(scheduling_compression){//if we are scheduling compression, we need to check evict
        enable_evict();
      }else{//not scheduling compression, we need to disable evict
        disable_evict();
      }
    }
  }
  else{//we only check compression
    if(scheduling_compression){
      enable_comp();
      enable_evict();
    }else{
      disable_comp();
    }
  }
  return scheduling_compression;
}
// comp eviciton 프로시저 - 압축된 페이지 제거
bool gmmu_t::page_comp_eviction_procedure() {
  bool comp_evict_schedule = false;
  if(!is_comp_evict_enabled()){//if comp_evict is not enabled, we need to return false
    return false;
  }
  /*if(m_config.real_WB_delay>10000){//if real_WB_delay is on, only 2time page eviction is allowed
    if(cur_cycle-last_page_eviction_cycle<m_config.real_WB_delay){
      return false;
    }
    last_page_eviction_cycle=cur_cycle;
  }*/
  std::list<std::pair<mem_addr_t, size_t>> comp_evicted_pages;
  std::list<eviction_t *>::iterator iter = compressed_pages.begin();
  std::vector<mem_addr_t> all_basic_blocks;
  struct lp_tree_node *root;

  mem_addr_t page_addr;
  mem_addr_t eviction_bb_addr;
  unsigned int eviction_size;
  for (; iter != compressed_pages.end(); iter++) {
    page_addr = (*iter)->addr;
    root = get_lp_node(page_addr);
    // MICRO 모드: eviction 단위는 2MB. compressed_pages 항목은 64KB 이지만, 해당 64KB 가
    // 속한 2MB leaf 전체를 대상으로 잡아 그 안의 모든 압축 64KB 블록을 한 번에 evict 한다.
    if(!(m_config.MICRO_barrier_Range && m_config.tree_logic == 2) &&
       (*iter)->size < root->leaf_size){
      eviction_bb_addr = (*iter)->addr;
      eviction_size = (*iter)->size;
    }
    else{
      eviction_bb_addr = get_basic_block(root, page_addr);
      eviction_size = root->leaf_size;
    }
    if (is_block_comp_evict(eviction_bb_addr, eviction_size,false)) {
      update_basic_block_comp_evict(root, eviction_bb_addr, eviction_size, false);
      if(eviction_size==PREFETCH_16K_SIZE){
        all_basic_blocks.push_back(eviction_bb_addr);
      }
      else{
        for(int i=0;i<eviction_size/PREFETCH_16K_SIZE;i++){
          if(is_block_comp_evict(eviction_bb_addr+i*PREFETCH_16K_SIZE,PREFETCH_16K_SIZE)){
            all_basic_blocks.push_back(eviction_bb_addr+i*PREFETCH_16K_SIZE);
          }
        }
      }
      comp_evict_schedule = true;
      //traverse_and_remove_lp_tree_comp_evict(root, all_basic_blocks);
      break;
    }
  }
  // 연속된 블록 그룹화 (압축 제거)
  std::vector<mem_addr_t>::iterator bb = all_basic_blocks.begin();
  // 최대 64K 단위(PREFETCH_16K_SIZE*4)로 그룹화하여 comp_evicted_pages에 추가
  while (bb != all_basic_blocks.end()) {
    std::vector<mem_addr_t>::iterator next_bb = bb;
    size_t cur_num = 0;
    // 연속된 16K 블록을 찾음
    do {
      next_bb++;
      cur_num++;
    } while (next_bb != all_basic_blocks.end() &&
             ((*next_bb) == ((*bb) + cur_num * PREFETCH_16K_SIZE)));

    // 연속된 영역 전체를 한 번에 추가
    size_t total_size = cur_num * PREFETCH_16K_SIZE;
    comp_evicted_pages.push_back(std::make_pair(*bb, total_size));
    bb = next_bb;
  }
  // 압축된 페이지 제거 처리
  for (const auto& iter : comp_evicted_pages) {
    std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();

    p_t->start_addr = iter.first;
    size_t total_comp_size = 0;

    // iter.second가 MIN_PREFETCH_SIZE보다 크지만, 나머지가 있을 수 있음
    if (iter.second < MIN_PREFETCH_SIZE) {
      // leaf 하나가 prefetch보다 작음 → 압축 크기를 나눠야 함
      int num_subleaves = MIN_PREFETCH_SIZE / iter.second;
      total_comp_size = get_tree_comp_size(iter.first) / num_subleaves;
    } else {
      int num_leaves = iter.second / MIN_PREFETCH_SIZE;
      size_t remain_size = iter.second % MIN_PREFETCH_SIZE;
      for (int i = 0; i < num_leaves; ++i) {//64K블럭단위처리
        mem_addr_t leaf_addr = iter.first + i * MIN_PREFETCH_SIZE;
        total_comp_size += get_tree_comp_size(leaf_addr);
      }
      // 나머지 영역이 있을 경우, 해당 영역의 압축 크기도 더해줌
      if (remain_size > 0) {
        mem_addr_t remain_addr = iter.first + num_leaves * MIN_PREFETCH_SIZE;
        // MIN_PREFETCH_SIZE보다 작은 영역이므로, 비율로 압축 크기를 계산
        size_t full_comp_size = get_tree_comp_size(remain_addr);
        total_comp_size += (full_comp_size * remain_size) / MIN_PREFETCH_SIZE;
      }
    }
    p_t->comp_size = total_comp_size;
    p_t->comp_band = m_config.comp_band_scalar;
    p_t->decomp_band = m_config.decomp_band_scalar;
    // m_gpu->get_global_memory()->get_pages_comp_band(iter->first);
    p_t->size = iter.second;
    // span (size) stays at the 2MB-grouped value; leaf_size only drives the LRU-list
    // granularity (compress_pages_erase / comp_evict_erase), which is 64KB in MICRO mode.
    p_t->leaf_size = micro_leaf(get_lp_node(iter.first));
    p_t->start_cycle = cur_cycle;

    // 압축 제거 타입 결정
    latency_type ltype = latency_type::COMP_EVICT;
    if (m_config.invalidate_clean) {
      for (std::list<eviction_t *>::iterator it = compressed_pages.begin();
           it != compressed_pages.end(); it++) {
        if ((*it)->addr <= iter.first &&
            iter.first < (*it)->addr + (*it)->size) {
          if ((*it)->RW == 1) {  // if read
            ltype = latency_type::COMP_INV_EVICT;
            break;
          }
        }
      }
    }
    p_t->type = ltype;

    // 페이지 목록 처리
    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);
      p_t->page_list.push_back(page_num);
      compress_pages_erase(page_num, p_t->leaf_size);
    } else {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);
      for (int i = 0; i < (int)(iter.second / m_config.page_size); i++) {
        p_t->page_list.push_back(page_num + i);
        compress_pages_erase(page_num + i, p_t->leaf_size );
      }
    }

    // 쓰기 큐에 추가
    reserve_crm_handling(p_t);
    pcie_write_stage_queue.push_back(p_t);
    num_write_stage_queue += p_t->comp_size/4096;
    num_comp_evict_stage_queue += p_t->page_list.size();
  }
  if(comp_evict_schedule){//if comp_evict is scheduled, we need to enable comp_evict
    enable_comp_evict();
    return true;
  }else{//if comp_evict is not scheduled, we need to disable comp_evict
    disable_comp_evict();
    return false;
  }
}
// 페이지 제거 프로시저 - 제거 관련 로직 처리 (압축된 페이지 제거 x)
bool gmmu_t::page_eviction_procedure() {
  /*(m_config.real_WB_delay){
   if(cur_cycle-last_page_eviction_cycle<m_config.real_WB_delay){
    return false;
   }
   if(!m_config.proactive_WB){
    if(!pcie_write_stage_queue.empty()){//if not proactive WB, before page_eviction is not allowed
      return false;
    }
  }
   last_page_eviction_cycle=cur_cycle;
  }*/

  oversubscription_on=true;
  if(!is_evict_enabled()) return false;
  
  bool evict_schedule = false;
  sort_valid_pages();

  std::list<std::pair<mem_addr_t, size_t>> evicted_pages;
  const int eviction_start = valid_pages.size() * m_config.reserve_accessed_page_percent / 100;

  auto iter = valid_pages.begin();
  std::advance(iter, eviction_start);

  if(evict_policy == eviction_policy::LRU4K) {
    // LRU4K 정책 처리
    while(iter != valid_pages.end() && !is_block_evictable((*iter)->addr, (*iter)->size)) iter++;

    if(iter != valid_pages.end()) {
      const mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);
      evicted_pages.emplace_back(page_addr, m_config.page_size);
    }

  } else if(evict_policy == eviction_policy::LRU || evict_policy == eviction_policy::LFU || 
            m_config.page_size == MAX_PREFETCH_SIZE) {
    // LRU/LFU 정책 처리  
    while(iter != valid_pages.end() && !is_block_evictable((*iter)->addr, (*iter)->size)) iter++;

    if(iter != valid_pages.end()) {
      const mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      evict_whole_tree(root);
      evicted_pages.emplace_back(root->addr, root->size);
    } else {
      iter = compressed_pages.begin();
      while(iter != compressed_pages.end() && !is_block_comp_evict((*iter)->addr, (*iter)->size)) iter++;
    }

  } else if(evict_policy == eviction_policy::RANDOM) {
    // 랜덤 정책 처리
    const int random_offset = rand() % (int)(valid_pages.size() * (1 - m_config.reserve_accessed_page_percent/100));
    std::advance(iter, random_offset);

    while(iter != valid_pages.end() && !is_block_evictable((*iter)->addr, (*iter)->size)) iter++;

    if(iter != valid_pages.end()) {
      const mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, false);
      evicted_pages.emplace_back(page_addr, m_config.page_size);
    }

  } else if(evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    // Sequential Local 정책 처리
    for(; iter != valid_pages.end(); iter++) {
      const mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      const mem_addr_t bb_addr = get_basic_block(root, page_addr);
      
      if(is_block_evictable(bb_addr, root->leaf_size)) {
        update_basic_block(root, bb_addr, root->leaf_size, false);
        evicted_pages.emplace_back(bb_addr, root->leaf_size);
        break;
      }
    }

  } else if(evict_policy == eviction_policy::TBN) {
    // TBN 정책 처리
    if (m_config.two_level_LRU) {
      // two-level LRU를 사용하는 경우: large_page_info 기반으로 희생자 선택
      std::vector<mem_addr_t> all_basic_blocks;
      struct lp_tree_node *root = nullptr;
      unsigned int target_eviction_size = 0;
      mem_addr_t target_bb_addr = 0;

      // comp_only=false로 호출해서 압축 여부와 무관하게 희생자를 찾음
      eviction_t *LRU_leaf = two_level_LRU_find_candidate(0, /*comp_only=*/false);
      if (LRU_leaf) {
        root = get_lp_node(LRU_leaf->addr);
        if (root == nullptr) {
          printf("bst debug page_eviction_procedure two_level_LRU root is nullptr\n");
        } else {
          // tree_logic == 1 또는 fault_based_LRU인 경우 2MB 블록 체크
          if(m_config.tree_logic == 1 || m_config.fault_based_LRU) {
            target_eviction_size = root->leaf_size;
            target_bb_addr = get_basic_block(root, LRU_leaf->addr);
            // 2MB 블록 전체가 evictable한지 체크
            if(is_block_evictable(root->addr, root->size, false, true)) {
              update_basic_block(root, target_bb_addr, target_eviction_size, false);
              evict_schedule = true;
              const int block_count = target_eviction_size / PREFETCH_16K_SIZE;
              for(int i = 0; i < block_count; ++i) {
                all_basic_blocks.emplace_back(target_bb_addr + i * PREFETCH_16K_SIZE);
              }
              traverse_and_remove_lp_tree(root, all_basic_blocks);
            }
          } else {
            // 일반적인 경우: leaf 사이즈와 비교해서 실제 제거할 블록 크기/주소 결정
            if (LRU_leaf->size < root->leaf_size) {
              target_eviction_size = LRU_leaf->size;
              target_bb_addr = LRU_leaf->addr;
            } else {
              target_eviction_size = root->leaf_size;
              target_bb_addr = get_basic_block(root, LRU_leaf->addr);
            }

            // is_block_evictable 체크
            if(is_block_evictable(target_bb_addr, target_eviction_size, false, false)) {
              // basic block 단위로 모두 수집
              const int block_count = target_eviction_size / PREFETCH_16K_SIZE;
              for (int i = 0; i < block_count; ++i) {
                all_basic_blocks.emplace_back(target_bb_addr + i * PREFETCH_16K_SIZE);
              }

              // lp_tree 및 valid 리스트 갱신
              update_basic_block(root, target_bb_addr, target_eviction_size, false);
              evict_schedule = true;
              if(!(m_config.tree_logic == 3)) {
                traverse_and_remove_lp_tree(root, all_basic_blocks);
              }
            }
          }

          // evict_schedule이 true이고 all_basic_blocks가 비어있지 않은 경우에만 evicted_pages에 추가
          if(evict_schedule && !all_basic_blocks.empty()) {
            // 연속된 basic block을 하나의 eviction range로 합쳐서 evicted_pages에 기록
            auto bb = all_basic_blocks.begin();
            while (bb != all_basic_blocks.end()) {
              auto next_bb = bb;
              size_t cur_num = 0;
              do {
                next_bb++;
                cur_num++;
              } while (next_bb != all_basic_blocks.end() &&
                       (*next_bb) ==
                           ((*bb) + cur_num * PREFETCH_16K_SIZE));

              evicted_pages.emplace_back((*bb), cur_num * PREFETCH_16K_SIZE);
              bb = next_bb;
            }
          }
        }
      }
    } else {
      uint8_t forest_LRU_object = 0;
      if(m_config.forest_LRU){
        // 각 alloc_count별 최근 접근 인덱스를 저장하는 맵
        std::map<uint8_t, int> alloc_count_latest_index;
        
        // 1. 각 alloc_count 그룹별로 가장 높은 인덱스(최근 접근) 찾기
        int current_index = 0;
        for(auto temp_iter = iter; temp_iter != valid_pages.end(); temp_iter++, current_index++){
          struct lp_tree_node *current_node = get_lp_node((*temp_iter)->addr);
          uint8_t current_alloc_count = current_node->alloc_count;
          
          // 해당 alloc_count의 최근 접근 인덱스를 업데이트 (더 높은 인덱스로)
          alloc_count_latest_index[current_alloc_count] = current_index;
        }
        
        // 2. 모든 alloc_count 중에서 가장 낮은 최근 접근 인덱스를 가진 것 찾기
        int lowest_latest_index = INT_MAX;
        uint8_t best_alloc_count = 0;
        
        for(const auto& pair : alloc_count_latest_index){
          if(pair.second < lowest_latest_index){
            lowest_latest_index = pair.second;
            best_alloc_count = pair.first;
          }
        }
        
        // 3. 결과 반환
        forest_LRU_object = best_alloc_count;
      }
      std::vector<mem_addr_t> all_basic_blocks;
      struct lp_tree_node *root;
      unsigned int target_eviction_size;
      mem_addr_t target_bb_addr;
      for(; iter != valid_pages.end(); iter++) {
        if(m_config.forest_LRU){
          if(get_lp_node((*iter)->addr)->alloc_count != forest_LRU_object){//only process LRU object
            continue;
          }
        }
        if(m_config.tree_logic == 1||m_config.fault_based_LRU){//TBN 2MB eviction or fault based LRU
          root = get_lp_node((*iter)->addr);
          target_eviction_size = root->leaf_size;
          target_bb_addr = get_basic_block(root, (*iter)->addr);
          if(is_block_evictable(root->addr, root->size, false, true)){//check 2MB block
            update_basic_block(root, target_bb_addr, target_eviction_size, false);//update target block
            evict_schedule = true;
            const int block_count = target_eviction_size / PREFETCH_16K_SIZE;
            for(int i = 0; i < block_count; ++i) {
              all_basic_blocks.emplace_back(target_bb_addr + i * PREFETCH_16K_SIZE);
            }
            traverse_and_remove_lp_tree(root, all_basic_blocks);//remove target block from lp tree
            break;
          }
        }
        else{
          const mem_addr_t page_addr = (*iter)->addr;
        root = get_lp_node(page_addr);

        if((*iter)->size < root->leaf_size){//leaf사이즈보다 작은 경우
          target_eviction_size = (*iter)->size;
          target_bb_addr = (*iter)->addr;
        }
        else{//leaf사이즈보다 큰 경우 leaf사이즈만 업데이트
          target_eviction_size = root->leaf_size;
          target_bb_addr = get_basic_block(root, page_addr);
        }
        if(is_block_evictable(target_bb_addr, target_eviction_size, false, false)) {
          // target_eviction_size가 PREFETCH_16K_SIZE의 배수이므로, std::vector에 한 번에 reserve 및 emplace
          const int block_count = target_eviction_size / PREFETCH_16K_SIZE;
          for(int i = 0; i < block_count; ++i) {
              all_basic_blocks.emplace_back(target_bb_addr + i * PREFETCH_16K_SIZE);
          }
          update_basic_block(root, target_bb_addr, target_eviction_size, false);
          evict_schedule = true;
          if(!m_config.tree_logic==3){
          traverse_and_remove_lp_tree(root, all_basic_blocks);
          }
          break;
        }
      }
      }

      if(iter != valid_pages.end()) {
        auto bb = all_basic_blocks.begin();
        while(bb != all_basic_blocks.end()) {
          auto next_bb = bb;
          size_t cur_num = 0;
          do {
            next_bb++;
            cur_num++;
          } while(next_bb != all_basic_blocks.end() && (*next_bb) == ((*bb) + cur_num * PREFETCH_16K_SIZE));
          
          evicted_pages.emplace_back((*bb), cur_num * PREFETCH_16K_SIZE);
          bb = next_bb;
        }
      }
    }
  }

  // 제거된 페이지 처리
  if(m_config.tree_logic == 1&&!evicted_pages.empty()){
    struct lp_tree_node *evicted_root=get_lp_node(evicted_pages.begin()->first);
    size_t root_addr=evicted_root->addr;
    size_t root_size=evicted_root->size;
    std::list<mem_addr_t> cur_page_list;
    latency_type ltype = latency_type::PCIE_WRITE_BACK;
    for(const auto& [addr, size] : evicted_pages) {//연속된 주소로 구성된 PCIE_WRITE_BACK 타입 페이지 처리
      mem_addr_t cur_page_num_of_root=m_gpu->get_global_memory()->get_page_num(addr);
      int page_count=size/m_config.page_size;
      for(int i=0; i<page_count; i++){
        if(m_gpu->get_global_memory()->is_valid(cur_page_num_of_root+i)){
          cur_page_list.push_back(cur_page_num_of_root+i);
          valid_pages_erase(cur_page_num_of_root+i, evicted_root->leaf_size);
        }
      }
    }
      auto p_t = std::make_shared<pcie_latency_t>();
      p_t->start_addr = root_addr;
      p_t->page_list = cur_page_list;
      p_t->leaf_size = evicted_root->leaf_size;
      p_t->type = ltype;
      p_t->start_cycle = cur_cycle;
      p_t->size = root_size; // 루트 전체 사이즈 설정
      num_write_stage_queue += p_t->page_list.size();
      reserve_crm_handling(p_t);
      pcie_write_stage_queue.push_back(p_t);
    }
  else{
  for(const auto& [addr, size] : evicted_pages) {//연속된 주소로 구성된 PCIE_WRITE_BACK 타입 페이지 처리
    auto p_t = std::make_shared<pcie_latency_t>();
    p_t->start_addr = addr;
    p_t->leaf_size = get_lp_node(addr)->leaf_size;
    p_t->start_cycle = cur_cycle;

    // 제거 타입 결정
    latency_type ltype = latency_type::PCIE_WRITE_BACK;
    for(const auto& valid_page : valid_pages) {
      if(valid_page->addr <= addr && addr < valid_page->addr + valid_page->size && valid_page->RW == 1) {
        ltype = latency_type::INVALIDATE;
        break;
      }
    }
    p_t->type = ltype;

    // 페이지 목록 처리
    const mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);
    if(m_config.page_size == MAX_PREFETCH_SIZE) {
      p_t->page_list.push_back(page_num);
      valid_pages_erase(page_num, p_t->leaf_size);
      assert(0 && "bst debug page_eviction_procedure this will not happen");
    } else {
      const int page_count = size / m_config.page_size;
      for(int i = 0; i < page_count; i++) {
        if(m_gpu->get_global_memory()->is_valid(page_num + i)) {
          p_t->page_list.push_back(page_num + i);
          valid_pages_erase(page_num + i, p_t->leaf_size);
        }
      }
    }

    p_t->size = p_t->page_list.size() * m_config.page_size;
    num_write_stage_queue += p_t->page_list.size();
    if(p_t->size){
      reserve_crm_handling(p_t);
      pcie_write_stage_queue.push_back(p_t);
    }
  }
}

  if(evict_schedule) {
    enable_evict();
    return true;
  }
  disable_evict();
  return false;
}
void gmmu_t::old_page_eviction_procedure() {
  sort_valid_pages();
  comp_evict = false;
  std::list<std::pair<mem_addr_t, size_t>> evicted_pages;
  std::list<std::pair<mem_addr_t, size_t>> comp_evicted_pages;

  int eviction_start =
      (int)(valid_pages.size() * m_config.reserve_accessed_page_percent / 100);

  if (evict_policy == eviction_policy::LRU4K) {
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr,
                               (*iter)->size)) {  // find valid & raw
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      if (compression_policy != compression_policy_struct::NONE) {
        update_basic_block_comp(root, page_addr, m_config.page_size, false);
      } else {
        update_basic_block(root, page_addr, m_config.page_size, false);
      }
      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    } else {  // evict compressed
      iter = compressed_pages.begin();
      while (iter != compressed_pages.end() &&
             !is_block_comp_evict(
                 (*iter)->addr,
                 (*iter)->size)) {  // find compressed & not write stage
        iter++;
      }
      if (iter != compressed_pages.end()) {
        mem_addr_t page_addr = (*iter)->addr;
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block_comp_evict(root, page_addr, m_config.page_size,
                                      false);
        comp_evicted_pages.push_back(
            std::make_pair(page_addr, m_config.page_size));
      }
    }
  } else if (evict_policy == eviction_policy::LRU ||
             evict_policy == eviction_policy::LFU ||
             m_config.page_size ==
                 MAX_PREFETCH_SIZE) {  // bst TODO : need to be implemnet for
                                       // comp
    // in lru, only evict the least recently used pages at the front of accessed
    // pages queue in lfu, only evict the page accessed least number of times
    // from the front of accessed pages queue
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      evict_whole_tree(root);

      evicted_pages.push_back(std::make_pair(root->addr, root->size));
    } else {  // evict compressed
      iter = compressed_pages.begin();
      while (iter != compressed_pages.end() &&
             !is_block_comp_evict(
                 (*iter)->addr,
                 (*iter)->size)) {  // find compressed & not write stage
        iter++;
      }
      if (iter != compressed_pages.end()) {
        mem_addr_t page_addr = (*iter)->addr;
        struct lp_tree_node *root = get_lp_node(page_addr);
        evict_whole_tree_comp_evict(root);
        comp_evicted_pages.push_back(std::make_pair(root->addr, root->size));
      }
    }
  } else if (evict_policy == eviction_policy::RANDOM) {
    // in random eviction, select a random page
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(
        iter, eviction_start +
                  (rand() %
                   (int)(valid_pages.size() *
                         (1 - m_config.reserve_accessed_page_percent / 100))));

    while (iter != valid_pages.end() &&
           !is_block_evictable((*iter)->addr, (*iter)->size)) {
      iter++;
    }

    if (iter != valid_pages.end()) {
      mem_addr_t page_addr = (*iter)->addr;
      struct lp_tree_node *root = get_lp_node(page_addr);
      if (compression_policy != compression_policy_struct::NONE) {
        update_basic_block_comp(root, page_addr, m_config.page_size, false);
      } else {
        update_basic_block(root, page_addr, m_config.page_size, false);
      }
      evicted_pages.push_back(std::make_pair(page_addr, m_config.page_size));
    } else {  // evict compressed
      iter = compressed_pages.begin();
      std::advance(iter, rand() % (int)(compressed_pages.size()));
      while (iter != compressed_pages.end() &&
             !is_block_comp_evict(
                 (*iter)->addr,
                 (*iter)->size)) {  // find compressed & not write stage
        iter++;
      }
      if (iter != compressed_pages.end()) {
        mem_addr_t page_addr = (*iter)->addr;
        struct lp_tree_node *root = get_lp_node(page_addr);
        update_basic_block_comp_evict(root, page_addr, m_config.page_size,
                                      false);
        comp_evicted_pages.push_back(
            std::make_pair(page_addr, m_config.page_size));
      }
    }
  } else if (evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    // we evict sixteen 4KB pages in the 2 MB allocation where this evictable
    // belong to
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;
      root = get_lp_node(page_addr);
      bb_addr = get_basic_block(root, page_addr);
      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        if (compression_policy != compression_policy_struct::PARTIAL) {
          update_basic_block_comp(root, page_addr, MIN_PREFETCH_SIZE, false);
        } else {
          update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        }
        break;
      }
    }

    if (iter != valid_pages.end()) {
      evicted_pages.push_back(std::make_pair(bb_addr, MIN_PREFETCH_SIZE));
    } else {  // evict compressed
      iter = compressed_pages.begin();
      for (; iter != compressed_pages.end(); iter++) {
        page_addr = (*iter)->addr;
        root = get_lp_node(page_addr);
        bb_addr = get_basic_block(root, page_addr);
        if (is_block_comp_evict(bb_addr, MIN_PREFETCH_SIZE)) {
          update_basic_block_comp_evict(root, page_addr, MIN_PREFETCH_SIZE,
                                        false);
          break;
        }
      }
      if (iter != compressed_pages.end()) {
        comp_evicted_pages.push_back(
            std::make_pair(bb_addr, MIN_PREFETCH_SIZE));
      }
    }
  } else if (evict_policy == eviction_policy::TBN) {
    // we evict multiple 64KB pages in the 2 MB allocation where this evictable
    // belong
    std::list<eviction_t *>::iterator iter = valid_pages.begin();
    std::advance(iter, eviction_start);

    // find all basic blocks which are not staged/scheduled for write back or
    // not invalid or not in ld/st unit
    std::vector<mem_addr_t> all_basic_blocks;

    struct lp_tree_node *root;
    mem_addr_t page_addr;
    mem_addr_t bb_addr;

    for (; iter != valid_pages.end(); iter++) {
      page_addr = (*iter)->addr;

      root = get_lp_node(page_addr);

      bb_addr = get_basic_block(root, page_addr);

      if (is_block_evictable(bb_addr, MIN_PREFETCH_SIZE)) {
        if (compression_policy ==
            compression_policy_struct::PARTIAL) {  // compression is on
          if (selective_compression) {
            if (get_tree_comp_size(page_addr) <
                MIN_PREFETCH_SIZE/m_config.selective_compression_threshold) {  // compress is effective
              update_basic_block_comp(root, page_addr, MIN_PREFETCH_SIZE,
                                      false);
            } else {  // compress is not effective
              update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
            }
          } else {  // always compress
            update_basic_block_comp(root, page_addr, MIN_PREFETCH_SIZE, false);
          }
        } else {  // compression is off
          update_basic_block(root, page_addr, MIN_PREFETCH_SIZE, false);
        }
        break;
      }
    }

    if (iter != valid_pages.end()) {
      all_basic_blocks.push_back(bb_addr);
      if (compression_policy ==
          compression_policy_struct::PARTIAL) {  // compression is on
        if (selective_compression) {
          if (get_tree_comp_size(bb_addr) <
              MIN_PREFETCH_SIZE/m_config.selective_compression_threshold) {  // compress is effective
            traverse_and_remove_lp_tree_comp(root, all_basic_blocks);
          } else {  // compress is not effective
            traverse_and_remove_lp_tree(root, all_basic_blocks);
          }
        } else {  // always compress
          traverse_and_remove_lp_tree_comp(root, all_basic_blocks);
        }
      } else {  // compression is off
        traverse_and_remove_lp_tree(root, all_basic_blocks);
      }
    } else {  // compressed evict
      iter = compressed_pages.begin();
      for (; iter != compressed_pages.end(); iter++) {
        page_addr = (*iter)->addr;
        root = get_lp_node(page_addr);
        bb_addr = get_basic_block(root, page_addr);
        if (is_block_comp_evict(bb_addr, MIN_PREFETCH_SIZE)) {
          update_basic_block_comp_evict(root, page_addr, MIN_PREFETCH_SIZE,
                                        false);
          break;
        }
      }
      if (iter != compressed_pages.end()) {
        all_basic_blocks.push_back(bb_addr);
        traverse_and_remove_lp_tree_comp_evict(
            root, all_basic_blocks);  // evict comp list
        comp_evict = true;
      }
    }
    // group all contiguous basic blocks if possible
    std::vector<mem_addr_t>::iterator bb = all_basic_blocks.begin();
    while (bb != all_basic_blocks.end()) {
      std::vector<mem_addr_t>::iterator next_bb = bb;
      size_t cur_num = 0;
      do {
        next_bb++;
        cur_num++;
      } while (next_bb != all_basic_blocks.end() &&
               ((*next_bb) == ((*bb) + cur_num * MIN_PREFETCH_SIZE)));
      if (!comp_evict) {
        evicted_pages.push_back(
            std::make_pair((*bb), (cur_num * MIN_PREFETCH_SIZE)));
      } else {
        comp_evicted_pages.push_back(
            std::make_pair((*bb), (cur_num * MIN_PREFETCH_SIZE)));
      }
      bb = next_bb;
    }
  }

  // always write back the chunk no matter what it has not dirty pages or dirty
  // pages
  for (const auto& iter : evicted_pages) {
    std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();

    p_t->start_addr = iter.first;
    p_t->comp_ratio =
        get_tree_comp_size(iter.first);
    p_t->comp_band = m_config.comp_band_scalar;
    p_t->decomp_band = m_config.decomp_band_scalar;
    p_t->size = iter.second;
    p_t->leaf_size = get_lp_node(iter.first)->leaf_size;
    p_t->start_cycle = cur_cycle;
    latency_type ltype = latency_type::PCIE_WRITE_BACK;

    for (std::list<eviction_t *>::iterator it = valid_pages.begin();
         it != valid_pages.end(); it++) {
      if ((*it)->addr <= iter.first &&
          iter.first < (*it)->addr + (*it)->size) {
        if ((*it)->RW == 1) {  // if read
          ltype = latency_type::INVALIDATE;
          break;
        }
      }
    }

    p_t->type = ltype;

    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);

      p_t->page_list.push_back(page_num);

      if (compression_policy ==
          compression_policy_struct::PARTIAL) {  // compression is on
        if (selective_compression) {             // selective compression is on
          if (m_gpu->get_global_memory()->get_pages_comp_ratio(
                  p_t->start_addr) >= 2) {  // compress is effective
            valid_pages_compressing(page_num, p_t->comp_size, p_t->leaf_size);
            p_t->type = latency_type::COMP;
          } else {  // compress is not effective
            valid_pages_erase(page_num, p_t->leaf_size);
          }
        } else {  // always compress
          valid_pages_compressing(page_num, p_t->comp_size, p_t->leaf_size);
          p_t->type = latency_type::COMP;
        }
      } else {  // compression is off
        valid_pages_erase(page_num, p_t->leaf_size);
      }
    } else {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);

      for (int i = 0; i < (int)(iter.second / m_config.page_size); i++) {
        p_t->page_list.push_back(page_num + i);
        if (compression_policy ==
            compression_policy_struct::PARTIAL) {  // only partial compression
          if (selective_compression) {  // selective compression is on
            if (m_gpu->get_global_memory()->get_pages_comp_ratio(
                    p_t->start_addr) >= 2) {  // compress is effective
              valid_pages_compressing(page_num + i, p_t->comp_size, p_t->leaf_size);
              p_t->type = latency_type::COMP;
            } else {  // compress is not effective
              valid_pages_erase(page_num + i, p_t->leaf_size);
            }
          } else {  // always compress
            valid_pages_compressing(page_num + i, p_t->comp_size, p_t->leaf_size);
            p_t->type = latency_type::COMP;
          }
        } else {  // compression is off
          valid_pages_erase(page_num + i, p_t->leaf_size);
        }
      }
    }
    if (compression_policy ==
        compression_policy_struct::PARTIAL) {  // compression is on
      if (selective_compression) {
        if (get_tree_comp_size(p_t->start_addr) <
            MIN_PREFETCH_SIZE/m_config.selective_compression_threshold) {  // compress is effective
          reserve_crm_handling(p_t);
          comp_stage_queue.push_back(p_t);
          num_comp_stage_queue += (p_t->size-p_t->comp_size)/m_config.page_size;
        } else {  // compress is not effective
          reserve_crm_handling(p_t);
          pcie_write_stage_queue.push_back(p_t);
          num_write_stage_queue += p_t->page_list.size();
        }
      } else {  // always compress
        reserve_crm_handling(p_t);
        comp_stage_queue.push_back(p_t);
        num_comp_stage_queue +=  (p_t->size-p_t->comp_size)/m_config.page_size;
      }
    } else {  // no compression
      reserve_crm_handling(p_t);
      pcie_write_stage_queue.push_back(p_t);
      num_write_stage_queue += p_t->page_list.size();
    }
  }
  for (const auto& iter : comp_evicted_pages) {
    std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();

    p_t->start_addr = iter.first;
    p_t->comp_ratio =
          get_tree_comp_size(iter.first);
    p_t->comp_band = m_config.comp_band_scalar;
    p_t->decomp_band = m_config.decomp_band_scalar;
    p_t->size = iter.second;
    p_t->leaf_size = get_lp_node(iter.first)->leaf_size;
    p_t->start_cycle = cur_cycle;

    latency_type ltype = latency_type::COMP_EVICT;
    if (m_config.invalidate_clean) {
      for (std::list<eviction_t *>::iterator it = compressed_pages.begin();
           it != compressed_pages.end(); it++) {
        if ((*it)->addr <= iter.first &&
            iter.first < (*it)->addr + (*it)->size) {
          if ((*it)->RW == 1) {  // if read
            ltype = latency_type::COMP_INV_EVICT;
            break;
          }
        }
      }
    }
    p_t->type = ltype;

    if (m_config.page_size == MAX_PREFETCH_SIZE) {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);

      p_t->page_list.push_back(page_num);
      compress_pages_erase(page_num, p_t->leaf_size);
    } else {
      mem_addr_t page_num =
          m_gpu->get_global_memory()->get_page_num(iter.first);

      for (int i = 0; i < (int)(iter.second / m_config.page_size); i++) {
        p_t->page_list.push_back(page_num + i);
        compress_pages_erase(page_num + i, p_t->leaf_size);
      }
    }
    reserve_crm_handling(p_t);
    pcie_write_stage_queue.push_back(p_t);
    num_write_stage_queue += p_t->page_list.size() / p_t->comp_ratio;
  }
}

// Helper function to split a page into smaller pages based on leaf_size
std::vector<eviction_t*> split_page(eviction_t* page, size_t leaf_size) {
  if (leaf_size > page->size) {
    throw std::invalid_argument("leaf_size cannot be larger than page size");
  }
  if (leaf_size == 0) {
    throw std::invalid_argument("leaf_size cannot be zero");
  }

  std::vector<eviction_t*> new_pages;
  size_t num_split = page->size / leaf_size;
  size_t remainder = page->size % leaf_size;

  // Create split pages
  for (size_t i = 0; i < num_split; ++i) {
    eviction_t* new_page = new eviction_t();
    new_page->addr = page->addr + i * leaf_size;
    new_page->size = leaf_size;
    new_page->comp_size = page->comp_size * leaf_size / page->size; // Proportional
    new_page->cycle = page->cycle;
    new_page->access_counter = page->access_counter;
    new_page->RW = page->RW;
    new_pages.push_back(new_page);
  }

  // Handle remainder
  if (remainder > 0) {
    eviction_t* new_page = new eviction_t();
    new_page->addr = page->addr + num_split * leaf_size;
    new_page->size = remainder;
    new_page->comp_size = page->comp_size * remainder / page->size;
    new_page->cycle = page->cycle;
    new_page->access_counter = page->access_counter;
    new_page->RW = page->RW;
    new_pages.push_back(new_page);
  }

  return new_pages;
}

void gmmu_t::erase_page_from_list(std::list<eviction_t*>& page_list,
                                  mem_addr_t page_addr, size_t leaf_size) {
  for (auto it = page_list.begin(); it != page_list.end(); ++it) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      eviction_t* old_page = *it;
      if (leaf_size >= old_page->size) {
        // Case 1: leaf_size matches page size, remove directly
        it = page_list.erase(it);
        delete old_page;
        return;
      }
      /*if (leaf_size > old_page->size) {
        assert(0&&"bst need to implement in future");
      }*/

      // Case 2: leaf_size < page size, split and remove matching page
      std::vector<eviction_t*> new_pages = split_page(old_page, leaf_size);
      it = page_list.erase(it);
      delete old_page;

      // Insert all new pages except the one containing page_addr
      std::vector<eviction_t*> pages_to_insert;
      for (eviction_t* new_page : new_pages) {
        if (new_page->addr <= page_addr && page_addr < new_page->addr + new_page->size) {
          delete new_page; // Delete the page to be erased
        } else {
          pages_to_insert.push_back(new_page);
        }
      }
      page_list.insert(it, pages_to_insert.begin(), pages_to_insert.end());
      return;
    }
  }
}

void gmmu_t::move_page_between_lists(std::list<eviction_t*>& from_list,
                                     std::list<eviction_t*>& to_list,
                                     mem_addr_t page_addr, size_t leaf_size) {
  for (auto it = from_list.begin(); it != from_list.end(); ++it) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      eviction_t* old_page = *it;

      if (leaf_size == old_page->size) {
        // 같은 범위의 항목이 없으면 단순히 이동
        it = from_list.erase(it);
        to_list.push_back(old_page);
        return;
      }
      
      if (leaf_size > old_page->size) {
        // leaf_size가 더 큰 경우: old_page와 연속된 페이지들만 병합
        mem_addr_t merge_start = (old_page->addr / leaf_size) * leaf_size;
        mem_addr_t merge_end = merge_start + leaf_size;

        // to_list에서 old_page와 연속된 페이지들만 찾음
        std::vector<std::list<eviction_t*>::iterator> to_merge_its;
        std::vector<eviction_t*> candidate_pages;
        
        for (auto to_it = to_list.begin(); to_it != to_list.end(); ++to_it) {
          if ((*to_it)->addr >= merge_start && (*to_it)->addr < merge_end) {
            if((*to_it)->size==leaf_size){
              from_list.erase(it);
              delete old_page;
              return;
            }
            candidate_pages.push_back(*to_it);
          }
        }
        
        // old_page도 후보에 추가
        candidate_pages.push_back(old_page);
        
        // 주소 순으로 정렬
        std::sort(candidate_pages.begin(), candidate_pages.end(), 
                  [](eviction_t* a, eviction_t* b) { return a->addr < b->addr; });
        
        // old_page 위치 찾기
        auto old_page_it = std::find(candidate_pages.begin(), candidate_pages.end(), old_page);
        if (old_page_it == candidate_pages.end()) {
          // 에러 처리 - 이런 일은 없어야 함
          printf("bst debug move_page_between_lists old_page not found\n");
          printf("bst debug move_page_between_lists old_page addr = %llu\n", old_page->addr);
          print_debug_all(true);
          it = from_list.erase(it);
          to_list.push_back(old_page);
          return;
        }
        
        // 연속된 페이지들 찾기
        std::vector<eviction_t*> continuous_pages;
        continuous_pages.push_back(old_page);
        
        // old_page 앞쪽에서 연속된 페이지들 찾기
        for (auto it_back = old_page_it; it_back != candidate_pages.begin(); ) {
          --it_back;
          eviction_t* prev_page = *it_back;
          eviction_t* curr_page = continuous_pages.front();
          
          if (prev_page->addr + prev_page->size == curr_page->addr) {
            continuous_pages.insert(continuous_pages.begin(), prev_page);
          } else {
            break;
          }
        }
        
        // old_page 뒤쪽에서 연속된 페이지들 찾기
        for (auto it_forward = old_page_it + 1; it_forward != candidate_pages.end(); ++it_forward) {
          eviction_t* next_page = *it_forward;
          eviction_t* curr_page = continuous_pages.back();
          
          if (curr_page->addr + curr_page->size == next_page->addr) {
            continuous_pages.push_back(next_page);
          } else {
            break;
          }
        }
        
        for (size_t i = 0; i < continuous_pages.size(); i++) {
        }
        
        // 연속된 페이지가 old_page 하나뿐인 경우
        if (continuous_pages.size() == 1) {
          it = from_list.erase(it);
          to_list.push_back(old_page);
          return;
        }
        
        // 병합 정보 계산
        mem_addr_t continuous_start = continuous_pages.front()->addr;
        mem_addr_t continuous_end = continuous_pages.back()->addr + continuous_pages.back()->size;
        size_t continuous_size = continuous_end - continuous_start;
        
        size_t merged_comp_size = 0;
        int merged_RW = old_page->RW;
        unsigned long long merged_cycle = old_page->cycle;
        unsigned long long merged_access_counter = old_page->access_counter;
        
        for (eviction_t* page : continuous_pages) {
          merged_comp_size += page->comp_size;
          if (page->cycle > merged_cycle) merged_cycle = page->cycle;
          if (page->access_counter > merged_access_counter) merged_access_counter = page->access_counter;
          if (page->RW > merged_RW) merged_RW = page->RW;
        }
        
        // to_list에서 연속된 페이지들 삭제 (old_page 제외)
        for (eviction_t* page : continuous_pages) {
          if (page != old_page) {
            for (auto to_it = to_list.begin(); to_it != to_list.end(); ++to_it) {
              if (*to_it == page) {
                delete *to_it;
                to_list.erase(to_it);
                break;
              }
            }
          }
        }
        
        // old_page도 from_list에서 삭제
        it = from_list.erase(it);
        
        // 새 eviction_t 생성
        eviction_t* merged_page = new eviction_t();
        merged_page->addr = continuous_start;
        merged_page->size = continuous_size;  // 연속된 영역의 실제 크기
        if(continuous_size > leaf_size){
          printf("bst debug move_page_between_lists continuous_size = %llu leaf_size = %llu\n", continuous_size, leaf_size);
          print_debug_all(true);
        }
        merged_page->comp_size = merged_comp_size;
        merged_page->cycle = merged_cycle;
        merged_page->access_counter = merged_access_counter;
        merged_page->RW = merged_RW;
        
        to_list.push_back(merged_page);
        delete old_page;
        return;
      }

      // leaf_size < old_page->size 인 경우 기존 split 방식
      std::vector<eviction_t*> new_pages = split_page(old_page, leaf_size);
      it = from_list.erase(it);
      delete old_page;

      // Insert all new pages except the one containing page_addr
      std::vector<eviction_t*> pages_to_insert;
      for (eviction_t* new_page : new_pages) {
        if (new_page->addr <= page_addr && page_addr < new_page->addr + new_page->size) {
          to_list.push_back(new_page); // Move to to_list
        } else {
          pages_to_insert.push_back(new_page);
        }
      }
      from_list.insert(it, pages_to_insert.begin(), pages_to_insert.end());
      return;
    }
  }
}

void gmmu_t::move_page_between_lists_comp(std::list<eviction_t*>& from_list,
                                          std::list<eviction_t*>& to_list,
                                          mem_addr_t page_addr,
                                          size_t comp_size,
                                          size_t leaf_size) {
  for (auto it = from_list.begin(); it != from_list.end(); ++it) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      eviction_t* old_page = *it;

      // old_page->size == leaf_size인 경우: 단순히 comp_size만 업데이트하고 이동
      if (old_page->size == leaf_size) {
        old_page->comp_size = comp_size;
        it = from_list.erase(it);
        to_list.push_back(old_page);
        return;
      }

      // leaf_size > old_page->size 인 경우: old_page와 연속된 페이지들만 병합
      if (leaf_size > old_page->size) {
        // leaf_size의 경계 계산
        mem_addr_t merge_start = (old_page->addr / leaf_size) * leaf_size;
        mem_addr_t merge_end = merge_start + leaf_size;

        // to_list에서 old_page와 연속된 페이지들만 찾음
        std::vector<eviction_t*> candidate_pages;
        
        // leaf 경계 내의 모든 페이지를 후보로 수집
        for (auto to_it = to_list.begin(); to_it != to_list.end(); ++to_it) {
          if ((*to_it)->addr >= merge_start && (*to_it)->addr < merge_end) {
            candidate_pages.push_back(*to_it);
          }
        }
        
        // old_page도 후보에 추가
        candidate_pages.push_back(old_page);
        
        // 주소 순으로 정렬
        std::sort(candidate_pages.begin(), candidate_pages.end(), 
                  [](eviction_t* a, eviction_t* b) { return a->addr < b->addr; });
        
        // old_page 위치 찾기
        auto old_page_it = std::find(candidate_pages.begin(), candidate_pages.end(), old_page);
        if (old_page_it == candidate_pages.end()) {
          // 에러 처리
          printf("bst debug move_page_between_lists_comp old_page not found\n");
          printf("bst debug move_page_between_lists_comp old_page addr = %llu\n", old_page->addr);
          print_debug_all(true);
          old_page->comp_size = comp_size;
          it = from_list.erase(it);
          to_list.push_back(old_page);
          return;
        }
        
        // 연속된 페이지들 찾기
        std::vector<eviction_t*> continuous_pages;
        continuous_pages.push_back(old_page);
        
        // old_page 앞쪽에서 연속된 페이지들 찾기
        for (auto it_back = old_page_it; it_back != candidate_pages.begin(); ) {
          --it_back;
          eviction_t* prev_page = *it_back;
          eviction_t* curr_page = continuous_pages.front();
          
          if (prev_page->addr + prev_page->size == curr_page->addr) {
            continuous_pages.insert(continuous_pages.begin(), prev_page);
          } else {
            break;
          }
        }
        
        // old_page 뒤쪽에서 연속된 페이지들 찾기
        for (auto it_forward = old_page_it + 1; it_forward != candidate_pages.end(); ++it_forward) {
          eviction_t* next_page = *it_forward;
          eviction_t* curr_page = continuous_pages.back();
          
          if (curr_page->addr + curr_page->size == next_page->addr) {
            continuous_pages.push_back(next_page);
          } else {
            break;
          }
        }
        
        // 연속된 페이지가 old_page 하나뿐인 경우
        if (continuous_pages.size() == 1) {
          old_page->comp_size = comp_size;
          it = from_list.erase(it);
          to_list.push_back(old_page);
          return;
        }
        
        // 병합 정보 계산
        mem_addr_t continuous_start = continuous_pages.front()->addr;
        mem_addr_t continuous_end = continuous_pages.back()->addr + continuous_pages.back()->size;
        size_t continuous_size = continuous_end - continuous_start;
        
        size_t merged_comp_size = comp_size;  // old_page의 새로운 comp_size
         unsigned long long merged_cycle = old_page->cycle;
        unsigned merged_access_counter = old_page->access_counter;
        unsigned merged_RW = old_page->RW;
        
        for (eviction_t* page : continuous_pages) {
          if (page != old_page) {
            merged_comp_size += page->comp_size;
          }
          if (page->cycle > merged_cycle) merged_cycle = page->cycle;
          if (page->access_counter > merged_access_counter) merged_access_counter = page->access_counter;
          if (page->RW > merged_RW) merged_RW = page->RW;
        }
        
        // to_list에서 연속된 페이지들 삭제 (old_page 제외)
        for (eviction_t* page : continuous_pages) {
          if (page != old_page) {
            for (auto to_it = to_list.begin(); to_it != to_list.end(); ++to_it) {
              if (*to_it == page) {
                delete *to_it;
                to_list.erase(to_it);
                break;
              }
            }
          }
        }
        
        // from_list에서 old_page 삭제
        it = from_list.erase(it);
        
        // 새 eviction_t 생성 및 to_list에 추가
        eviction_t* merged_page = new eviction_t();
        merged_page->addr = continuous_start;
        merged_page->size = continuous_size;  // 연속된 영역의 실제 크기
        if(continuous_size > leaf_size){
          printf("bst debug move_page_between_lists_comp continuous_size = %llu leaf_size = %llu\n", continuous_size, leaf_size);
          print_debug_all(true);
        }
        merged_page->comp_size = merged_comp_size;
        merged_page->cycle = merged_cycle;
        merged_page->access_counter = merged_access_counter;
        merged_page->RW = merged_RW;
        
        to_list.push_back(merged_page);
        delete old_page;
        return;
      }

      // leaf_size < old_page->size 인 경우 기존 split 방식
      std::vector<eviction_t*> new_pages = split_page(old_page, leaf_size);
      it = from_list.erase(it);
      delete old_page;

      // page_addr가 포함된 페이지만 to_list로 이동, 나머지는 from_list에 다시 삽입
      std::vector<eviction_t*> pages_to_insert;
      for (eviction_t* new_page : new_pages) {
        if (new_page->addr <= page_addr && page_addr < new_page->addr + new_page->size) {
          new_page->comp_size = comp_size; // 이동되는 페이지의 comp_size 갱신
          to_list.push_back(new_page);
        } else {
          pages_to_insert.push_back(new_page);
        }
      }
      from_list.insert(it, pages_to_insert.begin(), pages_to_insert.end());
      return;
    }
  }
}

bool gmmu_t::check_page_in_list(std::list<eviction_t *> &page_list,
                          mem_addr_t page_addr){
  for(auto iter = page_list.begin(); iter != page_list.end(); iter++){
    if((*iter)->addr <= page_addr && page_addr < (*iter)->addr + (*iter)->size){
      return true;
    }
  }
  return false;
}
bool gmmu_t::check_page_in_list_comp(std::list<eviction_t *> &page_list,
                          mem_addr_t page_addr){
  for(auto iter = page_list.begin(); iter != page_list.end(); iter++){
    if((*iter)->addr <= page_addr && page_addr < (*iter)->addr + (*iter)->size){
        unsigned long long total_comp_size = 0;
        // 64KB 배수일 때는 for 루프로 처리
        for(int i=0;i<(*iter)->size/MIN_PREFETCH_SIZE;i++){
          total_comp_size += get_tree_comp_size((*iter)->addr + i * MIN_PREFETCH_SIZE);
        }
        // 16KB나 32KB 나머지일 때는 MIN_PREFETCH 기준으로 나눈 get_tree_comp_size를 나눠서 넣기
        if((*iter)->size % MIN_PREFETCH_SIZE > 0){//16KB ore 32KB remaining
          size_t remain_size = (*iter)->size % MIN_PREFETCH_SIZE;
          total_comp_size += get_tree_comp_size((*iter)->addr) * remain_size / MIN_PREFETCH_SIZE;
        }
        (*iter)->comp_size = total_comp_size;
      return true;
    }
  }
  return false;
}
void gmmu_t::valid_pages_erase(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(valid_pages, evicting_pages, page_addr, leaf_size);
  for(auto iter = evicting_pages.begin(); iter != evicting_pages.end(); iter++){
    if(!check_page_in_list(evicting_pages, page_addr)){//이동했는데 없을경우우
      printf("bst debug valid_pages_erase page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::evicting_pages_erase(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  erase_page_from_list(evicting_pages, page_addr, leaf_size);
  for(auto iter = evicting_pages.begin(); iter != evicting_pages.end(); iter++){
    if(check_page_in_list(evicting_pages, page_addr)){//삭제했는데 여전히 있을 경우우
      printf("bst debug evicting_pages_erase page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::cancel_evicting_pages(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(evicting_pages, valid_pages, page_addr, leaf_size);
  for(auto iter = valid_pages.begin(); iter != valid_pages.end(); iter++){
    if(!check_page_in_list(valid_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug cancel_evicting_pages page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::compress_pages_erase(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(compressed_pages, comp_evict_pages, page_addr, leaf_size);
  for(auto iter = comp_evict_pages.begin(); iter != comp_evict_pages.end(); iter++){
    if(!check_page_in_list(comp_evict_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug compress_pages_erase page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::comp_evict_erase(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  erase_page_from_list(comp_evict_pages, page_addr, leaf_size);
  for(auto iter = comp_evict_pages.begin(); iter != comp_evict_pages.end(); iter++){
    if(check_page_in_list(comp_evict_pages, page_addr)){//삭제했는데 여전히 있을 경우
      printf("bst debug comp_evict_erase page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::compress_pages_comp_evict(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(compressed_pages, comp_evict_pages, page_addr, leaf_size);
  for(auto iter = comp_evict_pages.begin(); iter != comp_evict_pages.end(); iter++){
    if(!check_page_in_list(comp_evict_pages, page_addr)){
      printf("bst debug compress_pages_comp_evict page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}

void gmmu_t::comp_evict_erase_recover(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(comp_evict_pages, compressed_pages, page_addr, leaf_size);
  if(!check_page_in_list(compressed_pages, page_addr)){
    if(m_config.MICRO_barrier_Range && m_config.tree_logic == 2){
        // MICRO 모드: 이미 host 로 comp-evict 완료된 단일 64KB 블록만 compressed 로 복원.
        lp_tree_node *root = get_lp_node(page_addr);
        mem_addr_t bb64 = (page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE;
        update_basic_block_comp_evict(root, bb64, MIN_PREFETCH_SIZE, true);
        mem_addr_t bb64_page = m_gpu->get_global_memory()->get_page_num(bb64);
        for(mem_addr_t i = 0; i < MIN_PREFETCH_SIZE / m_config.page_size; i++){
          m_gpu->get_global_memory()->invalidate_page(bb64_page + i);
          m_gpu->get_global_memory()->set_iter_compressed(bb64_page + i);
        }
        size_t comp_size = get_tree_comp_size(bb64);
        m_gpu->get_global_memory()->change_size_comp_evict_with_comp_size(MIN_PREFETCH_SIZE, comp_size, false);
        eviction_t* comp_page = new eviction_t();
        comp_page->addr = bb64;
        comp_page->size = MIN_PREFETCH_SIZE;
        comp_page->comp_size = comp_size;
        comp_page->cycle = 0;
        comp_page->access_counter = 0;
        comp_page->RW = 0;
        compressed_pages.push_back(comp_page);
    }
    else if(m_config.tree_logic == 2 || m_config.tree_logic == 3){//range tree or forest
        // Tree 비트 업데이트 (압축 상태로 복원)
        lp_tree_node *root = get_lp_node(page_addr);
        mem_addr_t bb_addr = get_basic_block(root, page_addr);//get leaf_size block addr
        update_basic_block_comp_evict(root, bb_addr, root->leaf_size, true);
        printf("bst debug comp_evict_erase_recover root addr = %llu, page_addr = %llu, bb_addr = %llu\n", root->addr, page_addr, bb_addr);
        printf("bst debug comp_evict_erase_recover leaf_size = %llu, root_leaf_size = %llu\n", leaf_size, root->leaf_size);
        print_debug_all(true);
        // 페이지 개수 올바르게 계산
        mem_addr_t compressed_pages_num = (page_addr - bb_addr) / m_config.page_size;
        mem_addr_t max_pages_in_leaf = root->leaf_size / m_config.page_size;
        compressed_pages_num = std::min(compressed_pages_num, max_pages_in_leaf);
        
        // 페이지 상태 업데이트 (압축된 상태로 복원)
        mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(bb_addr);
        for(mem_addr_t i = 0; i < compressed_pages_num; i++){
          m_gpu->get_global_memory()->invalidate_page(page_num + i);
          m_gpu->get_global_memory()->set_iter_compressed(page_num + i);
        }
        
        // 압축 크기 조정 (64K 단위) - 압축 크기를 다시 추가
        mem_addr_t full_64k_blocks = compressed_pages_num / 16;
        mem_addr_t remaining_pages = compressed_pages_num % 16;
        
        for(mem_addr_t i = 0; i < full_64k_blocks; i++){
          size_t comp_size = get_tree_comp_size(bb_addr + i * MIN_PREFETCH_SIZE);
          m_gpu->get_global_memory()->change_size_comp_evict_with_comp_size(MIN_PREFETCH_SIZE, comp_size, false);
          eviction_t* comp_page = new eviction_t();
          comp_page->addr = bb_addr + i * MIN_PREFETCH_SIZE;
          comp_page->size = MIN_PREFETCH_SIZE;
          comp_page->comp_size = comp_size;
          comp_page->cycle = 0;
          comp_page->access_counter = 0;
          comp_page->RW = 0;
          compressed_pages.push_back(comp_page);
        }
        
        if(remaining_pages > 0){
          size_t total_comp_size = get_tree_comp_size(bb_addr);
          size_t partial_comp_size = (total_comp_size * remaining_pages) / 16;
          size_t partial_size = remaining_pages * m_config.page_size;
          m_gpu->get_global_memory()->change_size_comp_evict_with_comp_size(partial_size, partial_comp_size, false);
          eviction_t* comp_page = new eviction_t();
          comp_page->addr = bb_addr + full_64k_blocks * MIN_PREFETCH_SIZE;
          comp_page->size = remaining_pages * m_config.page_size;
          comp_page->comp_size = partial_comp_size;
          comp_page->cycle = 0;
          comp_page->access_counter = 0;
          comp_page->RW = 0;
          compressed_pages.push_back(comp_page);
        }
      }
      else{
        printf("bst debug comp_evict_erase_recover page_addr = %llu\n", page_addr);
        print_debug_all(true);
      }
    }
}

void gmmu_t::valid_pages_compressing(mem_addr_t page_num, size_t comp_size, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists_comp(valid_pages, compressing_pages, page_addr, comp_size, leaf_size);
  for(auto iter = compressing_pages.begin(); iter != compressing_pages.end(); iter++){
    if(!check_page_in_list_comp(compressing_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug valid_pages_compressing page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}


void gmmu_t::compressed_pages_decompressing(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(compressed_pages, decompressing_pages, page_addr, leaf_size);
  for(auto iter = decompressing_pages.begin(); iter != decompressing_pages.end(); iter++){
    if(!check_page_in_list(decompressing_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug compressed_pages_decompressing page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}
void gmmu_t::decompressing_pages_done(mem_addr_t page_num, size_t leaf_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(decompressing_pages, valid_pages, page_addr, leaf_size);
  for(auto iter = valid_pages.begin(); iter != valid_pages.end(); iter++){
    if(!check_page_in_list(valid_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug decompressing_pages_done page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}
void gmmu_t::compressing_pages_done(mem_addr_t page_num, size_t leaf_size) {
  enable_comp_evict();
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(compressing_pages, compressed_pages, page_addr, leaf_size);
  for(auto iter = compressed_pages.begin(); iter != compressed_pages.end(); iter++){
    if(!check_page_in_list(compressed_pages, page_addr)){//이동후 to_list에 없는 경우우
      printf("bst debug compressing_pages_done page_addr = %llu\n", page_addr);
      print_debug_all(true);
    }
  }
}
bool gmmu_t::cancel_compressing_pages(mem_addr_t page_num, size_t leaf_size, bool change_size) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  move_page_between_lists(compressing_pages, valid_pages, page_addr, leaf_size);
  bool changed_size = false;
    if(!check_page_in_list(valid_pages, page_addr)){
      if(m_config.MICRO_barrier_Range && m_config.tree_logic == 2){
        // MICRO 모드: 이미 compressed 상태가 된 단일 64KB 블록만 valid 로 되돌린다.
        // (2MB-prefix 일괄 복원은 혼합 leaf 에서 비트/회계 누수를 일으키므로 금지)
        lp_tree_node *root = get_lp_node(page_addr);
        mem_addr_t bb64 = (page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE;
        update_basic_block_comp(root, bb64, MIN_PREFETCH_SIZE, true);
        // 해당 64KB 를 포함하는 compressed_pages 엔트리(=64KB)를 valid_pages 로 이동
        for(auto iter = compressed_pages.begin(); iter != compressed_pages.end();){
          if((*iter)->addr <= page_addr && page_addr < (*iter)->addr + (*iter)->size){
            valid_pages.push_back(*iter);
            iter = compressed_pages.erase(iter);
            break;
          } else {
            ++iter;
          }
        }
        mem_addr_t bb64_page = m_gpu->get_global_memory()->get_page_num(bb64);
        for(mem_addr_t i = 0; i < MIN_PREFETCH_SIZE / m_config.page_size; i++){
          m_gpu->get_global_memory()->validate_page(bb64_page + i);
          m_gpu->get_global_memory()->set_iter_decompressed(bb64_page + i);
        }
        changed_size = true;
        size_t comp_size = get_tree_comp_size(bb64);
        if(!m_gpu->get_global_memory()->change_size_comp_with_comp_size(MIN_PREFETCH_SIZE - comp_size, false)) {
          printf("bst debug cancel_compressing_pages(MICRO) change_size_comp_with_comp_size failed\n");
          print_debug_all(true);
        }
      }
      else if(m_config.tree_logic == 2 || m_config.tree_logic == 3){//range tree or forest
        lp_tree_node *root = get_lp_node(page_addr);
        mem_addr_t bb_addr = get_basic_block(root, page_addr);//get leaf_size block addr

        // Tree 비트 업데이트
        update_basic_block_comp(root, bb_addr, root->leaf_size, true);

        // Compressed pages에서 해당 범위 찾아서 valid_pages로 이동
        for(auto iter = compressed_pages.begin(); iter != compressed_pages.end();){
          if((*iter)->addr <= page_addr && page_addr < (*iter)->addr + (*iter)->size){
            valid_pages.push_back(*iter);
            iter = compressed_pages.erase(iter);
            break;
          } else {
            ++iter;
          }
        }

        // 페이지 개수 올바르게 계산
        mem_addr_t compressed_pages_num = (page_addr - bb_addr) / m_config.page_size;
        mem_addr_t max_pages_in_leaf = root->leaf_size / m_config.page_size;
        compressed_pages_num = std::min(compressed_pages_num, max_pages_in_leaf);
        
        // 페이지 상태 업데이트
        mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(bb_addr);
        for(mem_addr_t i = 0; i < compressed_pages_num; i++){
          m_gpu->get_global_memory()->validate_page(page_num + i);
          m_gpu->get_global_memory()->set_iter_decompressed(page_num + i);
        }
        
        // 압축 크기 조정 (64K 단위)
        mem_addr_t full_64k_blocks = compressed_pages_num / 16;
        mem_addr_t remaining_pages = compressed_pages_num % 16;
        changed_size = true;
        for(mem_addr_t i = 0; i < full_64k_blocks; i++){
          size_t comp_size = get_tree_comp_size(bb_addr + i * MIN_PREFETCH_SIZE);
          size_t size_diff = MIN_PREFETCH_SIZE - comp_size;
          if(!m_gpu->get_global_memory()->change_size_comp_with_comp_size(size_diff, false)) {
            printf("bst debug cancel_compressing_pages change_size_comp_with_comp_size failed\n");
            print_debug_all(true);
          }
        }
        
        if(remaining_pages > 0){
          size_t total_comp_size = get_tree_comp_size(bb_addr);
          size_t partial_comp_size = (total_comp_size * remaining_pages) / 16;
          size_t partial_size_diff = (MIN_PREFETCH_SIZE * remaining_pages / 16) - partial_comp_size;
          if(!m_gpu->get_global_memory()->change_size_comp_with_comp_size(partial_size_diff, false)) {
            printf("bst debug cancel_compressing_pages change_size_comp_with_comp_size failed\n");
            print_debug_all(true);
          }
        }
      }
      else{
        printf("bst debug cancel_compressing_pages page_addr = %llu\n", page_addr);
        print_debug_all(true);
      }
    }
    return changed_size;
}
bool gmmu_t::compressed_page_find(mem_addr_t page_num) {
  mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
  for (std::list<eviction_t *>::iterator it = compressed_pages.begin();
       it != compressed_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      printf("bst debug compressed page find  = %llu, pg_num = %llu\n",
             page_addr, page_num);
      assert(0);
      break;
    }
  }
}
/*
void gmmu_t::compressed_pages_decomp_addr(mem_addr_t page_addr) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(page_addr);
  for (std::list<eviction_t *>::iterator it = compressed_pages.begin();
       it != compressed_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      if (!valid_pages_check(page_num)) {
        valid_pages.push_back(*it);  // add valid page
      }
      compressed_pages.erase(it);  // delete page from compressed list
      break;
    }
  }
}*/
void gmmu_t::valid_pages_clear() { valid_pages.clear(); }
void gmmu_t::compressed_pages_clear() { compressed_pages.clear(); }
void gmmu_t::comp_evict_pages_clear() { comp_evict_pages.clear(); }
void gmmu_t::LCP_compressed_pages_clear() { LCP_compressed_pages_map.clear(); }
void gmmu_t::refresh_valid_pages(mem_addr_t page_addr, bool check) {
  bool valid = false;
  enable_evict();
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(page_addr);
  struct lp_tree_node *root = get_lp_node(page_addr);
  size_t leaf_size = micro_leaf(root);
   if(!root->inefficient_compression){
    enable_comp();
   }
  // 기존 페이지에서 해당 주소를 포함하는 페이지 찾기
  std::list<eviction_t *>::iterator found_it = valid_pages.end();
  for (std::list<eviction_t *>::iterator it = valid_pages.begin();
       it != valid_pages.end(); it++) {
    if ((*it)->addr <= page_addr && page_addr < (*it)->addr + (*it)->size) {
      if((*it)->size == leaf_size){
        (*it)->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      }
      valid = true;
      found_it = it;
      break;
    }
  }
  
  if(check&&!valid){
      printf("bst debug refresh_valid_pages page_addr = %llu page_num = %llu\n", page_addr, page_num);
      print_debug_all(true);
  }
  
  if (!valid) {
    // 새로운 페이지를 추가할 때 leaf_size에 맞게 처리
    mem_addr_t new_page_addr;
    size_t new_page_size;
    if (m_config.MICRO_barrier_Range && m_config.tree_logic == 2) {
      // MICRO mode: lists are 64KB-granular. get_eviction_base_addr returns the 2MB
      // basic-block base under TBN policy, which would not contain page_addr at 64KB
      // size; compute the 64KB-aligned base directly so addr/size stay consistent.
      new_page_addr = (page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE;
      new_page_size = MIN_PREFETCH_SIZE;
    } else {
      new_page_addr = get_eviction_base_addr(page_addr);
      new_page_size = get_eviction_granularity(page_addr, leaf_size);
    }

    // leaf_size 경계 내에서 기존 페이지들과 병합 가능한지 확인
    mem_addr_t merge_start = (new_page_addr / leaf_size) * leaf_size;
    mem_addr_t merge_end = merge_start + leaf_size;
    
    std::vector<std::list<eviction_t*>::iterator> to_merge_its;
    size_t merged_size = 0;
    size_t merged_comp_size = 0;
    int merged_RW = 0;
    unsigned long long merged_cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
    unsigned long long merged_access_counter = 0;
    
    // merge_start ~ merge_end 범위에 있는 기존 페이지들 찾기
    for (auto it = valid_pages.begin(); it != valid_pages.end(); ++it) {
      if ((*it)->addr >= merge_start && (*it)->addr < merge_end) {
        to_merge_its.push_back(it);
        merged_size += (*it)->size;
        merged_comp_size += (*it)->comp_size;
        if ((*it)->access_counter > merged_access_counter) merged_access_counter = (*it)->access_counter;
        if ((*it)->RW > merged_RW) merged_RW = (*it)->RW;
      }
    }
    
    if (!to_merge_its.empty()) {
      // 기존 페이지들과 병합
      for (auto rit = to_merge_its.rbegin(); rit != to_merge_its.rend(); ++rit) {
        delete **rit;
        valid_pages.erase(*rit);
      }
      
      eviction_t* merged_page = new eviction_t();
      merged_page->addr = merge_start;
      merged_page->size = leaf_size;
      if(merged_size > leaf_size){
        printf("bst debug merged_size > leaf_size\n");
        print_debug_all(true);
      }
      merged_page->comp_size = merged_comp_size;
      merged_page->cycle = merged_cycle;
      merged_page->access_counter = merged_access_counter;
      merged_page->RW = merged_RW;
      
      valid_pages.push_back(merged_page);
    } else {
      // 병합할 페이지가 없으면 새 페이지 추가
      eviction_t *item = new eviction_t();
      item->addr = new_page_addr;
      item->size = new_page_size;
      // 원본 비트들 초기화
      item->access_counter = 0;
      item->RW = 0;
      item->comp_size = 0;
      item->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      valid_pages.push_back(item);
    }
  } else if (found_it != valid_pages.end()) {
    // 기존 페이지가 있지만 leaf_size와 크기가 다른 경우 처리
    eviction_t* existing_page = *found_it;
    //같은거는 이미 처리함.
    if (leaf_size != existing_page->size) {
      if (leaf_size > existing_page->size) {
        // leaf_size가 더 큰 경우: leaf_size 경계 내에서 병합
        mem_addr_t merge_start = (existing_page->addr / leaf_size) * leaf_size;
        mem_addr_t merge_end = merge_start + leaf_size;
        
        std::vector<std::list<eviction_t*>::iterator> to_merge_its;
        size_t merged_size = existing_page->size;
        size_t merged_comp_size = existing_page->comp_size;
        int merged_RW = existing_page->RW;
        unsigned long long merged_cycle = existing_page->cycle;
        unsigned long long merged_access_counter = existing_page->access_counter;
        
        for (auto it = valid_pages.begin(); it != valid_pages.end(); ++it) {
          if (it != found_it && (*it)->addr >= merge_start && (*it)->addr < merge_end) {
            to_merge_its.push_back(it);
            merged_size += (*it)->size;
            merged_comp_size += (*it)->comp_size;
            if ((*it)->cycle > merged_cycle) merged_cycle = (*it)->cycle;
            if ((*it)->access_counter > merged_access_counter) merged_access_counter = (*it)->access_counter;
            if ((*it)->RW > merged_RW) merged_RW = (*it)->RW;
          }
        }
        
        if (!to_merge_its.empty()) {
          // 기존 페이지들과 병합
          for (auto rit = to_merge_its.rbegin(); rit != to_merge_its.rend(); ++rit) {
            delete **rit;
            valid_pages.erase(*rit);
          }
          
          // found_it도 삭제하고 새로 생성
          valid_pages.erase(found_it);
          delete existing_page;
          
          eviction_t* merged_page = new eviction_t();
          merged_page->addr = merge_start;
          merged_page->size = merged_size;
          merged_page->comp_size = merged_comp_size;
          merged_page->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
          merged_page->access_counter = merged_access_counter;
          merged_page->RW = merged_RW;
          
          valid_pages.push_back(merged_page);
        }
      } else {
        // leaf_size < existing_page->size 인 경우 분할
        std::vector<eviction_t*> new_pages = split_page(existing_page, leaf_size);
        valid_pages.erase(found_it);
        delete existing_page;
        // 모든 새 페이지들을 valid_pages에 추가
        for (eviction_t* new_page : new_pages) {
          if (new_page->addr <= page_addr && page_addr < new_page->addr + new_page->size) {
            new_page->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
          }
          valid_pages.push_back(new_page);
        }
      }
    }
  }
  if(m_config.fault_based_LRU){
    for (std::list<eviction_t *>::iterator it = valid_pages.begin();
         it != valid_pages.end(); it++) {
      struct lp_tree_node *it_root = get_lp_node((*it)->addr);
      if (it_root == root) {
        (*it)->cycle = m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle;
      }
    }
  }
}

void gmmu_t::sort_valid_pages() {
    for (auto* page : valid_pages) {
        auto* node = get_lp_node(page->addr);
        while (node->size > page->size && node->size > node->leaf_size) {
            auto* child = page->addr < node->right->addr ? node->left : node->right;
            if (page->addr + page->size > child->addr + child->size) break;
            node = child;
        }
        page->access_counter = node->size == node->leaf_size
                                  ? node->access_counter & ((1 << 27) - 1)
                                  : node->access_counter;
        page->RW = node->RW;
    }

    // 3) 정책별 정렬
    switch (evict_policy) {
        case eviction_policy::LFU:
            valid_pages.sort([](eviction_t* a, eviction_t* b) {
                if (a->access_counter != b->access_counter)
                    return a->access_counter < b->access_counter;
                if (a->RW             != b->RW)
                    return a->RW < b->RW;
                return a->cycle        < b->cycle;
            });
            break;

        case eviction_policy::TBN:
        case eviction_policy::SEQUENTIAL_LOCAL: {
            // 3-1) addr별로 그룹화
            std::unordered_map<mem_addr_t, std::vector<eviction_t*>> pageGroups;
            pageGroups.reserve(valid_pages.size());
            for (auto* page : valid_pages) {
                auto* rootNode = get_lp_node(page->addr);
                pageGroups[rootNode->addr].push_back(page);
            }

            // 3-2) 그룹별 maxCycle 계산해 저장
            struct PageGroup {
                uint64_t      maxCycle;
                std::vector<eviction_t*> pages;
            };
            std::vector<PageGroup> groupList;
            groupList.reserve(pageGroups.size());
            for (auto& [addr, pages] : pageGroups) {
                uint64_t maxC = pages.front()->cycle;
                for (auto* p : pages)
                    if (p->cycle > maxC) maxC = p->cycle;
                groupList.push_back({ maxC, std::move(pages) });
            }

            // 3-3) 그룹 순서: maxCycle 오름차순
            std::sort(groupList.begin(), groupList.end(),
                      [](const PageGroup& A, const PageGroup& B){ return A.maxCycle < B.maxCycle; });

            // 3-4) 각 그룹 내 페이지는 cycle 오름차순, 플래튼
            std::list<eviction_t*> newValidPages;
            for (auto& grp : groupList) {
                auto& pages = grp.pages;
                std::sort(pages.begin(), pages.end(),
                          [](eviction_t* A, eviction_t* B){ return A->cycle < B->cycle; });
                for (auto* p : pages)
                    newValidPages.push_back(p);
            }
            valid_pages.swap(newValidPages);
            break;
        }

        default:
            valid_pages.sort([](eviction_t* a, eviction_t* b){
                return a->cycle < b->cycle;
            });
    }
}
// return num_pages * (1 - 1 / comp_ratio)
double gmmu_t::get_size_comp_occupancy(
    size_t num_pages, float comp_ratio) {  // calculate change space for
                                           // decompressionn and compression
  return (double)num_pages * (1 - 1 / comp_ratio);
}
unsigned long long gmmu_t::get_ready_cycle_transfer(unsigned num_pages) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b *
                     ((float)(num_pages * m_config.page_size) / 1024.0));

  return 
         (unsigned long long)((float)(m_config.page_size * num_pages) *
                              m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}
void gmmu_t::reserve_crm_handling(
    const std::shared_ptr<pcie_latency_t> &request) {
  if (request->crm_handling_reserved) return;
  request->crm_handling_reserved = true;
  if (compression_policy != compression_policy_struct::PARTIAL) return;

  const bool host_restore = request->type == latency_type::PCIE_READ;
  switch (request->type) {
    case latency_type::COMP:
    case latency_type::DECOMP:
    case latency_type::COMP_FAULT:
    case latency_type::COMP_EVICT:
    case latency_type::COMP_INV_EVICT:
    case latency_type::PCIE_WRITE_BACK:
    case latency_type::INVALIDATE:
      break;
    case latency_type::PCIE_READ:
      if (!evict_form) return;
      break;
    default:
      return;  // Raw migration, host fault handling and DMA keep their timing.
  }

  // Charge once per 64KB transition, even for a batched eviction or restore.
  // Demand and proactive work share the same serialized CRM control path.
  auto globalMem = m_gpu->get_global_memory();
  std::set<mem_addr_t> blocks;
  for (auto page : request->page_list) {
    if (host_restore && !globalMem->is_evicted_compressed(page)) continue;
    const mem_addr_t addr = globalMem->get_mem_addr(page);
    const mem_addr_t first = addr / MIN_PREFETCH_SIZE;
    const mem_addr_t last = (addr + m_config.page_size - 1) / MIN_PREFETCH_SIZE;
    for (mem_addr_t block = first; block <= last; ++block) blocks.insert(block);
  }
  for (auto block : blocks) {
    // Concurrent fragments of one local block transition share its CRM charge.
    if (request->type == latency_type::COMP || request->type == latency_type::DECOMP) {
      unsigned long long prior_ready = 0;
      for (const auto* queue : {&comp_stage_queue, &comp_latency_queue,
                                &decomp_stage_queue, &decomp_latency_queue}) {
        for (const auto &prior : *queue) {
          if (prior->type != request->type || prior == request) continue;
          for (auto page : prior->page_list) {
            if (globalMem->get_mem_addr(page) / MIN_PREFETCH_SIZE == block) {
              prior_ready = max(prior_ready, prior->crm_ready_cycle);
              break;
            }
          }
        }
      }
      if (prior_ready != 0) {
        request->crm_ready_cycle = max(request->crm_ready_cycle, prior_ready);
        continue;
      }
    }
    // The PCIe read path can split one block around individual fault pages.
    // All fragments of that host-compressed restore share one control charge.
    if (host_restore) {
      auto prior = crm_host_restore_ready.find(block);
      if (prior != crm_host_restore_ready.end()) {
        request->crm_ready_cycle = max(request->crm_ready_cycle, prior->second);
        continue;
      }
    }
    last_crm_handling_cycle = max(cur_cycle, last_crm_handling_cycle) +
                              m_config.page_comp_fault_latency;
    request->crm_ready_cycle = max(request->crm_ready_cycle,
                                  last_crm_handling_cycle);
    if (host_restore) crm_host_restore_ready[block] = last_crm_handling_cycle;
  }
  // Only control is serialized. Data-plane completion does not occupy CRM.
  // Context save/restore is included in calibrated comp/decomp latency inputs.
}
unsigned long long gmmu_t::get_ready_cycle_pages_compress(unsigned num_pages,
                                                          float band) {
  // float speed = 2.0 * band / M_PI * m_config.curve_b *
  //               ((float)(num_pages * m_config.page_size) / 1024.0);//we use
  //               static band

  return cur_cycle +
         (unsigned long long)((float)(m_config.page_size * num_pages) *
                              m_config.core_freq / band /
                              (1024.0 * 1024.0 * 1024.0));
}
unsigned long long gmmu_t::get_compress_cycle(unsigned num_pages,
                                              float pages_band) {
  return (unsigned long long)((float)(m_config.page_size * num_pages) *
                              m_config.core_freq / pages_band /
                              (1024.0 * 1024.0 * 1024.0));
}
unsigned long long gmmu_t::get_ready_cycle_with_decomp(
    unsigned num_pages, float comp_band, unsigned num_compressed_pages) {
  // 전송 속도 계산
  float transfer_speed =
      2.0 * m_config.curve_a / M_PI *
      atan(m_config.curve_b *
           ((float)(num_pages * m_config.page_size) / 1024.0));

  // 전송 시간과 압축 시간의 합 계산
  unsigned long long transfer_time =
      (unsigned long long)((float)(m_config.page_size * num_pages) *
                           m_config.core_freq / transfer_speed /
                           (1024.0 * 1024.0 * 1024.0));

  unsigned long long comp_time =
      (unsigned long long)((float)(m_config.page_size * num_compressed_pages) *
                           m_config.core_freq / comp_band /
                           (1024.0 * 1024.0 * 1024.0));

  return cur_cycle + transfer_time + comp_time;
}

unsigned long long gmmu_t::get_ready_cycle(unsigned num_pages) {
  return get_ready_cycle_bytes((size_t)num_pages * m_config.page_size);
}

unsigned long long gmmu_t::get_ready_cycle_bytes(size_t bytes) {
  if (bytes == 0) return cur_cycle;
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)bytes / 1024.0));
  return cur_cycle +
         (unsigned long long)((float)bytes * m_config.core_freq / speed /
                              (1024.0 * 1024.0 * 1024.0));
}

unsigned long long gmmu_t::get_ready_cycle_dma(unsigned size) {
  float speed = 2.0 * m_config.curve_a / M_PI *
                atan(m_config.curve_b * ((float)(size) / 1024.0));
  return cur_cycle + 200;
}

float gmmu_t::get_pcie_utilization(unsigned num_pages) {
  return 2.0 * m_config.curve_a / M_PI *
         atan(m_config.curve_b *
              ((float)(num_pages * m_config.page_size) / 1024.0)) /
         m_config.pcie_bandwidth;
}

void gmmu_t::activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  for (std::list<prefetch_req>::iterator iter = prefetch_req_buffer.begin();
       iter != prefetch_req_buffer.end(); iter++) {
    if (iter->start_addr == m_device_addr && iter->size == m_cnt &&
        iter->m_stream->get_uid() == m_stream->get_uid()) {
      assert(iter->cur_addr == m_device_addr);
      iter->active = true;
      return;
    }
  }
}

void gmmu_t::register_prefetch(mem_addr_t m_device_addr,
                               mem_addr_t m_device_allocation_ptr, size_t m_cnt,
                               struct CUstream_st *m_stream) {
  struct prefetch_req pre_q;

  pre_q.start_addr = m_device_addr;
  pre_q.cur_addr = m_device_addr;
  pre_q.allocation_addr = m_device_allocation_ptr;
  pre_q.size = m_cnt;
  pre_q.active = false;
  pre_q.m_stream = m_stream;

  prefetch_req_buffer.push_back(pre_q);
}
struct lp_tree_node *gmmu_t::make_4MB_lp_tree(struct lp_tree_node *node_left, struct lp_tree_node *node_right) {
  // 4MB 노드 생성: 두 자식 노드의 주소와 크기, 트리 타입, alloc_count를 사용
  struct lp_tree_node *node_4MB = build_lp_tree(
    node_left->addr,
    node_left->size + node_right->size,
    1,
    node_left->alloc_count
  );
  node_4MB->left = node_left;
  node_4MB->right = node_right;
  node_left->parent = node_4MB;
  node_right->parent = node_4MB;
  node_4MB->tree_type = 1;
  node_4MB->alloc_count = node_left->alloc_count;
  node_4MB->leaf_size = 256 * 1024;
  node_4MB->root_size = 4 * 1024 * 1024;
  return node_4MB;
}
void gmmu_t::reset_lp_tree_type(struct lp_tree_node *node,uint8_t tree_type){//change tree type of node and reset left and right node
  node->tree_type = tree_type;
  switch(tree_type) {
    case 0: // 기본 TBN 및 LC
      node->root_size = 2 * 1024 * 1024; // 2MB
      node->leaf_size = 64 * 1024;       // 64K
      break;
    case 1: // LS 트리
      node->root_size = 4 * 1024 * 1024;   // 4MB
      node->leaf_size = 256 * 1024;        // 256K
      break;
    case 2: // HCHI 트리
      node->root_size = 512 * 1024;        // 512K
      node->leaf_size = 64 * 1024;         // 64K
      break;
    case 3: // HCLI 트리
      node->root_size = 512 * 1024;        // 512K
      node->leaf_size = 16 * 1024;         // 16K
      break;
    case 4: // range
      node->root_size = 2 * 1024 * 1024;   // 2MB
      node->leaf_size = 2 * 1024 * 1024;   // 2MB
      break;
    default:
      node->root_size = 2 * 1024 * 1024; // 2MB
      node->leaf_size = 64 * 1024;       // 64K
      assert(0 && "default tree type is not supported");
      break;
  }
  if (node->size != PREFETCH_16K_SIZE) {
    reset_lp_tree_type(node->left,tree_type);
    reset_lp_tree_type(node->right,tree_type);
  }
  if(node->size == 2 * 1024 * 1024 && node->tree_type == 1&& node->parent != NULL) {//if node is 2MB root and tree type is 1, then change parent node to 4MB root and 256K leaf
    node->parent->root_size = 4 * 1024 * 1024;
    node->parent->leaf_size = 256 * 1024;
    node->parent->tree_type = 1;
  }
}
struct lp_tree_node *gmmu_t::build_lp_tree(mem_addr_t addr, size_t size,
                                           uint8_t tree_type,uint8_t alloc_count) {
  struct lp_tree_node *node = new lp_tree_node();
  node->addr = addr;
  node->size = size;
  node->comp_ratio = 0;
  node->valid_size = 0;
  node->compress_size = 0;
  node->access_counter = 0;
  node->RW = 0;
  node->comp_size = 0;
  node->large_pages_reserved = false;  // edit for real_TBN
  node->sampled = false;
  node->inefficient_compression = false;
  node->alloc_count = alloc_count;
  node->tree_type = tree_type;
  node->leaf_will_be_inefficient = false;
  node->fault_index = 0;
  node->prefetch_threshold=100.0;
  for(int i=0;i<32;i++){
    node->fault_from[i] = 0;
  }
  printf("build_lp_tree %zu %d %d\n", size, tree_type, alloc_count);
  //tree type 0은 기본 TBN 및 LC(2MB root, 64K leaf)
  //tree type 1은 LS 트리(4MB root, 256K leaf)
  //tree type 2는 HCHI 트리(512K root, 64K leaf)
  //tree type 3은 HCLI 트리(512K root, 16K leaf)
  //tree type 4는 range(2MB root, 2MB leaf)
  switch(tree_type) {
    case 0: // 기본 TBN 및 LC
      node->root_size = 2 * 1024 * 1024; // 2MB
      node->leaf_size = 64 * 1024;       // 64K
      break;
    case 1: // LS 트리
      node->root_size = 4 * 1024 * 1024;   // 4MB
      node->leaf_size = 256 * 1024;        // 256K
      break;
    case 2: // HCHI 트리
      node->root_size = 512 * 1024;        // 512K
      node->leaf_size = 64 * 1024;         // 64K
      break;
    case 3: // HCLI 트리
      node->root_size = 512 * 1024;        // 512K
      node->leaf_size = 16 * 1024;         // 16K
      break;
    case 4: // range
      node->root_size = 2 * 1024 * 1024;   // 2MB
      node->leaf_size = 2 * 1024 * 1024;   // 2MB
      break;
    default:
      node->root_size = 2 * 1024 * 1024; // 2MB
      node->leaf_size = 64 * 1024;       // 64K
      assert(0 && "default tree type is not supported");
      break;
  }
  if (size == PREFETCH_16K_SIZE) {
    node->left = NULL;
    node->right = NULL;
  } else {
    node->left = build_lp_tree(addr, size / 2,tree_type,alloc_count);
    node->left->parent = node;
    node->right = build_lp_tree(addr + size / 2, size / 2,tree_type,alloc_count);
    node->right->parent = node;
  }
  return node;
}

void gmmu_t::initialize_large_page(mem_addr_t start_addr, size_t size,uint8_t tree_type=0,uint8_t alloc_count=0) {
  struct lp_tree_node *root = build_lp_tree(start_addr, size, tree_type,alloc_count);
  large_page_info.push_back(root);

  total_allocation_size += size;
}

struct lp_tree_node *gmmu_t::get_lp_node(mem_addr_t addr) {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    if ((*iter)->addr <= addr && addr < (*iter)->addr + (*iter)->size) {
      if((*iter)->tree_type == 1&& (*iter)->parent!=NULL){
        return (*iter)->parent;
      }
      // root_size가 노드의 사이즈와 같을 때까지 내려가고, left/right를 주소에 맞게 따라간다
      struct lp_tree_node* node = *iter;
      while (node->root_size < node->size) {
        if (node->left && node->left->addr <= addr && addr < node->left->addr + node->left->size) {
          node = node->left;
        } else if (node->right && node->right->addr <= addr && addr < node->right->addr + node->right->size) {
          node = node->right;
        } else {
          break;
        }
      }
      return node;
      //  return *iter;
    }
  }
  return NULL;
}

mem_addr_t gmmu_t::get_basic_block(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->addr;
}

// In MICRO_barrier_Range mode (only with range tree, tree_logic==2) every eviction_t
// LRU list operation and comp/decomp work-unit is sized at 64KB (MIN_PREFETCH_SIZE) so
// that a 2MB leaf can hold a mix of valid/compressed 64KB blocks without the LRU lists
// over-merging to a uniform 2MB entry. The 2MB barrier only governs the prefetch-in and
// comp-evict spans, which iterate 16KB sub-blocks and are already mixed-state safe.
size_t gmmu_t::micro_leaf(struct lp_tree_node *node) {
  if (m_config.MICRO_barrier_Range && m_config.tree_logic == 2) {
    return MIN_PREFETCH_SIZE;
  }
  return node->leaf_size;
}

void gmmu_t::evict_whole_tree(struct lp_tree_node *node) {
  if (node != NULL) {
    if (compression_policy != compression_policy_struct::NONE) {
      node->compress_size = node->valid_size;
    }
    node->valid_size = 0;
    evict_whole_tree(node->left);
    evict_whole_tree(node->right);
  }
}
void gmmu_t::evict_whole_tree_comp_evict(struct lp_tree_node *node) {
  if (node != NULL && node->compress_size) {
    node->compress_size = 0;
    node->valid_size = 0;
    evict_whole_tree_comp_evict(node->left);
    evict_whole_tree_comp_evict(node->right);
  }
}
mem_addr_t gmmu_t::update_basic_block(
    struct lp_tree_node *node, mem_addr_t addr, size_t size,
    bool prefetch) {  // addr ~ addr+size에 속하는 모든 PREFETCH_16K_SIZE 단위에 대해 처리
  // leaf node인 경우: 16K 단위
  if (node->size == PREFETCH_16K_SIZE) {
    // 이 노드가 addr~addr+size-1 범위에 포함되는지 확인
    if (node->addr + node->size > addr && node->addr < addr + size) {
      if (prefetch) {
        if (node->valid_size != node->size) {
          node->valid_size = PREFETCH_16K_SIZE;
          node->compress_size = 0; // 모든 leaf node는 valid로, predecomp에서 설정
        }
      } else {
        if (node->valid_size != 0) { // leaf node를 invalid로 변경
          node->valid_size = 0;
          node->compress_size = 0;
        }
      }
    }
    return node->addr;
  }

  // 왼쪽, 오른쪽 자식 모두 addr~addr+size-1와 겹치는지 확인해서 재귀적으로 호출
  if (node->left->addr + node->left->size > addr && node->left->addr < addr + size) {
    update_basic_block(node->left, addr, size, prefetch);
  }
  if (node->right->addr + node->right->size > addr && node->right->addr < addr + size) {
    update_basic_block(node->right, addr, size, prefetch);
  }
  node->valid_size = node->left->valid_size + node->right->valid_size;
  node->compress_size = node->left->compress_size + node->right->compress_size;
  return node->addr;
}
mem_addr_t gmmu_t::update_basic_block_comp(
    struct lp_tree_node *node, mem_addr_t addr, size_t size,
    bool prefetch) {  // addr ~ addr+size에 속하는 모든 PREFETCH_16K_SIZE 단위에 대해 처리
  // print_caller_info();
  // addr ~ addr+size-1 범위가 node에 포함되는지 확인
  if (node->size == PREFETCH_16K_SIZE) {
    // 이 노드가 addr~addr+size-1 범위에 포함되는지 확인
    if (node->addr + node->size > addr && node->addr < addr + size) {
      if (prefetch) {  // predecomp
        if (node->valid_size != PREFETCH_16K_SIZE && node->compress_size) {
          node->valid_size = PREFETCH_16K_SIZE;
          node->compress_size = 0;
        }
      } else {  // compression
        if (node->valid_size != 0) {
          node->valid_size = 0;
          node->compress_size = PREFETCH_16K_SIZE;
        }
      }
    }
    return node->addr;
  }

  // 왼쪽, 오른쪽 자식 모두 addr~addr+size-1와 겹치는지 확인해서 재귀적으로 호출
  if (node->left->addr + node->left->size > addr && node->left->addr < addr + size) {
    update_basic_block_comp(node->left, addr, size, prefetch);
  }
  if (node->right->addr + node->right->size > addr && node->right->addr < addr + size) {
    update_basic_block_comp(node->right, addr, size, prefetch);
  }
  node->compress_size = node->left->compress_size + node->right->compress_size;
  node->valid_size = node->left->valid_size + node->right->valid_size;
  return node->addr;
}
//compressed -> invalid, invalid -> compressed
mem_addr_t gmmu_t::update_basic_block_comp_evict(
    struct lp_tree_node *node, mem_addr_t addr, size_t size,
    bool prefetch) {  // addr ~ addr+size에 속하는 모든 PREFETCH_16K_SIZE 단위에 대해 처리
  // print_caller_info();
  // addr ~ addr+size-1 범위가 node에 포함되는지 확인
  if (node->size == PREFETCH_16K_SIZE) {
    // 이 노드가 addr~addr+size-1 범위에 포함되는지 확인
    if (node->addr + node->size > addr && node->addr < addr + size) {
      if (prefetch) {  // predecomp
        if (node->valid_size != PREFETCH_16K_SIZE && !node->compress_size) {
          node->valid_size = 0;
          node->compress_size = PREFETCH_16K_SIZE;
        }
      } else {
        if (node->compress_size != 0) {
          node->compress_size = 0;
          node->valid_size = 0;
        }
      }
    }
    return node->addr;
  }

  // 왼쪽, 오른쪽 자식 모두 addr~addr+size-1와 겹치는지 확인해서 재귀적으로 호출
  if (node->left->addr + node->left->size > addr && node->left->addr < addr + size) {
    update_basic_block_comp_evict(node->left, addr, size, prefetch);
  }
  if (node->right->addr + node->right->size > addr && node->right->addr < addr + size) {
    update_basic_block_comp_evict(node->right, addr, size, prefetch);
  }
  node->compress_size = node->left->compress_size + node->right->compress_size;
  node->valid_size = node->left->valid_size + node->right->valid_size;
  return node->addr;
}
void gmmu_t::fill_lp_tree(struct lp_tree_node *node,
                          std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == PREFETCH_16K_SIZE) {
    if (node->valid_size == 0) {
      check_write_stage_queue(m_gpu->get_global_memory()->get_page_num(node->addr),false);//WB 시행시 valid페이지 리스트 추가, comp_evict시행시 다시 compressed로 변경, 처음접근은 해당사항없음
      node->valid_size = PREFETCH_16K_SIZE;
      scheduled_basic_blocks.insert(node->addr);
      node->compress_size = 0;//prevent comp_evict 시행시 both bit comp and valid is set
    }
  } else {
    fill_lp_tree(node->left, scheduled_basic_blocks);
    fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size = node->left->compress_size + node->right->compress_size;
  }
}

void gmmu_t::remove_lp_tree_comp(struct lp_tree_node *node,
                                 std::vector<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == PREFETCH_16K_SIZE) {
    if (node->valid_size == PREFETCH_16K_SIZE &&
        is_block_evictable(node->addr, PREFETCH_16K_SIZE,false)) {//valid to comp 16K block
      node->valid_size = 0;
      scheduled_basic_blocks.push_back(node->addr);
      node->compress_size = PREFETCH_16K_SIZE;
    }
  } else {
    if (node->size == node->leaf_size && !is_block_evictable(node->addr, node->leaf_size,false)) return ;//before go to 16K node check leaf node is evictable
    remove_lp_tree_comp(node->left, scheduled_basic_blocks);
    remove_lp_tree_comp(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
  }
}
void gmmu_t::remove_lp_tree(struct lp_tree_node *node,
                            std::vector<mem_addr_t> &scheduled_basic_blocks) {// this function only move valid to invalid for without CURE-MEM
  // 리프 노드(16K)에서만 제거 조건 확인
  if (node->size == PREFETCH_16K_SIZE) {
    if (node->valid_size == PREFETCH_16K_SIZE &&
        is_block_evictable(node->addr, PREFETCH_16K_SIZE,false)) {
      node->valid_size = 0;
      scheduled_basic_blocks.push_back(node->addr);
      if (node->compress_size) {
        assert(0 && "bst debug remove_lp_tree compress size is not 0");
        node->compress_size = 0;
      }
    }
    return;
  }
  if (node->size == node->leaf_size && !is_block_evictable(node->addr, node->leaf_size,false)) return ;//before go to 16K node check leaf node is evictable
    // 비리프 노드는 재귀적으로 처리
  remove_lp_tree(node->left, scheduled_basic_blocks);
  remove_lp_tree(node->right, scheduled_basic_blocks);
  node->valid_size = node->left->valid_size + node->right->valid_size;
  node->compress_size = node->left->compress_size + node->right->compress_size;
}
void gmmu_t::fill_lp_tree_comp(struct lp_tree_node *node,
                               std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == PREFETCH_16K_SIZE) {
    if (node->valid_size == 0){
      check_write_stage_queue(m_gpu->get_global_memory()->get_page_num(node->addr),false);
      if(node->compress_size) {  // check compressed leaf tree
        node->valid_size = PREFETCH_16K_SIZE;
        node->compress_size = 0;
        scheduled_basic_blocks.insert(node->addr);
      }
    }
  } else {
    fill_lp_tree_comp(node->left, scheduled_basic_blocks);
    fill_lp_tree_comp(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
  }
}
void gmmu_t::traverse_and_fill_lp_tree_comp(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size > node->leaf_size) {
    traverse_and_fill_lp_tree_comp(node->left, scheduled_basic_blocks);
    traverse_and_fill_lp_tree_comp(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
    if (node->valid_size != node->size &&
        (node->valid_size) >
            node->size /
                (m_config.tree_logic==4?node->prefetch_threshold:tbn_threshold_float)) {  // always consider only valid size
      fill_lp_tree_comp(node, scheduled_basic_blocks);
    }
  }
}
void gmmu_t::update_lp_tree_prefetch_threshold(struct lp_tree_node *node, float prefetch_threshold){
  if(node->size==node->leaf_size){
    node->prefetch_threshold=prefetch_threshold;
  }
  else{
    update_lp_tree_prefetch_threshold(node->left, prefetch_threshold);
    update_lp_tree_prefetch_threshold(node->right, prefetch_threshold);
  }
  node->prefetch_threshold=prefetch_threshold;
}
void gmmu_t::traverse_and_fill_lp_tree(
    struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size > node->leaf_size) {
    traverse_and_fill_lp_tree(node->left, scheduled_basic_blocks);
    traverse_and_fill_lp_tree(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size = node->left->compress_size + node->right->compress_size;
    if (node->valid_size != node->size &&
        node->valid_size > node->size / (m_config.tree_logic==4?node->prefetch_threshold:tbn_threshold_float)) {
      fill_lp_tree(node, scheduled_basic_blocks);
    }
  }
}
void gmmu_t::traverse_and_remove_lp_tree_comp(
    struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size > node->leaf_size) {
    traverse_and_remove_lp_tree_comp(node->left, scheduled_basic_blocks);
    traverse_and_remove_lp_tree_comp(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
    if (node->valid_size != 0 &&
        node->valid_size <
            node->size / (m_config.tree_logic==4?node->prefetch_threshold:tbn_threshold_float)) {  // always remove all trees
      remove_lp_tree_comp(node, scheduled_basic_blocks);
    }
  }
}
void gmmu_t::traverse_and_remove_lp_tree(
    struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size <= node->leaf_size) return;

  // 먼저 자식 노드 순회
  traverse_and_remove_lp_tree(node->left, scheduled_basic_blocks);
  traverse_and_remove_lp_tree(node->right, scheduled_basic_blocks);

  // valid_size, compress_size 갱신
  node->valid_size = node->left->valid_size + node->right->valid_size;
  node->compress_size = node->left->compress_size + node->right->compress_size;

  // 조건에 따라 remove 실행
  if (node->valid_size != 0 &&
      node->valid_size < node->size / (m_config.tree_logic==4?node->prefetch_threshold:tbn_threshold_float) || m_config.tree_logic==1||m_config.fault_based_LRU) {
    remove_lp_tree(node, scheduled_basic_blocks);
  }
}
void gmmu_t::remove_lp_tree_comp_evict(
    struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size == node->leaf_size) {
    if (is_block_comp_evict(node->addr, node->leaf_size)) {
      if (node->valid_size == 0 && node->compress_size) {
        node->compress_size = 0;
        node->valid_size = 0;
        scheduled_basic_blocks.push_back(node->addr);
      }
    }
  } else {
    remove_lp_tree_comp_evict(node->left, scheduled_basic_blocks);
    remove_lp_tree_comp_evict(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
  }
}
void gmmu_t::traverse_and_remove_lp_tree_comp_evict(
    struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks) {
  if (node->size != node->leaf_size) {
    traverse_and_remove_lp_tree_comp_evict(node->left, scheduled_basic_blocks);
    traverse_and_remove_lp_tree_comp_evict(node->right, scheduled_basic_blocks);
    node->valid_size = node->left->valid_size + node->right->valid_size;
    node->compress_size =
        node->left->compress_size + node->right->compress_size;
    if (node->compress_size != 0 &&
        node->valid_size ==
            0) {  // evict all page is compressed and no valid page
      // machine
      remove_lp_tree_comp_evict(node, scheduled_basic_blocks);
    }
  }
}
void gmmu_t::reserve_pages_insert(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);
  if (find(reserve_pages[page_num].begin(), reserve_pages[page_num].end(),
           ma_uid) == reserve_pages[page_num].end()) {
    reserve_pages[page_num].push_back(ma_uid);
  }
}

void gmmu_t::reserve_pages_remove(mem_addr_t addr, unsigned ma_uid) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);
  assert(reserve_pages.find(page_num) != reserve_pages.end());

  std::list<unsigned>::iterator iter = std::find(
      reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid);

  assert(iter != reserve_pages[page_num].end());

  reserve_pages[page_num].erase(iter);

  if (reserve_pages[page_num].empty()) {
    reserve_pages.erase(page_num);
    if(m_config.selective_compression){
      if(!get_lp_node(addr)->inefficient_compression){
        enable_evict();
      }
    }
    else{
      enable_evict();
    }
  }
  /*else{
    std::list<unsigned>::iterator check_iter = std::find(
      reserve_pages[page_num].begin(), reserve_pages[page_num].end(), ma_uid);
    assert(check_iter == reserve_pages[page_num].end());//must be no duplicate
  }*/
}
// find page_num in reserve_pages return true if not found
bool gmmu_t::reserve_pages_check(mem_addr_t addr) {
  mem_addr_t page_num = m_gpu->get_global_memory()->get_page_num(addr);

  return reserve_pages.find(page_num) == reserve_pages.end();
}

/*LCP_CompressedPage gmmu_t::is_LCP_compress_page(mem_addr_t page_no) {
    auto it = LCP_compressed_pages_map.find(page_no);
    if (it != LCP_compressed_pages_map.end()) {
        return it->second;  // 객체 복사 반환
    }
    return LCP_CompressedPage();  // 기본 객체 반환 (찾지 못한 경우)
}*/

void gmmu_t::update_hardware_prefetcher_oversubscribed() {
  if (oversub_prefetcher == hwardware_prefetcher_oversub::DISBALED) {
    prefetcher = hwardware_prefetcher::DISBALED;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::TBN) {
    prefetcher = hwardware_prefetcher::TBN;
  } else if (oversub_prefetcher ==
             hwardware_prefetcher_oversub::SEQUENTIAL_LOCAL) {
    prefetcher = hwardware_prefetcher::SEQUENTIAL_LOCAL;
  } else if (oversub_prefetcher == hwardware_prefetcher_oversub::RANDOM) {
    prefetcher = hwardware_prefetcher::RANDOM;
  }
}

void gmmu_t::log_kernel_info(unsigned kernel_id, unsigned long long time,
                             bool finish) {
  if (!finish) {
    kernel_info.insert(std::make_pair(kernel_id, std::make_pair(time, 0)));
    } else {
    std::map<unsigned,
             std::pair<unsigned long long, unsigned long long>>::iterator it =
        kernel_info.find(kernel_id);
    if (it != kernel_info.end()) {
      it->second.second = time;
    }
  }
}

void gmmu_t::update_memory_management_policy() {
  std::map<std::string, ds_pattern> accessPatterns;

  int i = 1;
  std::map<std::pair<mem_addr_t, size_t>, std::string> dataStructures;
  std::map<std::string, std::list<mem_addr_t>> dsUniqueBlocks;

  // get the managed allocations
  const std::map<uint64_t, struct allocation_info *> &managedAllocations =
      m_gpu->gpu_get_managed_allocations();

  // loop over managed allocations to create three maps
  // 1. data structures - key: pair of start addr and size; value: ds_i
  // 2. access pattern: key: ds_i; value: UNDECIDED pattern
  // 3. unique accessed blocks for reuse: key: ds_i; value: empty list of block
  // start address
  for (std::map<uint64_t, struct allocation_info *>::const_iterator iter =
           managedAllocations.begin();
       iter != managedAllocations.end(); iter++) {
    dataStructures.insert(
        std::make_pair(std::make_pair(iter->second->gpu_mem_addr,
                                      iter->second->allocation_size),
                       std::string("ds" + std::to_string(i))));

    accessPatterns.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         ds_pattern::UNDECIDED));
    dsUniqueBlocks.insert(std::make_pair(std::string("ds" + std::to_string(i)),
                                         std::list<mem_addr_t>()));
    i++;
  }

  // create three level hierarchy for kernel-wise then data-structure wise block
  // address first level: name of kernel (k_i); second level: ds_i; third level:
  // block addresses ordered by access time
  std::map<unsigned, std::map<std::string, std::list<mem_addr_t>>>
      kernel_pattern;

  for (std::map<unsigned,
                std::pair<unsigned long long, unsigned long long>>::iterator
           k_iter = kernel_info.begin();
       k_iter != kernel_info.end(); k_iter++) {
    unsigned long long start = k_iter->second.first;
    unsigned long long end = k_iter->second.second;

    std::map<std::string, std::list<mem_addr_t>> dsAccess;

    for (std::list<std::pair<unsigned long long, mem_addr_t>>::iterator
             acc_iter = block_access_list.begin();
         acc_iter != block_access_list.end(); acc_iter++) {
      unsigned long long access_cycle = acc_iter->first;
      mem_addr_t block_addr = acc_iter->second;

      if (access_cycle >= start && ((end == 0) || (access_cycle <= end))) {
        for (std::map<std::pair<mem_addr_t, size_t>, std::string>::iterator
                 ds_iter = dataStructures.begin();
             ds_iter != dataStructures.end(); ds_iter++) {
          if (block_addr >= ds_iter->first.first &&
              block_addr < ds_iter->first.first + ds_iter->first.second) {
            dsAccess[ds_iter->second].push_back(block_addr);
          }
        }
      }
    }

    kernel_pattern.insert(std::make_pair(k_iter->first, dsAccess));
  }

  // determine pattern per data structure
  // first loop on kernel level then on data structures accessed in that kernel
  for (std::map<unsigned,
                std::map<std::string, std::list<mem_addr_t>>>::iterator k_iter =
           kernel_pattern.begin();
       k_iter != kernel_pattern.end(); k_iter++) {
    for (std::map<std::string, std::list<mem_addr_t>>::iterator da_iter =
             k_iter->second.begin();
         da_iter != k_iter->second.end(); da_iter++) {
      // get the sorted list of block addresses belonging to the current
      // data-structure in current kernel
      std::list<mem_addr_t> curBlocks = std::list<mem_addr_t>(da_iter->second);
      curBlocks.sort();
      curBlocks.unique();

      // check for data reuse
      bool reuse = false;

      // first within this kernel
      // if the number of unique blocks accessed and total number of blocks
      // accessed are not same then there is repetition
      if (curBlocks.size() != da_iter->second.size()) {
        reuse = true;
      }

      // second check if the current accessed blocks are already seen or not
      std::map<std::string, std::list<mem_addr_t>>::iterator ub_it =
          dsUniqueBlocks.find(da_iter->first);

      // check for intersection between unique blocks accessed in current kernel
      // and the previous kernels is null set or not
      std::list<int> intersection;
      std::set_intersection(curBlocks.begin(), curBlocks.end(),
                            ub_it->second.begin(), ub_it->second.end(),
                            std::back_inserter(intersection));

      if (intersection.size() != 0) {
        reuse = true;
      }

      // add the current blocks to the seen set per data structure
      ub_it->second.merge(curBlocks);
      ub_it->second.sort();
      ub_it->second.unique();

      // now update the pattern
      std::map<std::string, ds_pattern>::iterator dsp_it =
          accessPatterns.find(da_iter->first);
      ds_pattern curPattern;

      // check for linearity or randomness in current kernel
      if (std::is_sorted(da_iter->second.begin(), da_iter->second.end())) {
        if (reuse) {
          curPattern = ds_pattern::LINEAR_REUSE;
        } else {
          curPattern = ds_pattern::LINEAR;
        }
      } else {
        if (reuse) {
          curPattern = ds_pattern::RANDOM_REUSE;
        } else {
          curPattern = ds_pattern::RANDOM;
        }
      }

      // determine the pattern
      if (dsp_it->second == ds_pattern::UNDECIDED) {
        dsp_it->second = curPattern;
      } else if (dsp_it->second == ds_pattern::LINEAR) {
        if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::LINEAR_REUSE;
        } else if (curPattern == ds_pattern::RANDOM) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::LINEAR_REUSE) {
        if (curPattern == ds_pattern::RANDOM ||
            curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM) {
        if (curPattern == ds_pattern::RANDOM_REUSE) {
          dsp_it->second = ds_pattern::RANDOM_REUSE;
        } else if (curPattern == ds_pattern::LINEAR) {
          dsp_it->second = ds_pattern::MIXED;
        } else if (curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      } else if (dsp_it->second == ds_pattern::RANDOM_REUSE) {
        if (curPattern == ds_pattern::LINEAR ||
            curPattern == ds_pattern::LINEAR_REUSE) {
          dsp_it->second = ds_pattern::MIXED_REUSE;
        }
      }
    }
  }

  bool is_random = false, is_random_reuse = false, is_linear = false,
       is_linear_reuse = false, is_mixed = false, is_mixed_reuse = false;

  for (std::map<std::string, ds_pattern>::iterator ap_iter =
           accessPatterns.begin();
       ap_iter != accessPatterns.end(); ap_iter++) {
    if (ap_iter->second == ds_pattern::RANDOM) {
      is_random = true;
    } else if (ap_iter->second == ds_pattern::RANDOM_REUSE) {
      is_random_reuse = true;
    } else if (ap_iter->second == ds_pattern::LINEAR) {
      is_linear = true;
    } else if (ap_iter->second == ds_pattern::LINEAR_REUSE) {
      is_linear_reuse = true;
    } else if (ap_iter->second == ds_pattern::MIXED) {
      is_mixed = true;
    } else if (ap_iter->second == ds_pattern::MIXED_REUSE) {
      is_mixed_reuse = true;
    }
  }

  if (is_random || is_random_reuse || is_mixed || is_mixed_reuse) {
    dma_mode = dma_type::OVERSUB;
    evict_policy = eviction_policy::TBN;
  } else if (is_linear_reuse) {
    evict_policy = eviction_policy::TBN;
  }
}

void gmmu_t::reset_lp_tree_node(struct lp_tree_node *node) {
  node->valid_size = 0;
  node->access_counter = 0;
  node->RW = 0;
  node->large_pages_reserved = false;
  node->compress_size = 0;
  node->inefficient_compression = false;
  if (node->size != node->leaf_size) {
    reset_lp_tree_node(node->left);
    reset_lp_tree_node(node->right);
  }
}

void gmmu_t::reset_large_page_info() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    reset_lp_tree_node(*iter);
  }

  over_sub = false;
}

mem_addr_t gmmu_t::get_eviction_base_addr(mem_addr_t page_addr) {
  mem_addr_t lru_addr;

  struct lp_tree_node *root = get_lp_node(page_addr);
  if (m_config.tbn_threshold == 1) {
    lru_addr = root->addr;
  } else if (evict_policy == eviction_policy::TBN ||
             evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_addr = get_basic_block(root, page_addr);
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_addr = page_addr;
  } else {
    lru_addr = root->addr;
  }

  return lru_addr;
}

size_t gmmu_t::get_eviction_granularity(mem_addr_t page_addr, unsigned int leaf_size) {
  size_t lru_size;

  struct lp_tree_node *root = get_lp_node(page_addr);
  if (m_config.tbn_threshold == 1) {
    lru_size = root->size;
  } else if (evict_policy == eviction_policy::TBN ||
             evict_policy == eviction_policy::SEQUENTIAL_LOCAL) {
    lru_size = leaf_size;
  } else if (evict_policy == eviction_policy::LRU4K ||
             evict_policy == eviction_policy::RANDOM) {
    lru_size = m_config.page_size;
  } else {
    lru_size = root->size;
  }

  return lru_size;
}

void gmmu_t::update_access_type(mem_addr_t addr, int type) {
  struct lp_tree_node *node = get_lp_node(addr);

  while (node->size != PREFETCH_16K_SIZE) {
    node->RW |= type;
    if (type == 2) {  // if it is a write, then it is not sampled
      node->sampled = false;
    }
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  node->RW |= type;
}
bool gmmu_t::is_sampled(mem_addr_t addr) {
  struct lp_tree_node *node = get_lp_node(addr);
  while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  return node->sampled;
}
size_t gmmu_t::get_tree_comp_size(mem_addr_t addr) {
  struct lp_tree_node *node = get_lp_node(addr);
  size_t ret_comp_size = 0;
  while (node->size != MIN_PREFETCH_SIZE) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if(m_config.static_comp_ratio != 0.0){
    return node->size/m_config.static_comp_ratio;
  }
  ret_comp_size = node->comp_size;

  return ret_comp_size;
}

void gmmu_t::set_tree_comp_size(mem_addr_t addr, size_t comp_size) {
  struct lp_tree_node *node = get_lp_node(addr);
  while (node->size != MIN_PREFETCH_SIZE) {//we only consider node comp size as 64KB(MIN_PREFETCH_SIZE) granularity
    node->comp_size = comp_size;
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  if(m_config.static_comp_ratio != 0.0){
    node->comp_size = comp_size*m_config.static_comp_ratio;
  }
  else{
    node->comp_size = comp_size;
  }
}

void gmmu_t::set_sampled(mem_addr_t addr) {
  struct lp_tree_node *node = get_lp_node(addr);
  while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  node->sampled = true;
}
void gmmu_t::clear_sampled(mem_addr_t addr) {
  struct lp_tree_node *node = get_lp_node(addr);
    while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }
  node->sampled = false;
}
int gmmu_t::get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return node->access_counter & ((1 << 27) - 1);
}

int gmmu_t::get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr) {
  while (node->size > node->leaf_size) {
    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  return (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
}

void gmmu_t::inc_bb_access_counter(mem_addr_t addr) {
  struct lp_tree_node *node = get_lp_node(addr);

  while (node->size > node->leaf_size) {
    node->access_counter++;

    if (node->left->addr <= addr &&
        addr < node->left->addr + node->left->size) {
      node = node->left;
    } else {
      node = node->right;
    }
  }

  if (node->access_counter == ((1 << 27) - 1)) {
    reset_bb_access_counter();
  }

  node->access_counter++;
}

void gmmu_t::inc_bb_round_trip(struct lp_tree_node *node) {
  if (node->size != node->leaf_size) {
    inc_bb_round_trip(node->left);
    inc_bb_round_trip(node->right);
  } else {
    uint16_t round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;

    if (round_trip == ((1 << 6) - 1)) {
      reset_bb_round_trip();
    }

    round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    round_trip++;

    node->access_counter =
        (round_trip << 27) | (node->access_counter & ((1 << 27) - 1));
  }
}

void gmmu_t::traverse_and_reset_access_counter(struct lp_tree_node *node) {
    if (node->size == node->leaf_size) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 27;
    int access_counter = (node->access_counter & ((1 << 27) - 1)) >> 1;

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
    node->access_counter = node->access_counter >> 1;
  }
}

void gmmu_t::reset_bb_access_counter() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_access_counter(*iter);
  }
}

void gmmu_t::traverse_and_reset_round_trip(struct lp_tree_node *node) {
  if (node->size == node->leaf_size) {
    int round_trip = (node->access_counter & (((1 << 6) - 1) << 27)) >> 28;
    int access_counter = node->access_counter & ((1 << 27) - 1);

    node->access_counter = (round_trip << 27) | access_counter;
  } else {
    traverse_and_reset_access_counter(node->left);
    traverse_and_reset_access_counter(node->right);
  }
}

void gmmu_t::reset_bb_round_trip() {
  for (std::list<struct lp_tree_node *>::iterator iter =
           large_page_info.begin();
       iter != large_page_info.end(); iter++) {
    traverse_and_reset_round_trip(*iter);
  }
}

bool gmmu_t::should_cause_page_migration(mem_addr_t addr, bool is_write) {
  if (dma_mode == dma_type::DISABLED) {
    return true;
  } else if (dma_mode == dma_type::ALWAYS) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = get_lp_node(addr);
      return get_bb_access_counter(root, addr) >= m_config.migrate_threshold;
    }
  } else if (dma_mode == dma_type::OVERSUB) {
    if (over_sub) {
      if (is_write) {
        return true;
      } else {
        struct lp_tree_node *root = get_lp_node(addr);
        return get_bb_access_counter(root, addr) >= m_config.migrate_threshold;
      }
    } else {
      return true;
    }
  } else if (dma_mode == dma_type::ADAPTIVE) {
    if (is_write) {
      return true;
    } else {
      struct lp_tree_node *root = get_lp_node(addr);
      int derived_threshold;

      if (over_sub) {
        derived_threshold = m_config.migrate_threshold *
                            m_config.multiply_dma_penalty *
                            (get_bb_round_trip(root, addr) + 1);
      } else {
        derived_threshold = static_cast<int>(
            1.0 + m_config.migrate_threshold *
                      m_gpu->get_global_memory()->get_projected_occupancy(
                          num_read_stage_queue, num_write_stage_queue,
                          m_config.free_page_buffer_percentage));
      }

      return get_bb_access_counter(root, addr) >= derived_threshold;
    }
  }
}
// Process a completed PCIe write latency queue.
void gmmu_t::processWriteBackLatencyQueue() {
  if (pcie_write_latency_queue &&
      (cur_cycle >= pcie_write_latency_queue->ready_cycle)) {
    auto globalMem = m_gpu->get_global_memory();
    need_throttle_SM = 1;
    bst_debug_print(
        "[DEBUG] Processing completed write latency queue at cycle %llu\n",
        cur_cycle);

    // Write back each page.
    for (const auto &page : pcie_write_latency_queue->page_list) {
      auto page_addr = globalMem->get_mem_addr(page);
      bst_debug_print("[DEBUG] Writing back page %lx\n", page_addr);
      m_gpu->gpu_writeback(page_addr);
    }
    if(m_config.tree_logic == 1){//TBN 2MB eviction
      if(m_config.proactive_WB!=1){//if no proactive WB, after page eviction, free pages
        num_write_stage_queue -= pcie_write_latency_queue->page_list.size();
        if(get_lp_node(globalMem->get_mem_addr(pcie_write_latency_queue->page_list.front()))->large_pages_reserved){
          lp_tree_node *cur_root = get_lp_node(globalMem->get_mem_addr(pcie_write_latency_queue->page_list.front()));
          bool same_node_in_stage_queue = false;
          if(pcie_write_stage_queue.size() > 0){
            for(const auto& entry : pcie_write_stage_queue){
              lp_tree_node *entry_root = get_lp_node(globalMem->get_mem_addr(entry->page_list.front()));
              if(entry_root == cur_root){
                same_node_in_stage_queue = true;
                break;
              }
            }
          }
          if(!same_node_in_stage_queue){
          globalMem->free_pages(512);
          get_lp_node(globalMem->get_mem_addr(pcie_write_latency_queue->page_list.front()))->large_pages_reserved = false;
          }
        }
      }
    }
    else if(m_config.proactive_WB!=1){//if no_proactive WB, after page eviction, free pages
      globalMem->free_pages(pcie_write_latency_queue->page_list.size());
      if (pcie_write_latency_queue->type == latency_type::COMP_EVICT ||
        pcie_write_latency_queue->type == latency_type::COMP_INV_EVICT) {
          bst_debug_print(
            "[DEBUG] Updating write stage queue size for compressed pages\n");
           num_write_stage_queue -= pcie_write_latency_queue->comp_size/4096;
           num_comp_evict_stage_queue -= pcie_write_latency_queue->page_list.size();
        } else {
          bst_debug_print(
           "[DEBUG] Updating write stage queue size for normal pages\n");
          num_write_stage_queue -= pcie_write_latency_queue->page_list.size();
        }
    }
    // Update simulation profiling stats if enabled.
    if (sim_prof_enable && !pcie_write_latency_queue->page_list.empty()) {
      auto front_addr =
          globalMem->get_mem_addr(pcie_write_latency_queue->page_list.front());
      for (auto it = writeback_stats.begin(); it != writeback_stats.end();
           ++it) {
        memory_stats *stats = static_cast<memory_stats *>(*it);
        if (stats->start_addr == front_addr) {
          stats->end_time = cur_cycle;
          sim_prof[stats->start_time].push_back(stats);
          writeback_stats.erase(it);
          bst_debug_print("[DEBUG] Updated writeback stats for page %lx\n",
                          front_addr);
          break;
        }
      }
    }

    
    pcie_write_latency_queue = nullptr;
  }
}
double gmmu_t::get_total_compressedsize(){
  double total_compressedsize = 0;
  for(auto iter = compressed_pages.begin(); iter != compressed_pages.end(); iter++){
    total_compressedsize += (*iter)->size;
  }
  for(auto iter = comp_evict_pages.begin(); iter != comp_evict_pages.end(); iter++){
    total_compressedsize += (*iter)->size;
  }
  if(comp_latency_queue.size()>0){
    for(auto iter = comp_latency_queue.begin(); iter != comp_latency_queue.end(); iter++){
      total_compressedsize += (*iter)->size;
    }
  }
  for(auto iter = decompressing_pages.begin(); iter != decompressing_pages.end(); iter++){
    total_compressedsize += (*iter)->size;
  }
  return total_compressedsize/4096;
}
// Schedule a new PCIe write back transfer from the staging queue.
void gmmu_t::scheduleWriteBackTransfer() {
  if (!pcie_write_stage_queue.empty() && !pcie_write_latency_queue) {
    if (cur_cycle < pcie_write_stage_queue.front()->crm_ready_cycle) return;
    need_throttle_SM = 1;
    bst_debug_print("[DEBUG] Scheduling new write transfer from stage queue\n");
    pcie_write_latency_queue = pcie_write_stage_queue.front();
    auto &queue = pcie_write_latency_queue;
    // 최적화: globalMem을 함수 내에서 한 번만 가져옴
    auto globalMem = m_gpu->get_global_memory();
    // Set ready_cycle based on the queue type.
    if (queue->type == latency_type::PCIE_WRITE_BACK ||
        queue->type == latency_type::INVALIDATE) {
      queue->ready_cycle = get_ready_cycle(queue->page_list.size());
    } else if (queue->type == latency_type::COMP_EVICT) {
      // if evict_form is true, then evict compressed pages do not need to be
      // decompressed
      if (!evict_form) {
        // 압축된 페이지 수 계산
        int num_compressed_pages = 0;
        for (auto page : queue->page_list) {
          if (globalMem->is_iter_compressed(page)) {
            num_compressed_pages++;
          }
        }
        if (num_compressed_pages > 0) {
          queue->ready_cycle = get_ready_cycle_with_decomp(
              queue->page_list.size(), queue->decomp_band,
              num_compressed_pages);
        } else {
          queue->ready_cycle =
              get_ready_cycle(queue->page_list.size());
        }
      } else {
        queue->ready_cycle = get_ready_cycle_bytes(queue->comp_size);
      }
    }

    // Update PCIe utilization profiling if enabled.
    if (util_prof) {
      for (unsigned long long period = cur_cycle; period < queue->ready_cycle;
           ++period) {
        m_new_stats->pcie_write_utilization.push_back(std::make_pair(
            period, get_pcie_utilization(queue->page_list.size())));
      }
    }

    bst_debug_print("[DEBUG] Page compression size: %llu\n", queue->comp_size);
    if (queue->type == latency_type::COMP_EVICT ||
               queue->type == latency_type::COMP_INV_EVICT) {
        if (!globalMem->change_size_comp_evict_with_comp_size(queue->size, queue->comp_size,true)) {
          bst_debug_print("[DEBUG]ERROR: Failed to free compressed pages\n");
          printf("bst debug processCompletedWriteLatencyQueue\n");
          printf("bst debug queue->comp_size = %llu\n", queue->comp_size);
          printf("bst debug queue->page_list.size() = %zu\n",
                 queue->page_list.size());
          printf("comp_evict_free_pages\n");
          print_debug_all();
        }
    }
    else{
      if(queue->type == latency_type::PCIE_WRITE_BACK || queue->type == latency_type::INVALIDATE){
        if(m_config.tree_logic == 1){//TBN 2MB eviction
          if(m_config.proactive_WB==1){//if proactive WB, after page eviction, free pages
          if(get_lp_node(globalMem->get_mem_addr(queue->page_list.front()))->large_pages_reserved){
            lp_tree_node *cur_root = get_lp_node(globalMem->get_mem_addr(pcie_write_latency_queue->page_list.front()));
            bool same_node_in_stage_queue = false;
            if(pcie_write_stage_queue.size() > 0){
              for(const auto& entry : pcie_write_stage_queue){
                // front는 자기자신이므로 제외
                if(entry == pcie_write_latency_queue) continue;
                lp_tree_node *entry_root = get_lp_node(globalMem->get_mem_addr(entry->page_list.front()));
                if(entry_root == cur_root){
                  same_node_in_stage_queue = true;
                  break;
                }
              }
            }
            if(!same_node_in_stage_queue){
              globalMem->free_pages(512);
              get_lp_node(globalMem->get_mem_addr(queue->page_list.front()))->large_pages_reserved = false;
            }
          }
          }
          queue->ready_cycle += m_config.real_WB_delay;
        }
        else{
          if(m_config.proactive_WB==1){//if proactive WB, after page eviction, free pages
            globalMem->free_pages(queue->page_list.size());
          }
          queue->ready_cycle += m_config.real_WB_delay;
        }
      }
    }
    // Process each page in the latency queue.
    for (const auto &page : queue->page_list) {
      m_new_stats->page_thrashing[page].push_back(false);

      if (globalMem->is_page_dirty(page)) {
        m_new_stats->page_evict_dirty++;
        bst_debug_print("[DEBUG] Evicting dirty page %lx\n", page);
      } else {
        m_new_stats->page_evict_not_dirty++;
        bst_debug_print("[DEBUG] Evicting clean page %lx\n", page);
      }

      if (evict_form && queue->type == latency_type::COMP_EVICT) {
        globalMem->set_evicted_compressed(page);
        // A new host-resident lifetime needs a new restore control reservation.
        crm_host_restore_ready.erase(globalMem->get_mem_addr(page) /
                                     MIN_PREFETCH_SIZE);
        // clear_sampled(globalMem->get_mem_addr(page));
      }
      globalMem->clear_page_dirty(page);
      globalMem->clear_page_access(page);
      globalMem->invalidate_page(page);
      if (queue->type == latency_type::PCIE_WRITE_BACK ||
          queue->type == latency_type::INVALIDATE) {
        evicting_pages_erase(page, queue->leaf_size);
        bst_debug_print("[DEBUG] Normal eviction for page %lx\n", page);
        auto page_addr = globalMem->get_mem_addr(page);
        if (m_config.compression_engine == 0) {
          //globalMem->free_pages(1);
          //}
        } else {
          auto it = LCP_compressed_pages_map.find(page);
          if (it != LCP_compressed_pages_map.end()) {
            globalMem->free_pages(it->second.allocatedPageSize / 4096);
          } else {
            printf("bst debug processCompletedWriteLatencyQueue\n");
            printf("bst debug page = %lx\n", page);
            printf(
                "bst debug compression_engine is on but page is not find in "
                "LCP_compressed_pages_map\n");
            print_debug_all();
            assert(0 &&
                   "bst debug compression_engine is on but page is not find in "
                   "LCP_compressed_pages_map");
          }
        }
      }
      // Handle compression eviction.
      else if (queue->type == latency_type::COMP_EVICT ||
               queue->type == latency_type::COMP_INV_EVICT) {
        bst_debug_print("[DEBUG] Compression eviction for page %lx\n", page);
        if (!globalMem->is_iter_compressed(page)) {
          bst_debug_print(
              "[DEBUG]ERROR: Attempting to evict uncompressed page %lx\n",
              page);
          print_debug_all();
          assert(0 && "attempt to evict an uncompressed page");
        }
        globalMem->set_iter_decompressed(page);

        comp_evict_erase(page, queue->leaf_size);
        auto page_addr = globalMem->get_mem_addr(page);
      }
      tlb_flush(page);
    }
    if(m_config.compression_engine!=0){
      assert(0 && "compression_engine is bst need to implement");
      //m_gpu->get_dram()->m_dram_BPC_decomp_queue_delay = max(m_gpu->get_dram()->m_dram_BPC_decomp_queue_delay,cur_cycle)+m_config.compress_latency*queue->page_list.size();
      //queue->ready_cycle = m_gpu->get_dram()->m_dram_BPC_decomp_queue_delay + get_ready_cycle_transfer(queue->page_list.size());
    }
    // Increase round-trip count for the first page.
    auto front_addr = globalMem->get_mem_addr(queue->page_list.front());
    struct lp_tree_node *root = get_lp_node(front_addr);
    inc_bb_round_trip(root);

    // Create event stats based on profiling and invalidate clean configuration.
    if (sim_prof_enable) {
      if (m_config.invalidate_clean) {
        if (queue->type == latency_type::INVALIDATE) {
          bst_debug_print("[DEBUG] Creating invalidate event stats\n");
          event_stats *inv =
              new memory_stats(invalidate, cur_cycle, cur_cycle, front_addr,
                               queue->page_list.size() * m_config.page_size, 0);
          sim_prof[inv->start_time].push_back(inv);
        } else if (queue->type == latency_type::COMP_INV_EVICT) {
                  m_new_stats->comp_evcit_num++;

          m_new_stats->comp_evict_clean++;
          bst_debug_print(
              "[DEBUG] Creating compression evict invalidate event stats\n");
          event_stats *inv =
              new memory_stats(comp_evict_inv, cur_cycle, cur_cycle, front_addr,
                               queue->page_list.size() * m_config.page_size, 0);
          sim_prof[inv->start_time].push_back(inv);
        }
      } else {
        if (queue->type == latency_type::INVALIDATE) {
          bst_debug_print("[DEBUG] Creating invalidate event stats\n");
          event_stats *inv =
              new memory_stats(invalidate, cur_cycle, cur_cycle, front_addr,
                               queue->page_list.size() * m_config.page_size, 0);
          writeback_stats.push_back(inv);
        } else if (queue->type == latency_type::COMP_INV_EVICT) {
          bst_debug_print(
              "[DEBUG] Creating compression evict invalidate event stats\n");
          event_stats *inv =
              new memory_stats(comp_evict_inv, cur_cycle, cur_cycle, front_addr,
                               queue->page_list.size() * m_config.page_size, 0);
          writeback_stats.push_back(inv);
        }
      }
      if (queue->type == latency_type::PCIE_WRITE_BACK) {
        bst_debug_print("[DEBUG] Creating write back event stats\n");
        event_stats *wb =
            new memory_stats(write_back, cur_cycle, front_addr,
                             queue->page_list.size() * m_config.page_size, 0);
        writeback_stats.push_back(wb);
      } else if (queue->type == latency_type::COMP_EVICT) {
                m_new_stats->comp_evcit_num++;

        m_new_stats->comp_evict_dirty++;
        bst_debug_print("[DEBUG] Creating compression evict event stats\n");
        event_stats *wb =
            new memory_stats(comp_evict_stat, cur_cycle, front_addr,
                             queue->page_list.size() * m_config.page_size, 0);
        writeback_stats.push_back(wb);
      }
    }
    pcie_write_stage_queue.pop_front();

    // If invalidate clean is enabled, clear the latency queue after processing.
    if (m_config.invalidate_clean) {
      if (queue->type == latency_type::INVALIDATE ||
          queue->type == latency_type::COMP_INV_EVICT) {
        bst_debug_print(
            "[DEBUG] Clearing write latency queue after invalidate\n");
        if (queue->type == latency_type::COMP_INV_EVICT) {
          bst_debug_print(
              "[DEBUG] Updating write stage queue size for compressed pages\n");
          num_write_stage_queue -= queue->comp_size/4096;
          num_comp_evict_stage_queue -= queue->page_list.size();
        } else {
          bst_debug_print(
              "[DEBUG] Updating write stage queue size for normal pages\n");
          num_write_stage_queue -= queue->page_list.size();
        }
        for (const auto &page : queue->page_list) {
          auto page_addr = globalMem->get_mem_addr(page);
          bst_debug_print("[DEBUG] Writing back page %lx\n", page_addr);
          m_gpu->gpu_writeback(page_addr);
        }
        pcie_write_latency_queue = nullptr;
      }
      else{
        if (pcie_write_latency_queue->type == latency_type::COMP_EVICT ||
          pcie_write_latency_queue->type == latency_type::COMP_INV_EVICT) {
            bst_debug_print(
              "[DEBUG] Updating write stage queue size for compressed pages\n");
             num_write_stage_queue -= pcie_write_latency_queue->comp_size/4096;
             num_comp_evict_stage_queue -= pcie_write_latency_queue->page_list.size();
          } else {
            bst_debug_print(
             "[DEBUG] Updating write stage queue size for normal pages\n");
            num_write_stage_queue -= pcie_write_latency_queue->page_list.size();
          }
      }
    }
    else if(m_config.proactive_WB==1){
      // Update write stage queue size.
      if (pcie_write_latency_queue->type == latency_type::COMP_EVICT ||
      pcie_write_latency_queue->type == latency_type::COMP_INV_EVICT) {
        bst_debug_print(
          "[DEBUG] Updating write stage queue size for compressed pages\n");
         num_write_stage_queue -= pcie_write_latency_queue->comp_size/4096;
         num_comp_evict_stage_queue -= pcie_write_latency_queue->page_list.size();
      } else {
        bst_debug_print(
         "[DEBUG] Updating write stage queue size for normal pages\n");
        num_write_stage_queue -= pcie_write_latency_queue->page_list.size();
      }
    }
  }
}
// 완료된 압축 대기 큐(comp_latency_queue)를 처리하는 함수
void gmmu_t::processCompressionLatencyQueue() {
  while (!comp_latency_queue.empty() &&
         cur_cycle >= comp_latency_queue.front()->ready_cycle) {
          need_throttle_SM = 1;
    m_new_stats->compress++;
    auto currentCompLatency = comp_latency_queue.front();
    auto globalMem = m_gpu->get_global_memory();
    //num_comp_stage_queue -= currentCompLatency->page_list.size();
    // 각 페이지에 대해 write-back 수행
    bool efffective_compression = true;
    lp_tree_node *root = get_lp_node(currentCompLatency->start_addr);
    for (const auto &page : currentCompLatency->page_list) {
      auto pageAddr = globalMem->get_mem_addr(page);
      bst_debug_print("[DEBUG] Writing back page %lx\n", pageAddr);
      if (m_config.selective_compression) {
        if (currentCompLatency->comp_size >
            currentCompLatency->size/m_config.selective_compression_threshold || root->inefficient_compression|| root->leaf_will_be_inefficient) {  // inefficient compression
          m_new_stats->inefficient_comp++;
          cancel_compressing_pages(page, currentCompLatency->leaf_size,true);
          update_basic_block_comp(root, pageAddr, m_config.page_size,true);
          efffective_compression = false;
          root->inefficient_compression = true;
        }
      }
      m_gpu->gpu_writeback(pageAddr);
    }
    if (!efffective_compression) {
      if(!m_gpu->get_global_memory()->change_size_comp_with_comp_size(currentCompLatency->size-currentCompLatency->comp_size, false)) {
        printf("bst debug processCompressionLatencyQueue !effective compression change_size_comp_with_comp_size failed\n");
        print_debug_all(true);
      }
      if (sim_prof_enable) {
        for (std::list<event_stats *>::iterator comp_iter =
                 compress_stats.begin();
             comp_iter != compress_stats.end(); comp_iter++) {
          if (((memory_stats *)(*comp_iter))->start_addr ==
              m_gpu->get_global_memory()->get_mem_addr(
                  currentCompLatency->page_list.front())) {
            compress_stats.erase(comp_iter);
            break;
          }
        }
      }
      comp_latency_queue.pop_front();
      continue;
    }
    bst_debug_print("[DEBUG] Processing completed compression latency queue\n");

    // 각 페이지에 대해 압축 및 페이지 상태 업데이트
    for (const auto &page : currentCompLatency->page_list) {
      bst_debug_print("[DEBUG] Compressing page %lx\n", page);
      if (globalMem->is_iter_compressed(page)) {
        printf("bst debug processCompletedCompressionLatencyQueue\n");
        printf("bst debug page = %llu\n", page);
        printf("bst debug page_list.size() = %zu\n",
               currentCompLatency->page_list.size());
        printf("bst debug comp_latency_queue.size() = %zu\n",
               comp_latency_queue.size());
        printf("bst debug comp compressed page\n");
        print_debug_all(true);
      }
      globalMem->set_iter_compressed(page);
      compressing_pages_done(page, currentCompLatency->leaf_size);
      if (!globalMem->is_valid(page)) {
        printf("bst debug processCompletedCompressionLatencyQueue\n");
        printf("bst debug page = %llu\n", page);
        printf("bst debug page_list.size() = %zu\n",
               currentCompLatency->page_list.size());
        printf("bst debug comp_latency_queue.size() = %zu\n",
               comp_latency_queue.size());
        printf("bst debug comp invalid page\n");
        print_debug_all(true);
        fflush(stdout);
      }
      globalMem->invalidate_page(page);
      // globalMem->clear_page_dirty(page);
      globalMem->clear_page_access(page);
    }

    // 시뮬레이션 프로파일링이 활성화된 경우, 해당 이벤트의 종료 시간을 업데이트
    if (sim_prof_enable && !currentCompLatency->page_list.empty()) {
      auto frontAddr =
          globalMem->get_mem_addr(currentCompLatency->page_list.front());
      for (auto iter = compress_stats.begin(); iter != compress_stats.end();
           ++iter) {
        if (static_cast<memory_stats *>(*iter)->start_addr == frontAddr) {
          event_stats *wb = *iter;
          wb->end_time = cur_cycle;
          sim_prof[wb->start_time].push_back(wb);
          compress_stats.erase(iter);
          bst_debug_print("[DEBUG] Updated compression stats for page %lx\n",
                          frontAddr);
          break;
        }
      }
    }
    bst_debug_print("[DEBUG] Updated compression stage queue size\n");
    comp_latency_queue.pop_front();
  }
}

// 대기 중인 압축 작업(comp_stage_queue)을 압축 대기 큐(comp_latency_queue)로
// 이동시키는 함수
void gmmu_t::scheduleCompressionTasks() {
  // 최적화: 자주 사용되는 값들을 미리 계산
  const bool selective_compression_enabled = m_config.selective_compression;
  auto globalMem = m_gpu->get_global_memory();
  
  const size_t capacity = max_parallel_comp_requests + get_memory_stalled_sm_count();
  const size_t reserved = failed_read_count > 0 ? 0 : decomp_stage_queue.size();
  const size_t comp_limit = capacity > reserved ? capacity - reserved : 0;
  while (!comp_stage_queue.empty() &&
         comp_latency_queue.size() + decomp_latency_queue.size() < comp_limit) {
    bst_debug_print("[DEBUG] Found pending compression task in stage queue\n");
    auto currentCompQueue = comp_stage_queue.front();
    if (max(currentCompQueue->ready_cycle, currentCompQueue->crm_ready_cycle) > cur_cycle) {
      return;
    }
    need_throttle_SM = 1;
    for(auto page : currentCompQueue->page_list){
      tlb_flush(page);
    }
    if (selective_compression_enabled) {
      lp_tree_node *root = get_lp_node(currentCompQueue->start_addr);
      if (root->inefficient_compression) {
        //m_new_stats->inefficient_comp++;
        for (auto page : currentCompQueue->page_list) {
          cancel_compressing_pages(page, micro_leaf(root));
          update_basic_block_comp(root, globalMem->get_mem_addr(page),
                                  m_config.page_size, true);
        }
        num_comp_stage_queue -= (currentCompQueue->size-currentCompQueue->comp_size)/m_config.page_size;
        comp_stage_queue.pop_front();
        bst_debug_print("[DEBUG] Removed compression task from stage queue\n");
        continue;
      }
      else if (currentCompQueue->comp_size >
               currentCompQueue->size / m_config.selective_compression_threshold) {
        root->inefficient_compression = true;
      }
    }
    // 압축 대기 큐에 추가하고 ready_cycle 계산
    
    comp_latency_queue.push_back(currentCompQueue);
    currentCompQueue->ready_cycle = get_ready_cycle_pages_compress(
        currentCompQueue->page_list.size(), currentCompQueue->comp_band);

    float pageCompRatio = currentCompQueue->comp_ratio;
    bst_debug_print(
        "[DEBUG] Compression ready cycle: %llu, compression ratio: %f\n",
        currentCompQueue->ready_cycle, pageCompRatio);

    // 압축 페이지 공간 할당 시도
    bst_debug_print("[DEBUG] Allocating compressed page space for page %lx\n",
                    currentCompQueue->start_addr);
    if (!globalMem->change_size_comp_with_comp_size(currentCompQueue->size-currentCompQueue->comp_size, true)) {
      bst_debug_print(
          "[DEBUG]RROR] Failed to allocate compressed page space\n");
      printf("bst debug schedulePendingCompressionTasks\n");
      printf("bst debug size %llu comp_size %llu comp_ratio %f\n", currentCompQueue->size, currentCompQueue->comp_size, pageCompRatio);
      printf("bst debug page_list.size() = %zu\n",
             currentCompQueue->page_list.size());
      printf("bst debug comp_stage_queue.size() = %zu\n",
             comp_stage_queue.size());
      printf("bst debug comp_latency_queue.size() = %zu\n",
             comp_latency_queue.size());
      printf("bst debug comp_free_pages error\n");
      print_debug_all();
    }
    m_new_stats->comp_size_list.push_back(currentCompQueue->comp_size);

    // 시뮬레이션 프로파일링 기록
    if (sim_prof_enable) {
      bst_debug_print("[DEBUG] Recording compression stats\n");
      event_stats *wb = new memory_stats(
          comp, cur_cycle,
          globalMem->get_mem_addr(currentCompQueue->page_list.front()),
          currentCompQueue->page_list.size() * m_config.page_size, 0);
      compress_stats.push_back(wb);
    }
    // 압축 스테이지 큐의 크기를 업데이트
    num_comp_stage_queue -= (currentCompQueue->size-currentCompQueue->comp_size)/m_config.page_size;
    comp_stage_queue.pop_front();
    bst_debug_print("[DEBUG] Removed compression task from stage queue\n");
  }
  if (comp_latency_queue.size() >
      maximum_used_compression_unit) {  // update maximum used compression unit
    maximum_used_compression_unit = comp_latency_queue.size();
  }
}
// 처리 완료된 PCIe read 전송을 분기하여 처리하는 함수
void gmmu_t::processReadLatencyQueue() {
  if (!pcie_read_latency_queue)// pcie_read_latency_queue가 비어있으면 종료
  {
    return;
  }
  fault_occured = true;
  if (cur_cycle < pcie_read_latency_queue->ready_cycle)// 현재 시간이 전송 준비 완료 시간보다 전이면 종료
  {
    return;
  }
  auto globalMem = m_gpu->get_global_memory();
  bst_debug_print("[DEBUG] Processing completed PCIe read at cycle %llu\n",
                  cur_cycle);
  if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {
    for (auto page : pcie_read_latency_queue->page_list) {
      bst_debug_print("[DEBUG] Processing page %lx\n", page);
      globalMem->validate_page(page);
      globalMem->clear_evicted_compressed(page);
      refresh_valid_pages(globalMem->get_mem_addr(page));
      m_new_stats->page_thrashing[page].push_back(true);

      // prefetch 요청 처리
      if (!prefetch_req_buffer.empty()) {
        prefetch_req &pre_q = prefetch_req_buffer.front();
        auto it = std::find(pre_q.pending_prefetch.begin(),
                            pre_q.pending_prefetch.end(), page);
        if (it != pre_q.pending_prefetch.end()) {
          bst_debug_print("[DEBUG] Checking prefetch buffer for page %lx\n",
                          page);
          bst_debug_print("[DEBUG] Found page in prefetch buffer\n");
          pre_q.pending_prefetch.erase(it);
                      pre_q.outgoing_replayable_nacks[page].merge(req_info[page]);
            // SM stall 해제 (req_info에서 제거되는 요청들의 SM들을 추적)
            for (auto mf : req_info[page]) {
              unsigned sm_id = mf->get_sid() / m_config.num_core_per_cluster();
              memory_stalled_sms.remove_stalled_sm(sm_id);
            }
            req_info.erase(page);
          skip_cycles = false;
          m_new_stats->pf_page_fault_latency[page].back() =
              cur_cycle - m_new_stats->pf_page_fault_latency[page].back();
          bst_debug_print("[DEBUG] Updated prefetch stats for page %lx\n",
                          page);
        }
      }

      // 페이지 폴트 요청 처리 (prefetch가 아닌 경우)
      if (req_info.find(page) != req_info.end()) {
        bst_debug_print("[DEBUG] Processing page fault request for page %lx\n",
                        page);
        for (auto mf : req_info[page]) {
          int simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
          bst_debug_print("[DEBUG] Replaying memory fetch to cluster %d\n",
                          simt_cluster_id);
          m_gpu->getSIMTCluster(simt_cluster_id)->push_gmmu_cu_queue(mf);
          // SM stall 해제
          memory_stalled_sms.remove_stalled_sm(simt_cluster_id);
        }
        req_info.erase(page);
        skip_cycles = false;
        m_new_stats->mf_page_fault_latency[page].back() =
            cur_cycle - m_new_stats->pf_page_fault_latency[page].back();
        bst_debug_print("[DEBUG] Updated page fault stats for page %lx\n",
                        page);
      }
    }
  } else if (pcie_read_latency_queue->type == latency_type::PAGE_FAULT) {
    bst_debug_print("[DEBUG] Processing completed page fault\n");
    if (sim_prof_enable) {
      for (auto it = fault_stats.begin(); it != fault_stats.end(); ++it) {
        if (((page_fault_stats *)(*it))->transfering_pages.front() ==
            pcie_read_latency_queue->page_list.front()) {
          event_stats *mf_fault = *it;
          mf_fault->end_time = cur_cycle;
          sim_prof[mf_fault->start_time].push_back(mf_fault);
          fault_stats.erase(it);
          bst_debug_print("[DEBUG] Updated fault stats\n");
          break;
        }
      }
    }
  } else if (pcie_read_latency_queue->type == latency_type::DMA) {
    bst_debug_print("[DEBUG] Processing completed DMA request\n");
    mem_fetch *mf = pcie_read_latency_queue->mf;
    int simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
    bst_debug_print("[DEBUG] Sending DMA result to cluster %d\n",
                    simt_cluster_id);
    m_gpu->getSIMTCluster(simt_cluster_id)->push_gmmu_cu_queue(mf);
  }
  pcie_read_latency_queue = nullptr;
  bst_debug_print("[DEBUG] Cleared PCIe read latency queue\n");
}

// 대기 중인 PCIe read 전송을 스케줄링하는 함수
void gmmu_t::scheduleReadTransfer() {
  // 스테이지 큐가 비어있거나 이미 전송 중이면 아무것도 하지 않음.
  if (pcie_read_stage_queue.empty() || pcie_read_latency_queue) return;
  fault_occured = true;
  bool schedule_transfer = false;
  bool tree_large_pages_reserved = false;
  // 첫 번째 항목을 캐싱 (shared_ptr 사용)
  std::shared_ptr<pcie_latency_t> first_item = pcie_read_stage_queue.front();
  auto globalMem = m_gpu->get_global_memory();
  lp_tree_node *lp_node = get_lp_node(first_item->start_addr);
  if (first_item->type == latency_type::PCIE_READ) {//용량체크
    /*
    if(m_config.compression_policy == 0){//normal 
      
      if(get_lp_node(globalMem->get_mem_addr(first_item->page_list.front()))->large_pages_reserved){//already reserved
        tree_large_pages_reserved = true;
        schedule_transfer = true;
      }
       else {//not reserved
        if (globalMem->get_free_pages() >= get_lp_node(globalMem->get_mem_addr(first_item->page_list.front()))->size/4096)//check freee page is enough
        {  
          schedule_transfer = true;
        }
      }
    } else { *///compression
      // 최적화: 한 번만 확인하여 조건 평가
      if(m_config.tree_logic == 1){
        //schedule_transfer = (globalMem->get_free_pages() >= 512);
        schedule_transfer = get_lp_node(globalMem->get_mem_addr(first_item->page_list.front()))->large_pages_reserved?true:globalMem->get_free_pages()>=512;//check large page is reserved or not
      }
      else{
      schedule_transfer = (globalMem->get_free_pages() >= first_item->page_list.size());
    }
    //}
  } else {
    schedule_transfer = true;
  }
  if (!schedule_transfer) {
    failed_read_count++;
    pcie_read_failed = true;
    if(!num_write_stage_queue){//if no schedule transfer, do page eviction for making space
      clear_skip_eviction();//enable evcition 
      page_eviction_procedure();
    }
    if(failed_read_count > 1000000000){
      printf("bst debug failed_read_count = %zu\n", failed_read_count);
      print_debug_all(true);
      assert(0);
    }
    return;
  }
  else{
    failed_read_count = 0;
    pcie_read_failed = false;
  }
  try_error = 0;  // 정상 작동 시 에러 상태 초기화
  bst_debug_print("[DEBUG] Found pending transfer in stage queue\n");

  // DMA 요청 우선 처리: 명시적 반복문으로 탐색 (shared_ptr 사용)
  std::list<std::shared_ptr<pcie_latency_t>>::iterator dma_it =
      pcie_read_stage_queue.end();
  for (auto it = pcie_read_stage_queue.begin();
       it != pcie_read_stage_queue.end(); ++it) {
    if ((*it)->type == latency_type::DMA) {
      dma_it = it;
      break;
    }
  }

  if (dma_it == pcie_read_stage_queue.end()) {
    if (cur_cycle < pcie_read_stage_queue.front()->crm_ready_cycle) return;
    bst_debug_print("[DEBUG] No DMA requests, scheduling regular transfer\n");
    pcie_read_latency_queue = pcie_read_stage_queue.front();
  } else {
    bst_debug_print("[DEBUG] Prioritizing DMA request\n");
    pcie_read_latency_queue = *dma_it;
  }

  // 전송 타입에 따른 설정
  if (pcie_read_latency_queue->type == latency_type::PCIE_READ) {
    bst_debug_print("[DEBUG] Setting up PCIe read transfer\n");
    if (evict_form) {
      // 압축된 페이지 수 계산
      unsigned num_compressed_pages = 0;
      size_t transfer_bytes = 0;
      for (auto page : pcie_read_latency_queue->page_list) {
        if (globalMem->is_evicted_compressed(page)) {
          ++num_compressed_pages;
          transfer_bytes += get_tree_comp_size(globalMem->get_mem_addr(page)) *
                            m_config.page_size / MIN_PREFETCH_SIZE;
        } else {
          transfer_bytes += m_config.page_size;
        }
      }

      if (num_compressed_pages > 0) {
        pcie_read_latency_queue->ready_cycle =
            get_ready_cycle_bytes(transfer_bytes) +
            get_compress_cycle(num_compressed_pages,
                               pcie_read_latency_queue->decomp_band);
      } else {
        pcie_read_latency_queue->ready_cycle =
            get_ready_cycle(pcie_read_latency_queue->page_list.size());
      }
    } else {
      pcie_read_latency_queue->ready_cycle =
          get_ready_cycle(pcie_read_latency_queue->page_list.size());
    }
    if (sim_prof_enable) {
      bst_debug_print("[DEBUG] Recording transfer stats\n");
      event_stats *cp_h2d = new memory_stats(
          memcpy_h2d, cur_cycle, pcie_read_latency_queue->ready_cycle,
          pcie_read_latency_queue->start_addr, pcie_read_latency_queue->size,
          0);
      sim_prof[cur_cycle].push_back(cp_h2d);
    }
    if (util_prof) {
      bst_debug_print("[DEBUG] Recording utilization stats\n");
      for (unsigned long long read_period = cur_cycle;
           read_period < pcie_read_latency_queue->ready_cycle; ++read_period) {
        m_new_stats->pcie_read_utilization.push_back(std::make_pair(
            read_period,
            get_pcie_utilization(pcie_read_latency_queue->page_list.size())));
      }
    }
    bst_debug_print("[DEBUG] Allocating %zu pages\n",
                    pcie_read_latency_queue->page_list.size());


    if (m_config.compression_engine == 0) {
      /*if(m_config.compression_policy == 0){//normal
      if(!tree_large_pages_reserved){//예약안되있을경우, 할당이 안되어있을경우
        globalMem->alloc_pages(get_lp_node(globalMem->get_mem_addr(pcie_read_latency_queue->page_list.front()))->size/4096);
        get_lp_node(globalMem->get_mem_addr(pcie_read_latency_queue->page_list.front()))->large_pages_reserved = true;
      }
      }
      else{*///CURE-MEM
      if(m_config.tree_logic == 1){
        if(!get_lp_node(globalMem->get_mem_addr(pcie_read_latency_queue->page_list.front()))->large_pages_reserved){
          globalMem->alloc_pages(512);
          if(m_config.proactive_WB==2&&g_memory->get_free_pages()<512){//if proactive WB, and free page is not enough, do page eviction
            page_eviction_procedure();
          }
          get_lp_node(globalMem->get_mem_addr(pcie_read_latency_queue->page_list.front()))->large_pages_reserved = true;
        }
      }
      else{
        globalMem->alloc_pages(pcie_read_latency_queue->page_list.size());
        if(m_config.proactive_WB==2){
          if(pcie_read_stage_queue.size() > 1 && g_memory->get_free_pages() < (*std::next(pcie_read_stage_queue.begin()))->page_list.size()){//proactive WB이고, free page가 두 번째 요청의 페이지 수보다 적으면 eviction 수행
          page_eviction_procedure();
        }
      }
      }
      //}
    } else {
      //m_dram->m_dram_BPC_comp_queue_delay = max(m_dram->m_dram_BPC_comp_queue_delay,cur_cycle)+pcie_read_latency_queue->page_list.size()*(m_config.compress_latency);
      //pcie_read_latency_queue->ready_cycle = m_dram->m_dram_BPC_comp_queue_delay + get_ready_cycle_transfer(pcie_read_latency_queue->page_list.size())+m_config.page_fault_latency*pcie_read_latency_queue->page_list.size();
      assert(0 && "compression_engine is bst need to implement");
      for (auto page : pcie_read_latency_queue->page_list) {
        CACHELINE_DATA *comp_cacheline =
            new CACHELINE_DATA[NUM_LINES_PER_PAGE];  // 메모리 할당
        globalMem->read_single_page(page, comp_cacheline);
        LCP_CompressedPage new_LCP_comp_page =
            compressPageLCP(compressor, comp_cacheline, 0, page);
        globalMem->alloc_pages(new_LCP_comp_page.allocatedPageSize / 4096);
        LCP_compressed_pages_map[page] =
            new_LCP_comp_page;  // alloc new compressed map
      }
    }
  }
   else if (pcie_read_latency_queue->type == latency_type::PAGE_FAULT) {
    fault_occured = true;
    bst_debug_print("[DEBUG] Setting up page fault transfer\n");
    pcie_read_latency_queue->ready_cycle =
        cur_cycle +
        m_config.page_fault_latency * pcie_read_latency_queue->page_list.size();
    if(m_config.compression_engine!=0){//BPC엔진일때 압축로직후 할당오버헤드 추가
      pcie_read_latency_queue->ready_cycle += m_config.page_fault_latency * pcie_read_latency_queue->page_list.size();
    }
    if (sim_prof_enable) {
      bst_debug_print("[DEBUG] Recording page fault stats\n");
      event_stats *mf_fault = new page_fault_stats(
          cur_cycle, pcie_read_latency_queue->page_list,
          pcie_read_latency_queue->page_list.size() * m_config.page_size);
      fault_stats.push_back(mf_fault);
    }
  } else if (pcie_read_latency_queue->type == latency_type::DMA) {
    bst_debug_print("[DEBUG] Setting up DMA transfer\n");
    pcie_read_latency_queue->ready_cycle =
        get_ready_cycle_dma(pcie_read_latency_queue->mf->get_access_size());
    if (sim_prof_enable) {
      bst_debug_print("[DEBUG] Recording DMA stats\n");
      event_stats *ma_dma =
          new memory_stats(dma, cur_cycle, pcie_read_latency_queue->ready_cycle,
                          pcie_read_latency_queue->mf->get_addr(),
                          pcie_read_latency_queue->mf->get_access_size(), 0);
      sim_prof[cur_cycle].push_back(ma_dma);
    }
  }

  bst_debug_print("[DEBUG] Updating read stage queue size\n");
  num_read_stage_queue -= pcie_read_latency_queue->page_list.size();
  // 스테이지 큐에서 스케줄된 전송 항목 제거: DMA 요청이 있으면 해당 항목 제거,
  // 아니면 front 제거.
  if (dma_it == pcie_read_stage_queue.end()) {
    bst_debug_print("[DEBUG] Removing regular transfer from stage queue\n");
    pcie_read_stage_queue.pop_front();
  } else {
    bst_debug_print("[DEBUG] Removing DMA transfer from stage queue\n");
    pcie_read_stage_queue.erase(dma_it);
  }
}
void gmmu_t::processDecompressionTransfers() {
  // 캐싱: global memory 포인터를 지역 변수에 저장
  auto globalMem = m_gpu->get_global_memory();

  // [1] 완료된 decompression 요청 처리 (decomp_latency_queue)
  while (!decomp_latency_queue.empty()) {
    fault_occured = true;
    auto current_decomp_latency = decomp_latency_queue.front();
    if (current_decomp_latency &&
        (cur_cycle >= current_decomp_latency->ready_cycle)) {
      bst_debug_print("[DEBUG] Processing completed decompression\n");
      if (current_decomp_latency->type == latency_type::DECOMP) {
        // 각 페이지에 대해 처리
        m_new_stats->decompress++;
        for (auto iter = current_decomp_latency->page_list.begin();
             iter != current_decomp_latency->page_list.end(); ++iter) {
          mem_addr_t page = *iter;
          bst_debug_print("[DEBUG] Decompressing page %lx\n", page);
          globalMem->validate_page(page);
          globalMem->set_iter_decompressed(page);
          decompressing_pages_done(page, current_decomp_latency->leaf_size);
          mem_addr_t page_addr = globalMem->get_mem_addr(page);
          refresh_valid_pages(page_addr);
          m_new_stats->page_compressed[page].push_back(true);

          // decompression 시 발생한 메모리 fetch 처리
          auto compIt = comp_info.find(page);
          if (compIt != comp_info.end()) {
            bst_debug_print(
                "[DEBUG] Processing memory fetches for decompressed page %lx\n",
                page);
            // 해당 페이지에 대한 모든 메모리 fetch 처리
            for (auto mfIter = compIt->second.begin();
                 mfIter != compIt->second.end(); ++mfIter) {
              mem_fetch *mf = *mfIter;
              int cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
              bst_debug_print(
                  "[DEBUG] Sending decompressed data to cluster %d\n",
                  cluster_id);
              m_gpu->getSIMTCluster(cluster_id)->push_gmmu_cu_queue(mf);
              // SM stall 해제 (compression 대기 해제)
              memory_stalled_sms.remove_stalled_sm(cluster_id);
            }
            auto &fault_latencies = m_new_stats->pf_comp_fault_latency[page];
            if (!compIt->second.empty() && !fault_latencies.empty()) {
              fault_latencies.back() = cur_cycle - fault_latencies.back();
            }
            comp_info.erase(compIt);
            skip_cycles = false;
            bst_debug_print(
                "[DEBUG] Updated compression fault stats for page %lx\n", page);
          }
        }
      } else if (current_decomp_latency->type == latency_type::COMP_FAULT) {
        bst_debug_print("[DEBUG] Processing completed compression fault\n");
        //m_new_stats->comp_fault_num++;
        if (sim_prof_enable) {
          for (auto iter = comp_fault_stats.begin();
               iter != comp_fault_stats.end(); ++iter) {
            // transfering_pages의 첫 페이지가 일치하면 해당 이벤트를 업데이트
            if (((page_comp_fault_stats *)(*iter))->transfering_pages.front() ==
                current_decomp_latency->page_list.front()) {
              event_stats *mf_fault = *iter;
              mf_fault->end_time = cur_cycle;
              sim_prof[mf_fault->start_time].push_back(mf_fault);
              comp_fault_stats.erase(iter);
              bst_debug_print("[DEBUG] Updated compression fault stats\n");
              break;
            }
          }
        }
      }
      decomp_latency_queue.pop_front();
      bst_debug_print("[DEBUG] Cleared decompression latency queue\n");
    } else {
      // 아직 처리할 decompression 요청이 남아있으므로 종료
      break;
    }
  }
}
void gmmu_t::scheduleDecompressionTransfers() {
  auto globalMem = m_gpu->get_global_memory();
  bool search_mf_wait_done = false;
  auto last_search_iter = decomp_stage_queue.begin();
  while (!decomp_stage_queue.empty() &&
         comp_latency_queue.size() + decomp_latency_queue.size() <
             get_memory_stalled_sm_count() + max_parallel_comp_requests) {
    fault_occured = true;
    // 먼저 mf_wait_for_comp가 true인 항목을 찾기
    auto selectedIter = decomp_stage_queue.end();
    bool found_mf_wait_for_comp = false;
    
    if (search_mf_wait_done) {
      // 전체 검색이 완료되었으므로 front에서 바로 처리
      bst_debug_print("[DEBUG] Search completed previously, processing front of decomp_stage_queue\n");
      selectedIter = decomp_stage_queue.begin();
    } else {
      // mf_wait_for_comp가 true인 항목을 찾기 위해 마지막 검색 위치부터 시작
      bst_debug_print("[DEBUG] Searching for mf_wait_for_comp entries in decomp_stage_queue from last position\n");
      auto stageIter = last_search_iter;//마지막검색
      
      for (; stageIter != decomp_stage_queue.end(); ++stageIter) {//마지막 검색위치부터 끝까지
        if ((*stageIter)->mf_wait_for_comp) {
          bst_debug_print("[DEBUG] Found mf_wait_for_comp entry at addr %lx\n", (*stageIter)->start_addr);
          selectedIter = stageIter;
          found_mf_wait_for_comp = true;
          last_search_iter = (stageIter == decomp_stage_queue.end()) ? decomp_stage_queue.begin() : std::next(stageIter);
          break;
        }
      }
      
      // 끝까지 검색했는데도 찾지 못한 경우
      if (!found_mf_wait_for_comp && stageIter == decomp_stage_queue.end()) {
        bst_debug_print("[DEBUG] Reached end of queue, no more mf_wait_for_comp entries\n");
        search_mf_wait_done = true;//전체 검색 완료
        last_search_iter = decomp_stage_queue.begin();//초기화 굳이안해도되지만..안전하게
        selectedIter = decomp_stage_queue.begin();//처음걸로 처리
      }
    }
    auto stageFront = *selectedIter;
    if (max(stageFront->ready_cycle, stageFront->crm_ready_cycle) > cur_cycle) {
      if(search_mf_wait_done){//처리할 항목이 없으면 종료
        return ;
      }
      search_mf_wait_done=true;
      if(!stageFront->comp_fault_checked){
        stageFront->comp_fault_checked=true;
      }
      last_search_iter=decomp_stage_queue.begin();
      continue;
    }
    
    size_t requiredPages;
    if (stageFront->type == latency_type::DECOMP) {
      requiredPages = (stageFront->size-stageFront->comp_size)/4096;
    } else {
      requiredPages = 1;  // COMP_FAULT는 바로 처리
    }
    
    if (globalMem->get_free_pages() >= requiredPages) {
      if(!pcie_read_failed){
        failed_read_count=0;
      }
      else if(search_mf_wait_done){//프리패치는 나중에 처리.
        return ;  // 추가 스케줄링 없이 종료
      }
    } else{

      failed_read_count++;
      return ;  // 추가 스케줄링 없이 종료
    }
    
    bst_debug_print(
        "[DEBUG] Conditions met for scheduling from decomp stage queue\n");

    // 선택된 항목을 decomp_latency_queue에 추가
    if (found_mf_wait_for_comp) {
      bst_debug_print("[DEBUG] Processing mf_wait_for_comp entry\n");
      decomp_latency_queue.push_back(*selectedIter);
    }
    else{
      bst_debug_print("[DEBUG] No special requests found, using front of queue\n");
      decomp_latency_queue.push_back(decomp_stage_queue.front());
    }
    
    // 새로 스케줄된 decompression 요청 처리
    auto current_decomp_queue = decomp_latency_queue.back();
    if (current_decomp_queue->type == latency_type::DECOMP) {
      bst_debug_print("[DEBUG] Processing DECOMP request\n");
      current_decomp_queue->ready_cycle =
          get_ready_cycle_pages_compress(current_decomp_queue->page_list.size(),
                                         current_decomp_queue->decomp_band);
      bst_debug_print("[DEBUG] Set ready cycle to %llu\n",
                      current_decomp_queue->ready_cycle);
      if (sim_prof_enable) {
        event_stats *cp_h2d = new memory_stats(
            decomp, cur_cycle, current_decomp_queue->ready_cycle,
            current_decomp_queue->start_addr, current_decomp_queue->size, 0);
        sim_prof[cur_cycle].push_back(cp_h2d);
        bst_debug_print("[DEBUG] Added decompression event to sim profile\n");
      }
      if (!m_gpu->get_global_memory()->change_size_comp_with_comp_size(
              current_decomp_queue->size-current_decomp_queue->comp_size, false)) {  // decompression 실패 시
        bst_debug_print(
            "[DEBUG] Failed to change memory size for decompression\n");
        printf("bst debug change size failed in decomp stage\n");
        printf("bst debug deecomp addr %lx\n",
               current_decomp_queue->start_addr);
        printf("bst debug deecomp size %zu\n",
               current_decomp_queue->comp_size);
        print_debug_all(true);
      }
      // decomp stage 큐의 사이즈를 조정 (할당된 페이지 수를 comp_ratio로 나눔)
      num_decomp_stage_queue -=
          (current_decomp_queue->size-current_decomp_queue->comp_size)/m_config.page_size;
    } else if (current_decomp_queue->type == latency_type::COMP_FAULT) {
      bst_debug_print("[DEBUG] Processing COMP_FAULT request\n");
      // Handling was already reserved and completed in the stage queue.
      current_decomp_queue->ready_cycle = cur_cycle;
      bst_debug_print("[DEBUG] Set compression fault ready cycle to %llu\n",
                      current_decomp_queue->ready_cycle);
      if (sim_prof_enable) {
        event_stats *mf_fault = new page_comp_fault_stats(
            cur_cycle, current_decomp_queue->page_list,
            current_decomp_queue->page_list.size() * m_config.page_size);
        comp_fault_stats.push_back(mf_fault);
        bst_debug_print("[DEBUG] Added compression fault event to stats\n");
      }
    }
    bst_debug_print("[DEBUG] Updated decomp stage queue size to %f\n",
                    num_decomp_stage_queue);

    // 스케줄링한 항목 제거: 선택된 항목 제거
    if (selectedIter == decomp_stage_queue.begin()) {
      bst_debug_print("[DEBUG] Removing front of decomp stage queue\n");
      decomp_stage_queue.pop_front();
    } else {
      bst_debug_print("[DEBUG] Removing selected item from decomp stage queue\n");
      decomp_stage_queue.erase(selectedIter);
    }
  }
}
void gmmu_t::cycle() {
  int simt_cluster_id = 0;
  cur_cycle = m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle;
  fflush(stdout);
  // 최적화: 자주 사용되는 global_memory 포인터를 캐싱
  auto globalMem = m_gpu->get_global_memory();
  if(process_print_on){
    printf("bst debug free %f gddr %f comp free %f comp_evict_stage_queue %f num_comp_stage_queue %f pcie_read_stage_queue %f\n",globalMem->get_free_pages(),globalMem->get_total_pages(),globalMem->get_comp_free_pages(),num_comp_evict_stage_queue,num_comp_stage_queue,num_read_stage_queue);
    printf("bst debug num_write_stage_queue %f num_decomp_stage_queue %f num_comp_stage_queue %f num_comp_evict_stage_queue %f\n", num_write_stage_queue, num_decomp_stage_queue, num_comp_stage_queue, num_comp_evict_stage_queue);
    printf("bst debug comp_latency_queue %zu decomp_latency_queue %zu\n", comp_latency_queue.size(), decomp_latency_queue.size());
    process_print_on=false;
  }
  bst_debug_print("[DEBUG] gmmu_t::cycle() - Current cycle: %llu\n", cur_cycle);
  bst_debug_print(
      "[DEBUG] gddr size = %f free pages = %f init pages = %f comp free pages "
      "= %f\n",
      globalMem->get_total_pages(),
      globalMem->get_free_pages(),
      globalMem->get_init_pages(),
      globalMem->get_comp_free_pages());
  bst_debug_print(
      "[DEBUG] num_read_stage_queue = %f, num_write_stage_queue = %f, "
      "num_decomp_stage_queue = %f, num_comp_stage_queue = %f\n",
      num_read_stage_queue, num_write_stage_queue, num_decomp_stage_queue,
      num_comp_stage_queue);
  bst_debug_print(
      "[DEBUG] comp_latency_queue = %zu, decomp_latency_queue = %zu\n",
      comp_latency_queue.size(), decomp_latency_queue.size());
//1=evict+comp_evict 2=evict 3=evict+comp 4=comp_evict 5=comp_evict+comp 6=comp
  int page_action =0;
  if(m_config.proactive_WB==1||(m_config.forest_LRU&&failed_read_count&&!num_write_stage_queue)){
  page_action = globalMem->should_page_action(
      num_read_stage_queue, num_write_stage_queue, num_decomp_stage_queue,
      num_comp_stage_queue, num_comp_evict_stage_queue,m_config.free_page_buffer_percentage,
      m_config.comp_buffer_percentage, m_config.comp_evict_start_percentage);
  }
  // 최적화: 반복되는 조건문을 미리 계산
  const bool is_partial_policy = (compression_policy == compression_policy_struct::PARTIAL);
  const bool throttle_enabled = (m_config.m_shader_config.ETC_throttle_SM || m_config.m_shader_config.CURE_throttle_SM);
  if (page_action == 2) { // evict
    need_throttle_SM =1;
    bool ret = false;
    if (is_partial_policy) {
      ret = page_compression_procedure(true); // compression이 안될 때
      if (!ret) {
        ret = page_comp_eviction_procedure(); // compression이나 eviction이 안될 때 evict를 대신 실행
      }
    } else {
      ret = page_eviction_procedure();
    }
  } else if (is_partial_policy) {
    if (page_action == 1) { // comp_evict+comp 두개 다 하기
      bool ret1 = page_compression_procedure(true);
      bool ret2 = page_comp_eviction_procedure();
      need_throttle_SM=1;
    } else if (page_action == 3) { // comp_evict
      bool ret1 = page_comp_eviction_procedure();
      bool ret2 = page_compression_procedure(false);
      need_throttle_SM=1;
    } else if (page_action == 4) { // comp
      bool ret = page_compression_procedure(false);
      need_throttle_SM=1;
    }
  } 
  
  processWriteBackLatencyQueue();
  scheduleWriteBackTransfer();
  processCompressionLatencyQueue();
  scheduleCompressionTasks();
  // check whether the current transfer in the pcie latency queue is finished
  bst_debug_print("[DEBUG] Checking for pending compression tasks\n");
  processReadLatencyQueue();
  scheduleReadTransfer();

  // schedule a transfer if there is a pending item in staging queue and
  // nothing is being served at the read latency queue and we have available
  // free pages
  bst_debug_print("[DEBUG] Checking for new PCIe read transfers to schedule\n");
  processDecompressionTransfers();
  scheduleDecompressionTransfers();
  std::map<mem_addr_t, std::list<mem_fetch *>> page_fault_this_turn;
  std::map<mem_addr_t, std::list<mem_fetch *>> page_comp_this_turn;
  // check the page_table_walk_delay_queue
  bst_debug_print("[DEBUG] Processing page table walk queue\n");
  while (!page_table_walk_queue.empty() &&
         ((cur_cycle) >= page_table_walk_queue.front().ready_cycle)) {
    //   if (cur_cycle < it->ready_cycle) {
    mem_fetch *mf = page_table_walk_queue.front().mf;
    if(m_config.two_level_LRU){
      lp_tree_node *accessed_node = get_lp_node(mf->get_addr());
      if(accessed_node->size>2*1024*1024){//if 4MB page, check which child node is accessed
        if(accessed_node->left->addr<=mf->get_addr() && mf->get_addr()<accessed_node->left->addr+accessed_node->left->size){
          accessed_node->left->accessed = true;
        }
        else{
          accessed_node->right->accessed = true;
        }
      }
      else if(accessed_node->size<2*1024*1024){
        lp_tree_node *real_2MB_root=accessed_node->parent;
        if(real_2MB_root==nullptr){
          // parent가 없으면 자기 자신을 업데이트
          accessed_node->accessed = true;
        }
        else{
          while(real_2MB_root->size<2*1024*1024){
            if(real_2MB_root->parent!=nullptr){
              real_2MB_root=real_2MB_root->parent;
            }
            else{
              break;
            }
          }
          real_2MB_root->accessed = true;
        }
      }
      else{ // 2MB 페이지인 경우
        accessed_node->accessed = true;
      }
    }
    // 맨 위에서 한 번만 계산
    mem_addr_t page_num = globalMem->get_page_num(mf->get_mem_access().get_addr());
    if(m_config.fault_based_LRU){
      check_write_stage_queue(page_num, false);
    }else{
      check_write_stage_queue(page_num, true);
    }
    // 그 다음에 page_list, comp_page_list 생성
    list<mem_addr_t> page_list = globalMem->get_faulty_pages(mf->get_addr(), mf->get_access_size());
    list<mem_addr_t> comp_page_list = globalMem->get_compressed_pages(mf->get_addr(), mf->get_access_size());
    
    if(compression_policy == compression_policy_struct::NONE){
      assert(comp_page_list.empty()&&"bst debug no compression policy must not have compressed page\n");
    }
    simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();//SM id 
    bst_debug_print("[DEBUG] Processing memory fetch for cluster %d\n",
                    simt_cluster_id);
    
    // 최적화: 빈 리스트 확인을 미리 수행하여 불필요한 계산 방지
    const bool has_page_faults = !page_list.empty();
    const bool has_comp_pages = !comp_page_list.empty();
    if(has_page_faults||has_comp_pages){
      if(m_config.tree_logic==4&&oversubscription_on){//if SAFE, check tree type
        lp_tree_node* fault_node = get_lp_node(mf->get_addr());
        fault_node->fault_from[fault_node->fault_index] = simt_cluster_id;
        fault_node->fault_index = (fault_node->fault_index + 1) % 32;
        
        // fault_from에 있는 unique한 simt_cluster_id 수를 빠르게 계산
        // 비트마스크를 사용하여 O(1) 시간에 unique count 계산
        std::unordered_set<int> unique_clusters;
        for (int i = 0; i < 32; i++) {
          if (fault_node->fault_from[i] >= 0) {
            unique_clusters.insert(fault_node->fault_from[i]);
          }
        }
        int unique_cluster_count = unique_clusters.size();
        if(unique_cluster_count>=10){
          update_lp_tree_prefetch_threshold(fault_node, 100.0);
        }
        else{
          update_lp_tree_prefetch_threshold(fault_node, 1.0);
        }
      }
    }
    // if there is no page fault, directly return to the upward queue of cluster
    if (!has_page_faults && !has_comp_pages) {  // all valid, it is okay comp latency, comp
                                   // stage, pcie_write_stgae
      bst_debug_print("[DEBUG] No page faults or compressed pages found\n");
      (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);

      m_new_stats->mf_page_hit[simt_cluster_id]++;
      bst_debug_print("[DEBUG] Incremented page hit counter for cluster %d\n",
                      simt_cluster_id);
    } else if (!has_comp_pages) {  // faulty list only
      bst_debug_print("[DEBUG] Found page fault\n");
      fault_occured = true;
      assert(page_list.size() == 1);
      m_new_stats->mf_page_miss[simt_cluster_id]++;
      // 최적화: page_list의 첫 번째 요소를 미리 가져옴
      mem_addr_t target_page = page_list.front();
      
      // the page request is already there in MSHR either as a page fault or as
      // part of scheduled prefetch request
      auto req_iter = req_info.find(target_page);
      if (req_iter != req_info.end()) {
        bst_debug_print("[DEBUG] Page fault already pending in MSHR\n");
        m_new_stats->mf_page_fault_pending++;
        req_iter->second.push_back(mf);
        // SM이 이미 stalled 상태지만 추가 요청이 들어옴 (intensity 증가)
        memory_stalled_sms.add_stalled_sm(simt_cluster_id);
      } else {
        // if the memory fetch is part of any requests in the prefetch command
        // buffer then add it to the incoming replayable_nacks
        
        // 최적화: prefetch_req_buffer 검색을 위한 변수 미리 선언
        bool found_in_prefetch = false;
        auto mf_addr = mf->get_addr();

        for (auto& prefetch_req : prefetch_req_buffer) {
          if (prefetch_req.start_addr <= mf_addr &&
              mf_addr < prefetch_req.start_addr + prefetch_req.size) {
            bst_debug_print("[DEBUG] Found memory fetch in prefetch buffer\n");
            m_new_stats->mf_page_fault_pending++;
            prefetch_req.incoming_replayable_nacks[target_page].push_back(mf);
            found_in_prefetch = true;
            break;
          }
        }

        // if the memory fetch is not part of any request in the prefetch
        // command buffer
        if (!found_in_prefetch) {
          bst_debug_print("[DEBUG] Memory fetch not in prefetch buffer\n");
          // if dma is enabled/it is a write access/read access counter hasn't
          if (!should_cause_page_migration(
                  mf->get_mem_access().get_addr(),
                  mf->get_mem_access().get_type() == GLOBAL_ACC_W)) {
            bst_debug_print("[DEBUG] Using DMA for memory fetch\n");
            m_new_stats->num_dma++;
            std::shared_ptr<pcie_latency_t> p_t =
                std::make_shared<pcie_latency_t>();
            mf->set_dma();

            p_t->mf = mf;
            p_t->type = latency_type::DMA;

            reserve_crm_handling(p_t);
            pcie_read_stage_queue.push_back(p_t);
            num_read_stage_queue +=
                page_list.size();  // maybe DMA does not need?
          } else {
            bst_debug_print("[DEBUG] Using page migration for memory fetch\n");
            if (dma_mode != dma_type::DISABLED &&
                mf->get_mem_access().get_type() == GLOBAL_ACC_W) {
              m_new_stats->dma_page_transfer_write++;
            } else if (dma_mode != dma_type::DISABLED &&
                       mf->get_mem_access().get_type() == GLOBAL_ACC_R) {
              m_new_stats->dma_page_transfer_read++;
            }

            page_fault_this_turn[page_list.front()].push_back(mf);
            // 새로운 페이지 폴트로 인한 SM stall 추적
            memory_stalled_sms.add_stalled_sm(simt_cluster_id);
          }
        }
      }
    } else if (!has_page_faults) {  // compressed page list only
      bst_debug_print("[DEBUG] Found compressed page\n");
      fault_occured = true;
      m_new_stats->mf_page_compressed[simt_cluster_id]++;
      
      // 최적화: comp_page_list의 첫 번째 요소를 미리 가져옴
      mem_addr_t target_comp_page = comp_page_list.front();
      
      auto comp_iter = comp_info.find(target_comp_page);
      if ((comp_iter == comp_info.end() || comp_iter->second.empty()) &&
          (page_comp_this_turn.find(target_comp_page) == page_comp_this_turn.end() ||
           page_comp_this_turn.at(target_comp_page).empty())) {
        m_new_stats->pf_comp_fault_latency[target_comp_page].push_back(cur_cycle);
      }
      if (comp_iter != comp_info.end()) {
        bst_debug_print("[DEBUG] Compression already pending\n");
        m_new_stats->mf_page_comp_pending++;
        comp_iter->second.push_back(mf);
        set_decomp_mf_wait_for_comp(mf->get_addr());
        // SM이 이미 compression으로 stalled 상태지만 추가 요청이 들어옴
        memory_stalled_sms.add_stalled_sm(simt_cluster_id);
      } else {
        bst_debug_print("[DEBUG] New compression transfer needed\n");
        m_new_stats->comp_page_tansfer++;
        page_comp_this_turn[target_comp_page].push_back(mf);
        // 새로운 compression transfer로 인한 SM stall 추적
        memory_stalled_sms.add_stalled_sm(simt_cluster_id);
      }
    } else {
      assert(0 && "bst debug comp and valid page list in same time \n");
    }
    page_table_walk_queue.pop_front();
  }

  bst_debug_print("[DEBUG] Starting hardware prefetch\n");
  // call hardware prefetcher based on the current page faults
  do_hardware_prefetch(page_fault_this_turn);

  bst_debug_print("[DEBUG] Starting hardware precompression\n");
  // fetch from cluster's cu to gmmu queue and push it into the page table way
  do_hardware_precomp(page_comp_this_turn);

  // delay queue
  bst_debug_print("[DEBUG] Processing cluster queues\n");
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (!(m_gpu->getSIMTCluster(i))->empty_cu_gmmu_queue()) {
      bst_debug_print("[DEBUG] Processing queue for cluster %d\n", i);
      mem_fetch *mf = (m_gpu->getSIMTCluster(i))->front_cu_gmmu_queue();
      struct page_table_walk_latency_t pt_t;
      pt_t.mf = mf;
      pt_t.ready_cycle = cur_cycle + m_config.page_table_walk_latency;
      page_table_walk_queue.push_back(pt_t);

      (m_gpu->getSIMTCluster(i))->pop_cu_gmmu_queue();
    }
  }

  bst_debug_print("[DEBUG] Checking prefetch requests\n");
  // check if there is an active outstanding prefetch request
  if (!prefetch_req_buffer.empty() && prefetch_req_buffer.front().active) {
    bst_debug_print("[DEBUG] Found active prefetch request\n");
    prefetch_req &pre_q = prefetch_req_buffer.front();

    // schedule for page transfers from the active prefetch request when there
    // is no pending transfer for the same can be the very first time or a
    // scheduled big chunk of pages (2MB) is finsihed just now
    if (pre_q.pending_prefetch.empty()) {
      bst_debug_print("[DEBUG] No pending prefetch transfers\n");
      // case when the last schedule finished, it is not the first time
      if (pre_q.cur_addr > pre_q.start_addr) {
        bst_debug_print("[DEBUG] Last schedule finished\n");
        if (sim_prof_enable) {
          update_sim_prof_prefetch_break_down(cur_cycle);
        }

        m_new_stats->pf_fault_latency.back().second =
            cur_cycle - m_new_stats->pf_fault_latency.back().second;

        // all the memory fetches created by core on page fault were aggreagted
        // earlier now they are replayed back together to the core
        for (map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
                 pre_q.outgoing_replayable_nacks.begin();
             iter != pre_q.outgoing_replayable_nacks.end(); iter++) {
          for (std::list<mem_fetch *>::iterator iter2 = iter->second.begin();
               iter2 != iter->second.end(); iter2++) {
            mem_fetch *mf = *iter2;

            simt_cluster_id = mf->get_sid() / m_config.num_core_per_cluster();
            bst_debug_print("[DEBUG] Replaying memory fetch to cluster %d\n",
                            simt_cluster_id);
            // push them to the upward queue to replay them back to the
            // corresponding core in bulk
            (m_gpu->getSIMTCluster(simt_cluster_id))->push_gmmu_cu_queue(mf);
          }
        }
        pre_q.outgoing_replayable_nacks.clear();
      }

      // all the memory fetches have been replayed and
      // the prefetch request is completed entirely
      // now signal the stream that the operation is finished so that it can
      // schedule something else
      if (pre_q.cur_addr == pre_q.start_addr + pre_q.size) {
        bst_debug_print("[DEBUG] Prefetch request completed\n");
        pre_q.m_stream->record_next_done();

        if (sim_prof_enable) {
          update_sim_prof_prefetch(pre_q.start_addr, pre_q.size, cur_cycle);
        }

        prefetch_req_buffer.pop_front();
        return;
      }

      mem_addr_t start_addr = 0;

      std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
      // break the loop if
      //  Case 1: reach the end of this prefetch
      //  Case 2: it reaches the 2MB line from starting of the allocation
      //  Case 3: it encounters a valid page in between
      bst_debug_print("[DEBUG] Starting prefetch loop\n");
      do {
        // get the page number for the current updated address
        mem_addr_t page_num =
            m_gpu->get_global_memory()->get_page_num(pre_q.cur_addr);

        // update the current address by page size as we break a big chunk (2MB)
        // in the granularity of the smallest unit of page
        pre_q.cur_addr += m_config.page_size;

        // check for Case 3, i.e., we encounter a valid page
        if (m_gpu->get_global_memory()->is_valid(page_num)) {
          bst_debug_print("[DEBUG] Found valid page %lx\n", page_num);
          m_new_stats->pf_page_hit++;

          // check if this page is currently written back

            check_write_stage_queue(page_num, false);

          // break out of loop only when we have already scheduled some pages
          // for transfer if not we will continue skipping valid pages if any
          // until we find some invalid pages to transfer
          if (!pre_q.pending_prefetch.empty()) {
            break;
          }
        } else {
          bst_debug_print("[DEBUG] Found invalid page %lx\n", page_num);
          m_new_stats->pf_page_miss++;

          // remember this page as pending under the prefetch request
          pre_q.pending_prefetch.push_back(page_num);

          if (start_addr == 0) {
            start_addr = m_gpu->get_global_memory()->get_mem_addr(page_num);
            p_t->start_addr = pre_q.cur_addr;
          }

          // just create a placeholder in MSHR for the memory fetches created by
          // core on page fault later in the time so that they go to outgoing
          // replayable nacks, rather than incoming
          req_info[page_num];

          // incoming nacks hold the list of page faults for the transfer which
          // has not been scheduled yet so instead of pushing them to MSHR and
          // then again getting back to the outgoing list directly switch
          // between the incoming and outgoing list of replayable nacks
          if (pre_q.incoming_replayable_nacks.find(page_num) !=
              pre_q.incoming_replayable_nacks.end()) {
            bst_debug_print(
                "[DEBUG] Moving page %lx from incoming to outgoing nacks\n",
                page_num);
            pre_q.outgoing_replayable_nacks[page_num].merge(
                pre_q.incoming_replayable_nacks[page_num]);
            pre_q.incoming_replayable_nacks.erase(page_num);
          }

          // schedule this page as it is not valid to the read stage queue
          p_t->page_list.push_back(page_num);
          m_new_stats->pf_page_fault_latency[page_num].push_back(cur_cycle);
        }

      } while (
          pre_q.cur_addr !=
              (pre_q.start_addr +
               pre_q.size) &&  // check for Case 1, i.e., we reached the end
                               // of prefetch request
          ((unsigned long long)(pre_q.cur_addr - pre_q.allocation_addr)) %
              ((unsigned long long)
                   MAX_PREFETCH_SIZE));  // Case 2: allowing maximum transfer
                                         // size as huge page size of 2MB

      if (!p_t->page_list.empty()) {
        bst_debug_print(
            "[DEBUG] Scheduling new prefetch transfer with %zu pages\n",
            p_t->page_list.size());
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->leaf_size = get_lp_node(p_t->start_addr)->leaf_size;
        p_t->type = latency_type::PCIE_READ;
        p_t->decomp_band = m_config.decomp_band_scalar;
        p_t->start_cycle = cur_cycle;
        printf("this haapen? bst\n");
        reserve_crm_handling(p_t);
        pcie_read_stage_queue.push_back(p_t);
        num_read_stage_queue += p_t->page_list.size();
      }

      m_new_stats->pf_fault_latency.push_back(std::make_pair(
          pre_q.pending_prefetch.size() * m_config.page_size, cur_cycle));

      if (sim_prof_enable && !pre_q.pending_prefetch.empty()) {
        bst_debug_print(
            "[DEBUG] Adding prefetch breakdown event to sim profile\n");
        event_stats *cp_pref_bd =
            new memory_stats(prefetch_breakdown, cur_cycle, start_addr,
                             pre_q.pending_prefetch.size() * m_config.page_size,
                             pre_q.m_stream->get_uid());
        sim_prof[cur_cycle].push_back(cp_pref_bd);
      }
    }
  }

  bst_debug_print("[DEBUG] Checking warp status\n");
  if (!skip_cycles && !all_warps.empty() &&
      all_warps.size() == fail_warps.size()) {
    bst_debug_print("[DEBUG] Processing failed warps\n");
    std::set<int> temp_set;
    for (std::list<shd_warp_t *>::iterator iter = fail_warps.begin();
         iter != fail_warps.end(); iter++) {
      temp_set.insert((*iter)->get_warp_id());
    }

    for (auto iter = req_info.begin(); iter != req_info.end(); ++iter) {
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end();
           ++iter2) {
        if (temp_set.find((*iter2)->get_inst().warp_id()) != temp_set.end())
          temp_set.erase(temp_set.find((*iter2)->get_inst().warp_id()));
      }
    }
    for (auto iter = comp_info.begin(); iter != comp_info.end(); ++iter) {
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end();
           ++iter2) {
        if (temp_set.find((*iter2)->get_inst().warp_id()) != temp_set.end())
          temp_set.erase(temp_set.find((*iter2)->get_inst().warp_id()));
      }
    }

    if (temp_set.empty()) {
      bst_debug_print(
          "[DEBUG] All warps processed, setting skip_cycles to true\n");
      skip_cycles = true;
    } else {
      bst_debug_print(
          "[DEBUG] Some warps remaining, setting skip_cycles to false\n");
      skip_cycles = false;
    }
    fflush(stdout);
  }
}
void gmmu_t::do_hardware_precomp(
    std::map<mem_addr_t, std::list<mem_fetch *>> &page_comp_this_turn) {
  if (!page_comp_this_turn.empty()) {
    std::list<std::list<mem_addr_t>> all_transfer_all_page;
    std::list<std::list<mem_addr_t>> all_transfer_faulty_pages;
    std::map<mem_addr_t, std::list<mem_fetch *>> temp_req_info;

      std::map<mem_addr_t, std::set<mem_addr_t>> lp_pf_groups;

      for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator it =
               page_comp_this_turn.begin();
           it != page_comp_this_turn.end(); ) {
        mem_addr_t page_addr =
            m_gpu->get_global_memory()->get_mem_addr(it->first);
            auto comp_iter = comp_info.find(it->first);
            if (comp_iter != comp_info.end()) {
              bst_debug_print("[DEBUG] Compression already pending for page %llu, removing from page_comp_this_turn\n", it->first);
              m_new_stats->mf_page_comp_pending++;
              comp_iter->second.insert(comp_iter->second.end(), it->second.begin(), it->second.end());
              set_decomp_mf_wait_for_comp(page_addr);
              it = page_comp_this_turn.erase(it);  // 삭제하고 다음 iterator로 이동
            } else {
              struct lp_tree_node *root = get_lp_node(page_addr);
              lp_pf_groups[root->addr].insert(
                  page_addr);  // tree에서 root기록하고 comp기록에서 실제로 comp된
              ++it;  // 다음 요소로 이동
            }
      }

      // 최적화: 반복자 및 루프 구조 개선, 불필요한 변수 제거, 캐싱 활용
      for (auto lp_pf_iter = lp_pf_groups.begin(); lp_pf_iter != lp_pf_groups.end(); ++lp_pf_iter) {
        std::set<mem_addr_t> schedulable_basic_blocks;
        std::list<mem_addr_t> cur_transfer_all_pages;
        std::list<mem_addr_t> cur_transfer_faulty_pages;

        // root를 한 번만 계산하여 재사용
        struct lp_tree_node *root_for_block = nullptr;
        mem_addr_t root_leaf_size = 0;

        // page_list 순회 (set이므로 정렬되어 있음)
        for (const auto &page_addr : lp_pf_iter->second) {
          struct lp_tree_node *root = get_lp_node(page_addr);
          if (!root_for_block) {
            root_for_block = root;
            root_leaf_size = root->leaf_size;
          }
          // On-access decompression span: 64KB in MICRO mode (decompress only the touched
          // 64KB block, leaving the rest of the 2MB leaf compressed), else the whole leaf.
          bool micro = (m_config.MICRO_barrier_Range && m_config.tree_logic == 2);
          mem_addr_t bb_addr = micro ? (page_addr / MIN_PREFETCH_SIZE) * MIN_PREFETCH_SIZE
                                     : get_basic_block(root, page_addr);
          size_t decomp_span = micro ? MIN_PREFETCH_SIZE : root->leaf_size;
          update_basic_block_comp(root, bb_addr, decomp_span, true);
          // span 범위 내에서 16K 단위로 basic block을 추가
          for(int num_16K=0; num_16K < (int)(decomp_span/16384); num_16K++){
            schedulable_basic_blocks.insert(bb_addr + num_16K*16384);
          }
          // decomp_stage_queue에서 span 범위 내의 모든 16K basic block에 대해 우선순위 부여
          for(int num_16K=0; num_16K < (int)(decomp_span/16384); num_16K++){
            mem_addr_t cur_bb_addr = bb_addr + num_16K*16384;
            for (auto check_decomp_stage_queue = decomp_stage_queue.begin();
                 check_decomp_stage_queue != decomp_stage_queue.end(); ++check_decomp_stage_queue) {
              if ((*check_decomp_stage_queue)->start_addr == cur_bb_addr) {
                decomp_stage_queue.splice(decomp_stage_queue.begin(), decomp_stage_queue, check_decomp_stage_queue);
                break;
              }
            }
          }

          cur_transfer_faulty_pages.push_back(m_gpu->get_global_memory()->get_page_num(page_addr));
        }

        // prefetcher가 TBN이거나 two_level_LRU일 때 LRU decompression 적용
        // MICRO 모드에서는 접근한 64KB만 풀어야 하므로 LRU 디컴프 프리패치를 건너뜀
        if ((prefetcher == hwardware_prefetcher::TBN || m_config.two_level_LRU) && m_config.enable_LRU_decompression && root_for_block
            && !(m_config.MICRO_barrier_Range && m_config.tree_logic == 2)) {
          traverse_and_fill_lp_tree_comp(root_for_block, schedulable_basic_blocks);
        }

        // 각 basic block에 대해 접근 및 페이지 수집
        for (const auto &bb : schedulable_basic_blocks) {
          block_access_list.push_back(std::make_pair(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, bb));
          // root_for_block이 nullptr일 경우를 방지
          std::list<mem_addr_t> all_block_pages = m_gpu->get_global_memory()->get_compressed_pages(bb, PREFETCH_16K_SIZE);

          for (const auto &pg : all_block_pages) {
            if (temp_req_info.find(pg) == temp_req_info.end()) {
              temp_req_info[pg];
              cur_transfer_all_pages.push_back(pg);
            }
          }
        }
        all_transfer_all_page.push_back(std::move(cur_transfer_all_pages));
        all_transfer_faulty_pages.push_back(std::move(cur_transfer_faulty_pages));
      }
    

    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter =
             temp_req_info.begin();
         iter != temp_req_info.end(); iter++) {
      comp_info[iter->first];
      comp_info[iter->first].merge(iter->second);
    }

    std::list<std::list<mem_addr_t>>::iterator all_pg_iter =
        all_transfer_all_page.begin();
    // std::list<std::list<mem_addr_t>>::iterator all_pf_iter;
    std::list<std::list<mem_addr_t>>::iterator all_pf_iter =
        all_transfer_faulty_pages
            .begin();  // compressed page is already uncompressed
    // 16K 단위로 all_transfer_all_page에 들어가 있으므로, 연속된 페이지를 합치고 64K(16페이지)보다 작은 경우 comp_size를 64K 기준으로 나눠서 설정
    for (; all_pg_iter != all_transfer_all_page.end(); all_pg_iter++, all_pf_iter++) {
      std::list<mem_addr_t>::iterator pf_iter = all_pf_iter->begin();
      std::list<mem_addr_t>::iterator pg_iter = all_pg_iter->begin();
      std::list<mem_addr_t>::iterator range_start = pg_iter;
      int range_count = 0;

      while (pg_iter != all_pg_iter->end()) {
        // 연속된 페이지 구간 찾기
        range_start = pg_iter;
        range_count = 1;
        std::list<mem_addr_t>::iterator next_iter = pg_iter;
        ++next_iter;
        while (next_iter != all_pg_iter->end() && *next_iter == *pg_iter + 1) {//연속찾기기
          ++range_count;
          ++pg_iter;
          ++next_iter;
        }

        // 64K(16페이지) 단위로 쪼개서 처리
        int processed = 0;
        while (processed < range_count) {
          std::list<mem_addr_t>::iterator chunk_start = range_start;
          std::advance(chunk_start, processed);
          const mem_addr_t chunk_addr = m_gpu->get_global_memory()->get_mem_addr(*chunk_start);
          const int pages_to_boundary =
              (MIN_PREFETCH_SIZE - chunk_addr % MIN_PREFETCH_SIZE) / m_config.page_size;
          const int chunk = std::min(pages_to_boundary, range_count - processed);
          std::list<mem_addr_t>::iterator chunk_end = chunk_start;
          std::advance(chunk_end, chunk);

          std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
          p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(*chunk_start);
          p_t->decomp_band = m_config.decomp_band_scalar;
          p_t->page_list = std::list<mem_addr_t>(chunk_start, chunk_end);
          p_t->size = p_t->page_list.size() * m_config.page_size;
          p_t->start_cycle = cur_cycle;
          p_t->type = latency_type::DECOMP;

          // comp_size 계산: 64K(16페이지)보다 작으면 get_tree_comp_size(64K) / (16/실제페이지수)
          if (chunk < 16) {
            // 64K 기준 압축 크기
            size_t base_comp_size = get_tree_comp_size(p_t->start_addr);
            p_t->comp_size = base_comp_size * chunk / 16;
          } else {
            p_t->comp_size = get_tree_comp_size(p_t->start_addr);
          }

          // leaf_size 설정 (MICRO 모드에서는 list 단위가 64KB)
          p_t->leaf_size = micro_leaf(get_lp_node(p_t->start_addr));
          for (auto iter = p_t->page_list.begin(); iter != p_t->page_list.end(); iter++) {
            compressed_pages_decompressing(*iter, p_t->leaf_size);
          }
          num_decomp_stage_queue += (p_t->size - p_t->comp_size) / m_config.page_size;

          // Consume every fault page in this request, including multiple faults
          // in the same chunk, so later requests are classified correctly.
          bool has_comp_fault = false;
          while (pf_iter != all_pf_iter->end() &&
                 std::find(p_t->page_list.begin(), p_t->page_list.end(),
                           *pf_iter) != p_t->page_list.end()) {
            has_comp_fault = true;
            if (m_config.enable_accurate_simulation) {
              m_new_stats->comp_fault_num++;
            }
            ++pf_iter;
          }
          p_t->mf_wait_for_comp = has_comp_fault;
          // Demand and proactive restores both pay CRM handling at enqueue.
          p_t->ready_cycle = cur_cycle;
          reserve_crm_handling(p_t);
          decomp_stage_queue.push_back(p_t);

          processed += chunk;
        }

        ++pg_iter;
      }
    }
    for (std::map<mem_addr_t, std::list<mem_fetch *>>::iterator iter2 =
             page_comp_this_turn.begin();
         iter2 != page_comp_this_turn.end(); iter2++) {
      comp_info[iter2->first] = iter2->second;
    }
  }
}
void gmmu_t::do_hardware_prefetch(
    std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn) {
  if (page_fault_this_turn.empty()) return;

  // 벡터 사용으로 캐시 지역성 및 할당 오버헤드 감소
  std::vector<std::vector<mem_addr_t>> all_transfer_all_page;
  std::vector<std::vector<mem_addr_t>> all_transfer_faulty_pages;
  std::vector<std::vector<mem_addr_t>> all_transfer_comp_pages;
  std::map<mem_addr_t, std::list<mem_fetch *>> temp_req_info;
  std::map<mem_addr_t, std::list<mem_fetch *>> temp_req_info_comp;

  if (prefetcher == hwardware_prefetcher::DISBALED ||
      prefetcher == hwardware_prefetcher::RANDOM) {
    for (const auto& it : page_fault_this_turn) {
      std::vector<mem_addr_t> temp_pages{it.first};

      mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(it.first);
      struct lp_tree_node *root = get_lp_node(page_addr);
      update_basic_block(root, page_addr, m_config.page_size, true);

      all_transfer_all_page.push_back(temp_pages);
      all_transfer_faulty_pages.push_back(temp_pages);

      temp_req_info[it.first];

      if (prefetcher == hwardware_prefetcher::RANDOM) {
        struct lp_tree_node *root2 = get_lp_node(page_addr);
        size_t npages = root2->size / m_config.page_size;
        if (npages > 0) {
          size_t random_idx = rand() % npages;
          mem_addr_t prefetch_addr = root2->addr + random_idx * m_config.page_size;
          mem_addr_t prefetch_page_num = m_gpu->get_global_memory()->get_page_num(prefetch_addr);

          if (!m_gpu->get_global_memory()->is_valid(prefetch_page_num) &&
              !m_gpu->get_global_memory()->is_iter_compressed(prefetch_page_num) &&
              page_fault_this_turn.find(prefetch_addr) == page_fault_this_turn.end() &&
              temp_req_info.find(prefetch_page_num) == temp_req_info.end() &&
              req_info.find(prefetch_page_num) == req_info.end()) {
            mem_addr_t page_addr2 = m_gpu->get_global_memory()->get_mem_addr(prefetch_page_num);
            struct lp_tree_node *root3 = get_lp_node(page_addr2);
            update_basic_block(root3, page_addr2, m_config.page_size, true);

            all_transfer_all_page.back().push_back(prefetch_page_num);
            temp_req_info[prefetch_page_num];
          }
        }
      }
    }
  } else {
    // TBN prefetcher
    std::map<mem_addr_t, std::vector<mem_addr_t>> lp_pf_groups;

    for (const auto& pf : page_fault_this_turn) {
      mem_addr_t page_addr = m_gpu->get_global_memory()->get_mem_addr(pf.first);
      struct lp_tree_node *root = get_lp_node(page_addr);
      lp_pf_groups[root->addr].push_back(page_addr);
    }

    for (auto& lp_pf_pair : lp_pf_groups) {
      std::set<mem_addr_t> schedulable_basic_blocks; // set 사용
      std::vector<mem_addr_t> cur_transfer_all_pages;
      std::vector<mem_addr_t> cur_transfer_faulty_pages;
      std::vector<mem_addr_t> cur_transfer_comp_pages;

      struct lp_tree_node *root = get_lp_node(lp_pf_pair.first);

      for (mem_addr_t page_addr : lp_pf_pair.second) {
        struct lp_tree_node *page_root = get_lp_node(page_addr);
        mem_addr_t bb_addr = get_basic_block(page_root, page_addr);
        update_basic_block(page_root, bb_addr, page_root->leaf_size, true);
        if(page_root->leaf_size==PREFETCH_16K_SIZE){//16K block
          schedulable_basic_blocks.insert(bb_addr);
        }
        else{
          size_t num_blocks = (page_root->leaf_size < page_root->size ? 
                             page_root->leaf_size : page_root->size) / PREFETCH_16K_SIZE;//leaf보다 size가 클 수 있음
          mem_addr_t base_addr = bb_addr;
          auto global_mem = m_gpu->get_global_memory();
          
          for(size_t i = 0; i < num_blocks; i++) {
            mem_addr_t curr_addr = base_addr + i * PREFETCH_16K_SIZE;
            if(!global_mem->is_valid(global_mem->get_page_num(curr_addr))) {
              schedulable_basic_blocks.insert(curr_addr);//include invalid page and compressed page
            }
          }
        }
        cur_transfer_faulty_pages.push_back(m_gpu->get_global_memory()->get_page_num(page_addr));
      }

      if (prefetcher == hwardware_prefetcher::TBN) {
        traverse_and_fill_lp_tree(root, schedulable_basic_blocks);
      }
      if(schedulable_basic_blocks.empty()&& !cur_transfer_faulty_pages.empty()){

        print_debug_all(true);
      }
      for (mem_addr_t bb : schedulable_basic_blocks) {//16K단위 compressed or invalid pages
        block_access_list.emplace_back(m_gpu->gpu_tot_sim_cycle + m_gpu->gpu_sim_cycle, bb);
        std::list<mem_addr_t> all_block_pages = m_gpu->get_global_memory()->get_faulty_pages(bb, PREFETCH_16K_SIZE);
        std::list<mem_addr_t> compped_block_pages = m_gpu->get_global_memory()->get_compressed_pages(bb, PREFETCH_16K_SIZE);

        for (mem_addr_t pg : all_block_pages) {
          if (temp_req_info.find(pg) == temp_req_info.end()) {
            temp_req_info[pg];
            cur_transfer_all_pages.push_back(pg);
          }
        }
        if (m_config.compression_policy == 0) {
          assert(compped_block_pages.empty() && "bst debug no compression policy must not have compressed page\n");
        }
        for (mem_addr_t pg : compped_block_pages) {
          if (temp_req_info_comp.find(pg) == temp_req_info_comp.end()) {
            temp_req_info_comp[pg];
            cur_transfer_comp_pages.push_back(pg);
          }
        }
      }

      all_transfer_all_page.push_back(std::move(cur_transfer_all_pages));
      all_transfer_faulty_pages.push_back(std::move(cur_transfer_faulty_pages));
      all_transfer_comp_pages.push_back(std::move(cur_transfer_comp_pages));
    }
  }

  // temp_req_info를 req_info에 병합
  for (auto& iter : temp_req_info) {
    req_info[iter.first].merge(iter.second);
  }
  for (auto& iter : temp_req_info_comp) {
    comp_info[iter.first].merge(iter.second);
  }

  // 압축 페이지를 연속적인 그룹으로 나누어 16개 이하 단위로 처리
  constexpr size_t CHUNK_SIZE_PAGES = 16;
  for (const auto& comp_pages : all_transfer_comp_pages) {
    if (comp_pages.empty()) continue;
    
    // 연속적인 페이지들을 그룹으로 나누기
    std::vector<std::vector<mem_addr_t>> continuous_groups;
    std::vector<mem_addr_t> current_group;
    current_group.push_back(comp_pages[0]);
    
    for (size_t i = 1; i < comp_pages.size(); i++) {
      // 현재 페이지가 이전 페이지와 연속적인지 확인
      if (comp_pages[i] == comp_pages[i-1] + 1) {
        current_group.push_back(comp_pages[i]);
      } else {
        // 연속적이지 않으면 현재 그룹을 저장하고 새 그룹 시작
        continuous_groups.push_back(std::move(current_group));
        current_group.clear();
        current_group.push_back(comp_pages[i]);
      }
    }
    // 마지막 그룹 추가
    if (!current_group.empty()) {
      continuous_groups.push_back(std::move(current_group));
    }
    
    // 각 연속적인 그룹을 16개 이하 청크(64K)로 나누어 처리
    for (const auto& group : continuous_groups) {
      for (size_t start = 0; start < group.size();) {
        const mem_addr_t chunk_addr = m_gpu->get_global_memory()->get_mem_addr(group[start]);
        const size_t pages_to_boundary =
            (MIN_PREFETCH_SIZE - chunk_addr % MIN_PREFETCH_SIZE) / m_config.page_size;
        const size_t end = std::min(start + pages_to_boundary, group.size());
        
        std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
        p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(group[start]);
        p_t->page_list.assign(group.begin() + start, group.begin() + end);
        p_t->size = p_t->page_list.size() * m_config.page_size;
        p_t->leaf_size = micro_leaf(get_lp_node(p_t->start_addr));
        p_t->start_cycle = cur_cycle;
        p_t->decomp_band = m_config.decomp_band_scalar;
        
        // 16개 이상이면 기존 방식, 16개 미만이면 해당 개수로 get_tree_comp_size 사용
        if (p_t->page_list.size() == CHUNK_SIZE_PAGES) {
          p_t->comp_size = get_tree_comp_size(p_t->start_addr);
        } else {
          p_t->comp_size = get_tree_comp_size(p_t->start_addr)*p_t->page_list.size()/CHUNK_SIZE_PAGES;
        }
        p_t->type = latency_type::DECOMP;
        num_decomp_stage_queue += (p_t->size - p_t->comp_size) / m_config.page_size;
        reserve_crm_handling(p_t);
        decomp_stage_queue.push_back(p_t);
        for (auto pg : p_t->page_list) {
          compressed_pages_decompressing(pg, p_t->leaf_size);
        }
        start = end;
      }
    }
  }

  // 전체 페이지 전송 및 페이지 폴트 처리
  size_t num_blocks = all_transfer_all_page.size();
  for (size_t blk = 0; blk < num_blocks; ++blk) {
    const std::vector<mem_addr_t>& all_pages = all_transfer_all_page[blk];
    const std::vector<mem_addr_t>& pf_pages = all_transfer_faulty_pages[blk];
    size_t all_sz = all_pages.size(), pf_sz = pf_pages.size();
    size_t pg_idx = 0, pf_idx = 0;

    // 페이지 폴트가 없으면 전체 블록을 한 번에 push
    if (pf_sz == 0 && all_sz > 0) {
      std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
      p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(all_pages.front());
      p_t->page_list.assign(all_pages.begin(), all_pages.end());
      p_t->size = p_t->page_list.size() * m_config.page_size;
      p_t->leaf_size = get_lp_node(p_t->start_addr)->leaf_size;
      p_t->start_cycle = cur_cycle;
      p_t->type = latency_type::PCIE_READ;
      p_t->decomp_band = m_config.decomp_band_scalar;
      reserve_crm_handling(p_t);
      pcie_read_stage_queue.push_back(p_t);
      num_read_stage_queue += p_t->page_list.size();
      continue;
    }

    // 아니면, 모든 페이지를 순회하며 page fault나 불연속에서 분할
    size_t chunk_start = 0;
    while (pg_idx < all_sz) {
      // 불연속 페이지 체크
      if (pg_idx > chunk_start && all_pages[pg_idx] != all_pages[pg_idx - 1] + 1) {
        if (pg_idx > chunk_start) {
          std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
          p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(all_pages[chunk_start]);
          p_t->page_list.assign(all_pages.begin() + chunk_start, all_pages.begin() + pg_idx);
          p_t->size = p_t->page_list.size() * m_config.page_size;
          p_t->leaf_size = get_lp_node(p_t->start_addr)->leaf_size;
          p_t->start_cycle = cur_cycle;
          p_t->type = latency_type::PCIE_READ;
          p_t->decomp_band = m_config.decomp_band_scalar;
          reserve_crm_handling(p_t);
          pcie_read_stage_queue.push_back(p_t);
          num_read_stage_queue += p_t->page_list.size();
        }
        chunk_start = pg_idx;
      }

      // page fault 발생 시
      if (pf_idx < pf_sz && all_pages[pg_idx] == pf_pages[pf_idx]) {
        // page fault 이전까지의 청크 처리
        if (pg_idx > chunk_start) {
          std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
          p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(all_pages[chunk_start]);
          p_t->page_list.assign(all_pages.begin() + chunk_start, all_pages.begin() + pg_idx);
          p_t->size = p_t->page_list.size() * m_config.page_size;
          p_t->leaf_size = get_lp_node(p_t->start_addr)->leaf_size;
          p_t->start_cycle = cur_cycle;
          p_t->type = latency_type::PCIE_READ;
          p_t->decomp_band = m_config.decomp_band_scalar;
          reserve_crm_handling(p_t);
          pcie_read_stage_queue.push_back(p_t);
          num_read_stage_queue += p_t->page_list.size();
        }
        
        // page fault 이벤트 추가
        if (m_config.enable_accurate_simulation) {
          std::shared_ptr<pcie_latency_t> f_t = std::make_shared<pcie_latency_t>();
          f_t->page_list.push_back(pf_pages[pf_idx]);
          f_t->type = latency_type::PAGE_FAULT;
          reserve_crm_handling(f_t);
          pcie_read_stage_queue.push_back(f_t);

          num_read_stage_queue += 1;
        }
        
        // page fault 페이지도 일반 전송으로 처리
        std::shared_ptr<pcie_latency_t> pf_read_t = std::make_shared<pcie_latency_t>();
        pf_read_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(all_pages[pg_idx]);
        pf_read_t->page_list.push_back(all_pages[pg_idx]);
        pf_read_t->size = m_config.page_size;
        pf_read_t->leaf_size = get_lp_node(pf_read_t->start_addr)->leaf_size;
        pf_read_t->start_cycle = cur_cycle;
        pf_read_t->type = latency_type::PCIE_READ;
        pf_read_t->decomp_band = m_config.decomp_band_scalar;
        reserve_crm_handling(pf_read_t);
        pcie_read_stage_queue.push_back(pf_read_t);
        num_read_stage_queue += 1;
        
        chunk_start = pg_idx + 1;  // 다음 청크 시작점 설정
        ++pf_idx;
      }
      ++pg_idx;
    }
    // 남은 페이지 처리
    if (chunk_start < all_sz) {
      std::shared_ptr<pcie_latency_t> p_t = std::make_shared<pcie_latency_t>();
      p_t->start_addr = m_gpu->get_global_memory()->get_mem_addr(all_pages[chunk_start]);
      p_t->page_list.assign(all_pages.begin() + chunk_start, all_pages.end());
      p_t->size = p_t->page_list.size() * m_config.page_size;
      p_t->leaf_size = get_lp_node(p_t->start_addr)->leaf_size;
      p_t->start_cycle = cur_cycle;
      p_t->type = latency_type::PCIE_READ;
      p_t->decomp_band = m_config.decomp_band_scalar;
      reserve_crm_handling(p_t);
      pcie_read_stage_queue.push_back(p_t);
      num_read_stage_queue += p_t->page_list.size();
    }
  }

  // 페이지 폴트 정보 및 통계 갱신
  for (const auto& iter2 : page_fault_this_turn) {
    assert(req_info[iter2.first].size() == 0);
    req_info[iter2.first] = iter2.second;
    m_new_stats->mf_page_fault_outstanding++;
    m_new_stats->mf_page_fault_pending += req_info[iter2.first].size() - 1;
    m_new_stats->mf_page_fault_latency[iter2.first].push_back(cur_cycle);
  }
}

void gpgpu_sim::cycle() {
  int clock_mask = next_clock_domain();

  // the gmmu has the same clock as the core
  if (clock_mask & GMMU) {
    m_gmmu->cycle();
  }
  throttle_SM();
  // skip cycles because all warps stall to wait for mem_fetch come back from
  // gmmu
  if (skip_cycles_enable && skip_cycles) {
    if (clock_mask & CORE) {
      skipped_cycles++;
      gpu_sim_cycle++;
    }
    return;
  }

  if (clock_mask & CORE) {
    // shader core loading (pop from ICNT into core) follows CORE clock
    for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++)
      m_cluster[i]->icnt_cycle();
  }
  unsigned partiton_replys_in_parallel_per_cycle = 0;
  if (clock_mask & ICNT) {
    // pop from memory controller to interconnect
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      mem_fetch *mf = m_memory_sub_partition[i]->top();
      if (mf) {
        unsigned response_size =
            mf->get_is_write() ? mf->get_ctrl_size() : mf->size();
        if (::icnt_has_buffer(m_shader_config->mem2device(i), response_size)) {
          // if (!mf->get_is_write())
          mf->set_return_timestamp(gpu_sim_cycle + gpu_tot_sim_cycle);
          mf->set_status(IN_ICNT_TO_SHADER, gpu_sim_cycle + gpu_tot_sim_cycle);
          ::icnt_push(m_shader_config->mem2device(i), mf->get_tpc(), mf,
                      response_size);
          m_memory_sub_partition[i]->pop();
          partiton_replys_in_parallel_per_cycle++;
        } else {
          gpu_stall_icnt2sh++;
        }
      } else {
        m_memory_sub_partition[i]->pop();
      }
    }
  }
  partiton_replys_in_parallel += partiton_replys_in_parallel_per_cycle;

  if (clock_mask & DRAM) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m_memory_config->simple_dram_model)
        m_memory_partition_unit[i]->simple_dram_model_cycle();
      else
        m_memory_partition_unit[i]
            ->dram_cycle();  // Issue the dram command (scheduler + delay model)
      // Update performance counters for DRAM
      m_memory_partition_unit[i]->set_dram_power_stats(
          m_power_stats->pwr_mem_stat->n_cmd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_activity[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_nop[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_act[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_pre[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_rd[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_wr_WB[CURRENT_STAT_IDX][i],
          m_power_stats->pwr_mem_stat->n_req[CURRENT_STAT_IDX][i]);
    }
  }

  // L2 operations follow L2 clock domain
  unsigned partiton_reqs_in_parallel_per_cycle = 0;
  if (clock_mask & L2) {
    m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX].clear();
    for (unsigned i = 0; i < m_memory_config->m_n_mem_sub_partition; i++) {
      // move memory request from interconnect into memory partition (if not
      // backed up) Note:This needs to be called in DRAM clock domain if there
      // is no L2 cache in the system In the worst case, we may need to push
      // SECTOR_CHUNCK_SIZE requests, so ensure you have enough buffer for them
      if (m_memory_sub_partition[i]->full(SECTOR_CHUNCK_SIZE)) {
        gpu_stall_dramfull++;
      } else {
        mem_fetch *mf = (mem_fetch *)icnt_pop(m_shader_config->mem2device(i));
        m_memory_sub_partition[i]->push(mf, gpu_sim_cycle + gpu_tot_sim_cycle);
        if (mf) partiton_reqs_in_parallel_per_cycle++;
      }
      m_memory_sub_partition[i]->cache_cycle(gpu_sim_cycle + gpu_tot_sim_cycle);
      m_memory_sub_partition[i]->accumulate_L2cache_stats(
          m_power_stats->pwr_mem_stat->l2_cache_stats[CURRENT_STAT_IDX]);
    }
  }
  partiton_reqs_in_parallel += partiton_reqs_in_parallel_per_cycle;
  if (partiton_reqs_in_parallel_per_cycle > 0) {
    partiton_reqs_in_parallel_util += partiton_reqs_in_parallel_per_cycle;
    gpu_sim_cycle_parition_util++;
  }

  if (clock_mask & ICNT) {
    icnt_transfer();
  }

  if (clock_mask & CORE) {
    all_warps.clear();
    fail_warps.clear();
    // L1 cache + shader core pipeline stages
    unsigned long long active_unthrottled = 0, total_unthrottled = 0;
    m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX].clear();
  unsigned last = m_last_fetched_cluster;
unsigned num_clusters = m_shader_config->n_simt_clusters;
unsigned throttle_sm_count = 0;
unsigned after_issue_active_sms=0;
for (unsigned j = 0; j < num_clusters; ++j) {
    unsigned i = (last + j) % num_clusters;
    if (m_cluster[i]->get_not_completed() || get_more_cta_left()) {
        m_cluster[i]->core_cycle(throttle_sm_count++>=num_throttled_SM);//if throttle_sm_count is greater than num_throttled_SM, then the core is throttled
        *active_sms += m_cluster[i]->get_n_active_sms();
        after_issue_active_sms+=m_cluster[i]->get_n_hardware_active_sms();
    }
    // Update power stats
    m_cluster[i]->get_icnt_stats(
        m_power_stats->pwr_mem_stat->n_simt_to_mem[CURRENT_STAT_IDX][i],
        m_power_stats->pwr_mem_stat->n_mem_to_simt[CURRENT_STAT_IDX][i]);
    m_cluster[i]->get_cache_stats(
        m_power_stats->pwr_mem_stat->core_cache_stats[CURRENT_STAT_IDX]);

    // Update m_last_fetched_cluster to the first throttled cluster
    if (throttle_sm_count==num_throttled_SM) {
      m_last_fetched_cluster = i;
    }

    m_cluster[i]->get_current_occupancy(
        gpu_occupancy.aggregate_warp_slot_filled,
        gpu_occupancy.aggregate_theoretical_warp_slots);
}
  if(m_config.m_shader_config.CURE_throttle_SM)
  {
    unsigned idle_sms=m_shader_config->n_simt_clusters-(unsigned)(after_issue_active_sms);
    //잉여한 SM수는 전체 클러스터 수 - 현재 사용중인 SM수(Throttle된 것보다 작업이 있는 SM수가 더 적을 수 있음)
    m_gmmu->max_parallel_comp_requests = idle_sms/2; 
  }
  gpu_occupancy.aggregate_warp_slot_filled += m_gmmu->get_comp_latency_queue_size() * 32 * 0.96;

    float temp = 0;
    for (unsigned i = 0; i < m_shader_config->num_shader(); i++) {
      temp += m_shader_stats->m_pipeline_duty_cycle[i];
    }
    temp = temp / m_shader_config->num_shader();
    *average_pipeline_duty_cycle = ((*average_pipeline_duty_cycle) + temp);
    // cout<<"Average pipeline duty cycle:
    // "<<*average_pipeline_duty_cycle<<endl;

    if (g_single_step &&
        ((gpu_sim_cycle + gpu_tot_sim_cycle) >= g_single_step)) {
      raise(SIGTRAP);  // Debug breakpoint
    }
    gpu_sim_cycle++;

    if (g_interactive_debugger_enabled) gpgpu_debug();

    // McPAT main cycle (interface with McPAT)
#ifdef GPGPUSIM_POWER_MODEL
    if (m_config.g_power_simulation_enabled) {
      if (m_config.g_power_simulation_mode == 0) {
        mcpat_cycle(m_config, getShaderCoreConfig(), m_gpgpusim_wrapper,
                    m_power_stats, m_config.gpu_stat_sample_freq,
                    gpu_tot_sim_cycle, gpu_sim_cycle, gpu_tot_sim_insn,
                    gpu_sim_insn, m_config.g_dvfs_enabled);
      }
    }
#endif

    issue_block2core();
    decrement_kernel_latency();

    // Depending on configuration, invalidate the caches once all of threads are
    // completed.
    int all_threads_complete = 1;
    if (m_config.gpgpu_flush_l1_cache) {
      for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
        if (m_cluster[i]->get_not_completed() == 0)
          m_cluster[i]->cache_invalidate();
        else
          all_threads_complete = 0;
      }
    }

    if (m_config.gpgpu_flush_l2_cache) {
      if (!m_config.gpgpu_flush_l1_cache) {
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_cluster[i]->get_not_completed() != 0) {
            all_threads_complete = 0;
            break;
          }
        }
      }

      if (all_threads_complete && !m_memory_config->m_L2_config.disabled()) {
        printf("Flushed L2 caches...\n");
        if (m_memory_config->m_L2_config.get_num_lines()) {
          int dlc = 0;
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
            dlc = m_memory_sub_partition[i]->flushL2();
            assert(dlc == 0);  // TODO: need to model actual writes to DRAM here
            printf("Dirty lines flushed from L2 %d is %d\n", i, dlc);
          }
        }
      }
    }

    if (!(gpu_sim_cycle % m_config.gpu_stat_sample_freq)) {
      time_t days, hrs, minutes, sec;
      time_t curr_time;
      time(&curr_time);
      unsigned long long elapsed_time =
          MAX(curr_time - gpgpu_ctx->the_gpgpusim->g_simulation_starttime, 1);
      if ((elapsed_time - last_liveness_message_time) >=
              m_config.liveness_message_freq &&
          DTRACE(LIVENESS)) {
        days = elapsed_time / (3600 * 24);
        hrs = elapsed_time / 3600 - 24 * days;
        minutes = elapsed_time / 60 - 60 * (hrs + 24 * days);
        sec = elapsed_time - 60 * (minutes + 60 * (hrs + 24 * days));

        unsigned long long active = 0, total = 0;
        unsigned long long active_unthrottled = 0, total_unthrottled = 0;
        size_t current_comp_size = m_gmmu->get_comp_latency_queue_size();
        for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
          if (m_config.m_shader_config.CURE_throttle_SM && i >= num_throttled_SM) {
            // 비활성화된 SM들: 이전 워프 수 기반으로 96% occupancy 반영
            if(current_comp_size > 0){
              unsigned long long simulated_active = (unsigned long long)(total_unthrottled * 0.96);
              active += simulated_active;
              total += total_unthrottled;
              current_comp_size-=1;
              }
              else{
                total+=total_unthrottled;
              }
          } else {
            // 활성화된 SM들: 정상적인 occupancy 계산
            unsigned long long prev_active = active, prev_total = total;            
            m_cluster[i]->get_current_occupancy(active, total);
            active_unthrottled = prev_active-active;
            total_unthrottled = prev_total-total;
          }
        }
        DPRINTFG(LIVENESS,
                 "uArch: inst.: %lld (ipc=%4.1f, occ=%0.4f\% [%llu / %llu]) "
                 "sim_rate=%u (inst/sec) elapsed = %u:%u:%02u:%02u / %s",
                 gpu_tot_sim_insn + gpu_sim_insn,
                 (double)gpu_sim_insn / (double)gpu_sim_cycle,
                 float(active) / float(total) * 100, active, total,
                 (unsigned)((gpu_tot_sim_insn + gpu_sim_insn) / elapsed_time),
                 (unsigned)days, (unsigned)hrs, (unsigned)minutes,
                 (unsigned)sec, ctime(&curr_time));
        fflush(stdout);
        last_liveness_message_time = elapsed_time;
      }
      visualizer_printstat();
      m_memory_stats->memlatstat_lat_pw();
      if (m_config.gpgpu_runtime_stat &&
          (m_config.gpu_runtime_stat_flag != 0)) {
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_BW_STAT) {
          for (unsigned i = 0; i < m_memory_config->m_n_mem; i++)
            m_memory_partition_unit[i]->print_stat(stdout);
          printf("maxmrqlatency = %d \n", m_memory_stats->max_mrq_latency);
          printf("maxmflatency = %d \n", m_memory_stats->max_mf_latency);
        }
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SHD_INFO)
          shader_print_runtime_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_L1MISS)
          shader_print_l1_miss_stat(stdout);
        if (m_config.gpu_runtime_stat_flag & GPU_RSTAT_SCHED)
          shader_print_scheduler_stat(stdout, false);
      }
    }

    /*if (!(gpu_sim_cycle % 50000)) {
      // deadlock detection
      if (m_config.gpu_deadlock_detect && gpu_sim_insn == last_gpu_sim_insn) {
        gpu_deadlock = true;
      } else {
        last_gpu_sim_insn = gpu_sim_insn;
      }
    }*/
    // remove deadlock_detection
    try_snap_shot(gpu_sim_cycle);
    spill_log_to_file(stdout, 0, gpu_sim_cycle);

#if (CUDART_VERSION >= 5000)
    // launch device kernel
    gpgpu_ctx->device_runtime->launch_one_device_kernel();
#endif
  }
}

void shader_core_ctx::dump_warp_state(FILE *fout) const {
  fprintf(fout, "\n");
  fprintf(fout, "per warp functional simulation status:\n");
  for (unsigned w = 0; w < m_config->max_warps_per_shader; w++)
    m_warp[w]->print(fout);
}

void gpgpu_sim::perf_memcpy_to_gpu(size_t dst_start_addr, size_t count) {
  if (m_memory_config->m_perf_sim_memcpy) {
    // if(!m_config.trace_driven_mode)    //in trace-driven mode, CUDA runtime
    // can start nre data structure at any position 	assert (dst_start_addr %
    // 32
    //== 0);

    for (unsigned counter = 0; counter < count; counter += 32) {
      const unsigned wr_addr = dst_start_addr + counter;
      addrdec_t raw_addr;
      mem_access_sector_mask_t mask;
      mask.set(wr_addr % 128 / 32);
      m_memory_config->m_address_mapping.addrdec_tlx(wr_addr, &raw_addr);
      const unsigned partition_id =
          raw_addr.sub_partition /
          m_memory_config->m_n_sub_partition_per_memory_channel;
      m_memory_partition_unit[partition_id]->handle_memcpy_to_gpu(
          wr_addr, raw_addr.sub_partition, mask);
    }
  }
}

void gpgpu_sim::dump_pipeline(int mask, int s, int m) const {
  /*
     You may want to use this function while running GPGPU-Sim in gdb.
     One way to do that is add the following to your .gdbinit file:

        define dp
           call g_the_gpu.dump_pipeline_impl((0x40|0x4|0x1),$arg0,0)
        end

     Then, typing "dp 3" will show the contents of the pipeline for shader
     core 3.
  */

  printf("Dumping pipeline state...\n");
  if (!mask) mask = 0xFFFFFFFF;
  for (unsigned i = 0; i < m_shader_config->n_simt_clusters; i++) {
    if (s != -1) {
      i = s;
    }
    if (mask & 1)
      m_cluster[m_shader_config->sid_to_cluster(i)]->display_pipeline(
          i, stdout, 1, mask & 0x2E);
    if (s != -1) {
      break;
    }
  }
  if (mask & 0x10000) {
    for (unsigned i = 0; i < m_memory_config->m_n_mem; i++) {
      if (m != -1) {
        i = m;
      }
      printf("DRAM / memory controller %u:\n", i);
      if (mask & 0x100000) m_memory_partition_unit[i]->print_stat(stdout);
      if (mask & 0x1000000) m_memory_partition_unit[i]->visualize();
      if (mask & 0x10000000) m_memory_partition_unit[i]->print(stdout);
      if (m != -1) {
        break;
      }
    }
  }
  fflush(stdout);
}

const shader_core_config *gpgpu_sim::getShaderCoreConfig() {
  return m_shader_config;
}

const memory_config *gpgpu_sim::getMemoryConfig() { return m_memory_config; }

simt_core_cluster *gpgpu_sim::getSIMTCluster(int index) {
  return *(m_cluster + index);
}
gmmu_t *gpgpu_sim::getGmmu() { return m_gmmu; }
void gpgpu_sim::flush_l2_page(unsigned page_num) {
    for (unsigned i=0;i<m_memory_config->m_n_mem_sub_partition;i++){
        m_memory_sub_partition[i]->flushL2page(page_num * m_config.page_size, m_config.page_size);
    }
}