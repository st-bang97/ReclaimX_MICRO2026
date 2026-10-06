// Copyright (c) 2009-2021, Tor M. Aamodt, Wilson W.L. Fung, Vijay Kandiah,
// Nikos Hardavellas Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
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

#ifndef GPU_SIM_H
#define GPU_SIM_H
#include <stdio.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include "../abstract_hardware_model.h"
#include "../cuda-sim/memory.h"
#include "../option_parser.h"
#include "../trace.h"
#include "addrdec.h"
#include "gpu-cache.h"
#include "shader.h"
#include "Compressor.h"
#include <unordered_set>
// constants for statistics printouts
#define GPU_RSTAT_SHD_INFO 0x1
#define GPU_RSTAT_BW_STAT 0x2
#define GPU_RSTAT_WARP_DIS 0x4
#define GPU_RSTAT_DWF_MAP 0x8
#define GPU_RSTAT_L1MISS 0x10
#define GPU_RSTAT_PDOM 0x20
#define GPU_RSTAT_SCHED 0x40
#define GPU_MEMLATSTAT_MC 0x2

// constants for configuring merging of coalesced scatter-gather requests
#define TEX_MSHR_MERGE 0x4
#define CONST_MSHR_MERGE 0x2
#define GLOBAL_MSHR_MERGE 0x1

// clock constants
#define MhZ *1000000

#define CREATELOG 111
#define SAMPLELOG 222
#define DUMPLOG 333

class gpgpu_context;

extern tr1_hash_map<new_addr_type, unsigned> address_random_interleaving;

enum dram_ctrl_t { DRAM_FIFO = 0, DRAM_FRFCFS = 1 };

enum hw_perf_t {
  HW_BENCH_NAME = 0,
  HW_KERNEL_NAME,
  HW_L1_RH,
  HW_L1_RM,
  HW_L1_WH,
  HW_L1_WM,
  HW_CC_ACC,
  HW_SHRD_ACC,
  HW_DRAM_RD,
  HW_DRAM_WR,
  HW_L2_RH,
  HW_L2_RM,
  HW_L2_WH,
  HW_L2_WM,
  HW_NOC,
  HW_PIPE_DUTY,
  HW_NUM_SM_IDLE,
  HW_CYCLES,
  HW_VOLTAGE,
  HW_TOTAL_STATS
};

struct power_config {
  power_config() { 
    m_valid = true; 
    
    // Initialize pointer variables to NULL
    g_power_config_name = NULL;
    g_power_filename = NULL;
    g_power_trace_filename = NULL;
    g_metric_trace_filename = NULL;
    g_steady_state_tracking_filename = NULL;
    gpu_steady_state_definition = NULL;
    g_hw_perf_file_name = NULL;
    g_hw_perf_bench_name = NULL;
    gpu_nonlinear_model_config = NULL;
    
    // Initialize boolean variables
    g_power_simulation_enabled = false;
    g_power_trace_enabled = false;
    g_steady_power_levels_enabled = false;
    g_power_per_cycle_dump = false;
    g_power_simulator_debug = false;
    g_dvfs_enabled = false;
    g_aggregate_power_stats = false;
    g_use_nonlinear_model = false;
    
    // Initialize integer variables
    g_power_trace_zlevel = 0;
    g_power_simulation_mode = 0;
    
    // Initialize double variables
    gpu_steady_power_deviation = 0.0;
    gpu_steady_min_period = 0.0;
    gpu_idle_core_power = 0.0;
    gpu_min_inc_per_active_sm = 0.0;
    
    // Initialize array
    for (int i = 0; i < hw_perf_t::HW_TOTAL_STATS; i++) {
      accelwattch_hybrid_configuration[i] = false;
    }
  }
  void init() {
    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf1[1024];
    // snprintf(buf1, 1024, "accelwattch_power_report__%s.log", date);
    snprintf(buf1, 1024, "accelwattch_power_report.log");
    g_power_filename = strdup(buf1);
    char buf2[1024];
    snprintf(buf2, 1024, "gpgpusim_power_trace_report__%s.log.gz", date);
    g_power_trace_filename = strdup(buf2);
    char buf3[1024];
    snprintf(buf3, 1024, "gpgpusim_metric_trace_report__%s.log.gz", date);
    g_metric_trace_filename = strdup(buf3);
    char buf4[1024];
    snprintf(buf4, 1024, "gpgpusim_steady_state_tracking_report__%s.log.gz",
             date);
    g_steady_state_tracking_filename = strdup(buf4);
    // for(int i =0; i< hw_perf_t::HW_TOTAL_STATS; i++){
    //   accelwattch_hybrid_configuration[i] = 0;
    // }

    if (g_steady_power_levels_enabled) {
      sscanf(gpu_steady_state_definition, "%lf:%lf",
             &gpu_steady_power_deviation, &gpu_steady_min_period);
    }

    // NOTE: After changing the nonlinear model to only scaling idle core,
    // NOTE: The min_inc_per_active_sm is not used any more
    // if (g_use_nonlinear_model)
    //   sscanf(gpu_nonlinear_model_config, "%lf:%lf", &gpu_idle_core_power,
    //          &gpu_min_inc_per_active_sm);
  }
  void reg_options(class OptionParser *opp);

  char *g_power_config_name;

  bool m_valid;
  bool g_power_simulation_enabled;
  bool g_power_trace_enabled;
  bool g_steady_power_levels_enabled;
  bool g_power_per_cycle_dump;
  bool g_power_simulator_debug;
  char *g_power_filename;
  char *g_power_trace_filename;
  char *g_metric_trace_filename;
  char *g_steady_state_tracking_filename;
  int g_power_trace_zlevel;
  char *gpu_steady_state_definition;
  double gpu_steady_power_deviation;
  double gpu_steady_min_period;

  char *g_hw_perf_file_name;
  char *g_hw_perf_bench_name;
  int g_power_simulation_mode;
  bool g_dvfs_enabled;
  bool g_aggregate_power_stats;
  bool accelwattch_hybrid_configuration[hw_perf_t::HW_TOTAL_STATS];

  // Nonlinear power model
  bool g_use_nonlinear_model;
  char *gpu_nonlinear_model_config;
  double gpu_idle_core_power;
  double gpu_min_inc_per_active_sm;
};

class memory_config {
 public:
  memory_config(gpgpu_context *ctx) {
    m_valid = false;
    gpgpu_dram_timing_opt = NULL;
    gpgpu_L2_queue_config = NULL;
    write_queue_size_opt = NULL;
    gpgpu_ctx = ctx;
    
    // Initialize boolean variables
    m_L2_texure_only = false;
    l2_ideal = false;
    gpgpu_memlatency_stat = false;
    elimnate_rw_turnaround = false;
    dual_bus_interface = false;
    seperate_write_queue_enabled = false;
    m_perf_sim_memcpy = false;
    simple_dram_model = false;
    
    // Initialize unsigned variables
    gpgpu_frfcfs_dram_sched_queue_size = 0;
    gpgpu_dram_return_queue_size = 0;
    m_n_mem = 0;
    m_n_sub_partition_per_memory_channel = 0;
    m_n_mem_sub_partition = 0;
    gpu_n_mem_per_ctrlr = 0;
    rop_latency = 0;
    dram_latency = 0;
    
    // Initialize DRAM timing parameters
    tCCDL = 0;
    tRTPL = 0;
    tCCD = 0;
    tRRD = 0;
    tRCD = 0;
    tRCDWR = 0;
    tRAS = 0;
    tRP = 0;
    tRC = 0;
    tCDLR = 0;
    tWR = 0;
    CL = 0;
    WL = 0;
    BL = 0;
    tRTW = 0;
    tWTR = 0;
    tWTP = 0;
    busW = 0;
    nbkgrp = 0;
    bk_tag_length = 0;
    nbk = 0;
    data_command_freq_ratio = 0;
    dram_atom_size = 0;
    icnt_flit_size = 0;
    dram_bnk_indexing_policy = 0;
    dram_bnkgrp_indexing_policy = 0;
    gpgpu_frfcfs_dram_write_queue_size = 0;
    write_high_watermark = 0;
    write_low_watermark = 0;
    
    // Initialize enum
    scheduler_type = DRAM_FIFO;
  }
  void init(int comp_engine=0) {
    printf("bst debug memory_config init compression_engine = %d\n", comp_engine);
    if(comp_engine == 1){
      rop_latency += 6;
      dram_latency += 6;
    }
    assert(gpgpu_dram_timing_opt);
    if (strchr(gpgpu_dram_timing_opt, '=') == NULL) {
      // dram timing option in ordered variables (legacy)
      // Disabling bank groups if their values are not specified
      nbkgrp = 1;
      tCCDL = 0;
      tRTPL = 0;
      sscanf(gpgpu_dram_timing_opt, "%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d:%d",
             &nbk, &tCCD, &tRRD, &tRCD, &tRAS, &tRP, &tRC, &CL, &WL, &tCDLR,
             &tWR, &nbkgrp, &tCCDL, &tRTPL);
    } else {
      // named dram timing options (unordered)
      option_parser_t dram_opp = option_parser_create();

      option_parser_register(dram_opp, "nbk", OPT_UINT32, &nbk,
                             "number of banks", "");
      option_parser_register(dram_opp, "CCD", OPT_UINT32, &tCCD,
                             "column to column delay", "");
      option_parser_register(
          dram_opp, "RRD", OPT_UINT32, &tRRD,
          "minimal delay between activation of rows in different banks", "");
      option_parser_register(dram_opp, "RCD", OPT_UINT32, &tRCD,
                             "row to column delay", "");
      option_parser_register(dram_opp, "RAS", OPT_UINT32, &tRAS,
                             "time needed to activate row", "");
      option_parser_register(dram_opp, "RP", OPT_UINT32, &tRP,
                             "time needed to precharge (deactivate) row", "");
      option_parser_register(dram_opp, "RC", OPT_UINT32, &tRC, "row cycle time",
                             "");
      option_parser_register(dram_opp, "CDLR", OPT_UINT32, &tCDLR,
                             "switching from write to read (changes tWTR)", "");
      option_parser_register(dram_opp, "WR", OPT_UINT32, &tWR,
                             "last data-in to row precharge", "");

      option_parser_register(dram_opp, "CL", OPT_UINT32, &CL, "CAS latency",
                             "");
      option_parser_register(dram_opp, "WL", OPT_UINT32, &WL, "Write latency",
                             "");

      // Disabling bank groups if their values are not specified
      option_parser_register(dram_opp, "nbkgrp", OPT_UINT32, &nbkgrp,
                             "number of bank groups", "1");
      option_parser_register(
          dram_opp, "CCDL", OPT_UINT32, &tCCDL,
          "column to column delay between accesses to different bank groups",
          "0");
      option_parser_register(
          dram_opp, "RTPL", OPT_UINT32, &tRTPL,
          "read to precharge delay between accesses to different bank groups",
          "0");

      option_parser_delimited_string(dram_opp, gpgpu_dram_timing_opt, "=:;");
      fprintf(stdout, "DRAM Timing Options:\n");
      option_parser_print(dram_opp, stdout);
      option_parser_destroy(dram_opp);
    }
    if(comp_engine == 1){ //BPC de/compression 11 cycle buffering 4 cycle total 15 cycle
      CL+=15;
      WL+=15;
    }
    else if(comp_engine == 2){ //LCP compression
      //CL+=7; //this done in background
      WL+=5;
    }
    int nbkt = nbk / nbkgrp;
    unsigned i;
    for (i = 0; nbkt > 0; i++) {
      nbkt = nbkt >> 1;
    }
    bk_tag_length = i - 1;
    assert(nbkgrp > 0 && "Number of bank groups cannot be zero");
    tRCDWR = tRCD - (WL + 1);
    if (elimnate_rw_turnaround) {
      tRTW = 0;
      tWTR = 0;
    } else {
      tRTW = (CL + (BL / data_command_freq_ratio) + 2 - WL);
      tWTR = (WL + (BL / data_command_freq_ratio) + tCDLR);
    }
    tWTP = (WL + (BL / data_command_freq_ratio) + tWR);
    dram_atom_size =
        BL * busW * gpu_n_mem_per_ctrlr;  // burst length x bus width x # chips
                                          // per partition

    assert(m_n_sub_partition_per_memory_channel > 0);
    assert((nbk % m_n_sub_partition_per_memory_channel == 0) &&
           "Number of DRAM banks must be a perfect multiple of memory sub "
           "partition");
    m_n_mem_sub_partition = m_n_mem * m_n_sub_partition_per_memory_channel;
    fprintf(stdout, "Total number of memory sub partition = %u\n",
            m_n_mem_sub_partition);

    m_address_mapping.init(m_n_mem, m_n_sub_partition_per_memory_channel);
    m_L2_config.init(&m_address_mapping);

    m_valid = true;

    sscanf(write_queue_size_opt, "%d:%d:%d",
           &gpgpu_frfcfs_dram_write_queue_size, &write_high_watermark,
           &write_low_watermark);
  }
  void reg_options(class OptionParser *opp);

  bool m_valid;
  mutable l2_cache_config m_L2_config;
  bool m_L2_texure_only;

  char *gpgpu_dram_timing_opt;
  char *gpgpu_L2_queue_config;
  bool l2_ideal;
  unsigned gpgpu_frfcfs_dram_sched_queue_size;
  unsigned gpgpu_dram_return_queue_size;
  enum dram_ctrl_t scheduler_type;
  bool gpgpu_memlatency_stat;
  unsigned m_n_mem;
  unsigned m_n_sub_partition_per_memory_channel;
  unsigned m_n_mem_sub_partition;
  unsigned gpu_n_mem_per_ctrlr;

  unsigned rop_latency;
  unsigned dram_latency;

  // DRAM parameters

  unsigned tCCDL;  // column to column delay when bank groups are enabled
  unsigned tRTPL;  // read to precharge delay when bank groups are enabled for
                   // GDDR5 this is identical to RTPS, if for other DRAM this is
                   // different, you will need to split them in two

  unsigned tCCD;    // column to column delay
  unsigned tRRD;    // minimal time required between activation of rows in
                    // different banks
  unsigned tRCD;    // row to column delay - time required to activate a row
                    // before a read
  unsigned tRCDWR;  // row to column delay for a write command
  unsigned tRAS;    // time needed to activate row
  unsigned tRP;     // row precharge ie. deactivate row
  unsigned
      tRC;  // row cycle time ie. precharge current, then activate different row
  unsigned tCDLR;  // Last data-in to Read command (switching from write to
                   // read)
  unsigned tWR;    // Last data-in to Row precharge

  unsigned CL;    // CAS latency
  unsigned WL;    // WRITE latency
  unsigned BL;    // Burst Length in bytes (4 in GDDR3, 8 in GDDR5)
  unsigned tRTW;  // time to switch from read to write
  unsigned tWTR;  // time to switch from write to read
  unsigned tWTP;  // time to switch from write to precharge in the same bank
  unsigned busW;

  unsigned nbkgrp;  // number of bank groups (has to be power of 2)
  unsigned
      bk_tag_length;  // number of bits that define a bank inside a bank group

  unsigned nbk;

  bool elimnate_rw_turnaround;

  unsigned
      data_command_freq_ratio;  // frequency ratio between DRAM data bus and
                                // command bus (2 for GDDR3, 4 for GDDR5)
  unsigned
      dram_atom_size;  // number of bytes transferred per read or write command

  linear_to_raw_address_translation m_address_mapping;

  unsigned icnt_flit_size;

  unsigned dram_bnk_indexing_policy;
  unsigned dram_bnkgrp_indexing_policy;
  bool dual_bus_interface;

  bool seperate_write_queue_enabled;
  char *write_queue_size_opt;
  unsigned gpgpu_frfcfs_dram_write_queue_size;
  unsigned write_high_watermark;
  unsigned write_low_watermark;
  bool m_perf_sim_memcpy;
  bool simple_dram_model;

  gpgpu_context *gpgpu_ctx;
};

extern bool g_interactive_debugger_enabled;

class gpgpu_sim_config : public power_config,
                         public gpgpu_functional_sim_config {
 public:
  gpgpu_sim_config(gpgpu_context *ctx)
      : m_shader_config(ctx), m_memory_config(ctx) {
    m_valid = false;
    gpgpu_ctx = ctx;
  }
  void reg_options(class OptionParser *opp);
  void init(int comp_engine=0) {
    gpu_stat_sample_freq = 10000;
    gpu_runtime_stat_flag = 0;
    sscanf(gpgpu_runtime_stat, "%d:%x", &gpu_stat_sample_freq,
           &gpu_runtime_stat_flag);
    m_shader_config.init();
    ptx_set_tex_cache_linesize(m_shader_config.m_L1T_config.get_line_sz());
    printf("bst debug sim_init_perf before m_memory_config.rop_latency = %d\n", m_memory_config.rop_latency);

    m_memory_config.init(comp_engine);
    printf("bst debug sim_init_perf after m_memory_config.rop_latency = %d\n", m_memory_config.rop_latency);
    init_clock_domains();
    calc_comp_lat();
    power_config::init();
    Trace::init();

    // initialize file name if it is not set
    time_t curr_time;
    time(&curr_time);
    char *date = ctime(&curr_time);
    char *s = date;
    while (*s) {
      if (*s == ' ' || *s == '\t' || *s == ':') *s = '-';
      if (*s == '\n' || *s == '\r') *s = 0;
      s++;
    }
    char buf[1024];
    snprintf(buf, 1024, "gpgpusim_visualizer__%s.log.gz", date);
    g_visualizer_filename = strdup(buf);

    m_valid = true;
  }
  unsigned get_core_freq() const { return core_freq; }
  unsigned num_shader() const { return m_shader_config.num_shader(); }
  unsigned num_cluster() const { return m_shader_config.n_simt_clusters; }
  unsigned num_core_per_cluster() const {
    return m_shader_config.n_simt_cores_per_cluster;
  }
  unsigned get_max_concurrent_kernel() const { return max_concurrent_kernel; }
  unsigned checkpoint_option;

  size_t stack_limit() const { return stack_size_limit; }
  size_t heap_limit() const { return heap_size_limit; }
  size_t sync_depth_limit() const { return runtime_sync_depth_limit; }
  size_t pending_launch_count_limit() const {
    return runtime_pending_launch_count_limit;
  }
  bool get_evict_form() const { return evict_form; }
  bool flush_l1() const { return gpgpu_flush_l1_cache; }
  void convert_byte_string();
  int compression_engine;// 0 for none, 1 for LCP_BPC, 2 for LCP_BDI
  void calc_comp_lat();
  bool get_invalidate_clean()  { return invalidate_clean; }
  uint64_t get_page_fault_latency() const { return page_fault_latency; }
  unsigned get_compress_latency() const { return compress_latency; }
  bool change_comp_ratio;
  bool fault_based_LRU;
  bool two_level_LRU;
  bool forest_LRU;
  float static_comp_ratio;
 private:
  void init_clock_domains(void);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  bool m_valid;
  shader_core_config m_shader_config;
  memory_config m_memory_config;
  // clock domains - frequency
  double core_freq;
  double icnt_freq;
  double dram_freq;
  double l2_freq;
  double core_period;
  double icnt_period;
  double dram_period;
  double l2_period;
  // GPGPU-Sim timing model options
  unsigned long long gpu_max_cycle_opt;
  unsigned long long gpu_max_insn_opt;
  unsigned gpu_max_cta_opt;
  unsigned gpu_max_completed_cta_opt;
  char *gpgpu_runtime_stat;
  bool gpgpu_flush_l1_cache;
  bool gpgpu_flush_l2_cache;
  bool gpu_deadlock_detect;
  bool progress_print;
  int gpgpu_frfcfs_dram_sched_queue_size;
  int gpgpu_cflog_interval;
  char *gpgpu_clock_domains;
  unsigned max_concurrent_kernel;

  // visualizer
  bool g_visualizer_enabled;
  char *g_visualizer_filename;
  int g_visualizer_zlevel;

  // statistics collection
  int gpu_stat_sample_freq;
  int gpu_runtime_stat_flag;

  unsigned long long page_table_walk_latency;
  int eviction_policy;
  int tbn_threshold;
  bool full_comp_policy;
  bool comp_evict_policy;
  float comp_ratio;
  int compression_policy;
  bool selective_compression;
  bool evict_form;
  bool invalidate_clean;
  float reserve_accessed_page_percent;
  float free_page_buffer_percentage;
  float comp_buffer_percentage;
  float comp_evict_start_percentage;
  char *pcie_bandwidth_string;
  float pcie_bandwidth;
  float selective_compression_threshold;
  bool enable_LRU_compression;
  bool enable_LRU_decompression;
  float curve_a;
  float curve_b;

  int enable_dma;
  int multiply_dma_penalty;
  unsigned migrate_threshold;

  bool enable_smart_runtime;

  // Device Limits
  size_t stack_size_limit;
  size_t heap_size_limit;
  size_t runtime_sync_depth_limit;
  size_t runtime_pending_launch_count_limit;

  // gpu compute capability options
  unsigned int gpgpu_compute_capability_major;
  unsigned int gpgpu_compute_capability_minor;
  unsigned long long liveness_message_freq;

  friend class gpgpu_sim;

 public:
  int hardware_prefetch;
  int hwprefetch_oversub;
  bool collect_memory_access_stats;
  bool collect_access_trace;

  friend class gmmu_t;
};

extern unsigned long long kernel_time;
extern unsigned long long memory_copy_time_h2d;
extern unsigned long long memory_copy_time_d2h;
extern unsigned long long prefetch_time;
extern unsigned long long devicesync_time;
extern unsigned long long writeback_time;
extern unsigned long long dma_time;
extern unsigned long long comp_time;
extern unsigned long long decomp_time;
enum stats_type {
  prefetch = 0,
  prefetch_breakdown,
  memcpy_h2d,
  memcpy_d2h,
  memcpy_d2d,
  kernel_launched,
  page_fault,
  device_sync,
  write_back,
  invalidate,
  dma,
  comp,
  comp_evict_stat,
  comp_evict_inv,
  decomp
};

class event_stats {
 public:
  event_stats(enum stats_type t, unsigned long long s_time,
              unsigned long long e_time)
      : type(t), start_time(s_time), end_time(e_time) {}
  event_stats(enum stats_type t, unsigned long long s_time)
      : type(t), start_time(s_time), end_time(0) {}
  enum stats_type type;
  unsigned long long start_time;
  unsigned long long end_time;

  virtual void print(FILE *fout, float freq) = 0;
  virtual void calculate() = 0;
};

class memory_stats : public event_stats {
 public:
  memory_stats(enum stats_type t, unsigned long long s_time, mem_addr_t s_addr,
               size_t sz, unsigned s_id)
      : event_stats(t, s_time), start_addr(s_addr), size(sz), stream_id(s_id) {}
  memory_stats(enum stats_type t, unsigned long long s_time,
               unsigned long long e_time, mem_addr_t s_addr, size_t sz,
               unsigned s_id)
      : event_stats(t, s_time, e_time),
        start_addr(s_addr),
        size(sz),
        stream_id(s_id) {}
  mem_addr_t start_addr;
  size_t size;
  unsigned stream_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t St: %x Sz: %lu \t Sm: %u \t ",
            start_time, end_time, start_addr, size, stream_id);
    if (type == memcpy_h2d)
      fprintf(fout, "T: memcpy_h2d");
    else if (type == memcpy_d2h)
      fprintf(fout, "T: memcpy_d2h");
    else if (type == memcpy_d2d)
      fprintf(fout, "T: memcpy_d2d");
    else if (type == prefetch)
      fprintf(fout, "T: prefetch");
    else if (type == prefetch_breakdown)
      fprintf(fout, "T: prefetch_breakdown");
    else if (type == device_sync)
      fprintf(fout, "T: device_sync");
    else if (type == write_back)
      fprintf(fout, "T: write_back");
    else if (type == invalidate)
      fprintf(fout, "T: invalidate");
    else if (type == dma)
      fprintf(fout, "T: dma");
    else if (type == comp)
      fprintf(fout, "T: comp");
    else if (type == comp_evict_stat)
      fprintf(fout, "T: comp_evict");
    else if (type == comp_evict_inv)
      fprintf(fout, "T: comp_evict_inv");
    else if (type == decomp)
      fprintf(fout, "T: decomp");
    else
      fprintf(fout, "T: unknown, type: %d", type);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }
  virtual void calculate() {
    if (type == memcpy_h2d) {
      memory_copy_time_h2d += end_time - start_time;
    } else if (type == memcpy_d2h) {
      memory_copy_time_d2h += end_time - start_time;
    } else if (type == prefetch_breakdown) {
      prefetch_time += end_time - start_time;
    } else if (type == device_sync) {
      devicesync_time += end_time - start_time;
    } else if (type == write_back) {
      writeback_time += end_time - start_time;
    } else if (type == dma) {
      dma_time += end_time - start_time;
    } else if (type == comp) {
      comp_time += end_time - start_time;
    } else if (type == decomp) {
      decomp_time += end_time - start_time;
    }
  }
};

class kernel_stats : public event_stats {
 public:
  kernel_stats(unsigned long long s_time, unsigned s_id, unsigned k_id)
      : event_stats(kernel_launched, s_time),
        stream_id(s_id),
        kernel_id(k_id) {}
  unsigned stream_id;
  unsigned kernel_id;

  virtual void print(FILE *fout, float freq) {
    fprintf(
        fout,
        "F: %8llu----T: %8llu \t \t \t Kl: %u \t Sm: %u \t T: kernel_launch",
        start_time, end_time, kernel_id, stream_id);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() { kernel_time += end_time - start_time; }
};

class page_fault_stats : public event_stats {
 public:
  page_fault_stats(unsigned long long s_time, const std::list<mem_addr_t> &pgs,
                   unsigned sz)
      : event_stats(page_fault, s_time),
        pages(pgs),
        transfering_pages(pgs),
        size(sz) {}
  std::list<mem_addr_t> pages;
  std::list<mem_addr_t> transfering_pages;
  size_t size;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t Sz: %lu \t T: page_fault",
            start_time, end_time, size);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() {}
};
class page_comp_fault_stats : public event_stats {
 public:
  page_comp_fault_stats(unsigned long long s_time, const std::list<mem_addr_t> &pgs,
                   unsigned sz)
      : event_stats(page_fault, s_time),
        pages(pgs),
        transfering_pages(pgs),
        size(sz) {}
  std::list<mem_addr_t> pages;
  std::list<mem_addr_t> transfering_pages;
  size_t size;

  virtual void print(FILE *fout, float freq) {
    fprintf(fout, "F: %8llu----T: %8llu \t Sz: %lu \t T: comp_fault",
            start_time, end_time, size);
    fprintf(fout, "(%f)\n", ((float)(end_time - start_time)) / freq);
  }

  virtual void calculate() {}
};
// In order to skip idle cycles due to page fault, we need collect warp info
// in all cores.
extern std::list<shd_warp_t *> all_warps;
extern std::list<shd_warp_t *> fail_warps;
extern bool skip_cycles;
extern bool skip_cycles_enable;

extern std::map<unsigned long long, std::list<event_stats *>> sim_prof;

extern bool sim_prof_enable;

void print_sim_prof(FILE *fout, float freq);

void calculate_sim_prof(FILE *fout, gpgpu_sim *gpu);

void update_sim_prof_kernel(unsigned kernel_id, unsigned long long end_time);

void update_sim_prof_prefetch(mem_addr_t start_addr, size_t size,
                              unsigned long long end_time);

void update_sim_prof_prefetch_break_down(unsigned long long end_time);

void print_UVM_stats(gpgpu_new_stats *new_stats, gpgpu_sim *gpu, FILE *fout);

class access_info {
 public:
  mem_addr_t page_no;
  mem_addr_t mem_addr;
  size_t size;
  unsigned long long cycle;
  bool is_read;
  unsigned sm_id;
  unsigned warp_id;
  access_info(mem_addr_t p_n, mem_addr_t addr, size_t s, unsigned long long c,
              bool rw, unsigned s_id, unsigned w_id)
      : page_no(p_n),
        mem_addr(addr),
        size(s),
        cycle(c),
        is_read(rw),
        sm_id(s_id),
        warp_id(w_id) {}
};

class gpgpu_new_stats {
 public:
  gpgpu_new_stats(const gpgpu_sim_config &config);
  ~gpgpu_new_stats();
  void print(FILE *fout) const;
  void print_pcie(FILE *fout) const;
  void print_access_pattern_detail(FILE *fout) const;
  void print_access_pattern(FILE *fout) const;
  void print_time_and_access(FILE *fout) const;

  // for each shader of all global memory access

  // tlb hit
  unsigned long long *tlb_hit;
  // tlb miss
  unsigned long long *tlb_miss;

  // tlb validate
  unsigned long long *tlb_val;
  // tlb eviction
  unsigned long long *tlb_evict;
  // tlb invalidated by page eviction
  unsigned long long *tlb_page_evict;

  // in tlb miss, page hit
  unsigned long long *mf_page_hit;
  unsigned long long *mf_page_compressed;
  // in tlb miss, page miss
  unsigned long long *mf_page_miss;

  // in tlb miss, page miss, the first create fault
  unsigned long long mf_page_fault_outstanding;
  // in tlb miss, page miss, the following that appends to mshr
  unsigned long long mf_page_fault_pending;
  unsigned long long mf_page_comp_pending;
  unsigned long long page_evict_dirty;

  unsigned long long page_evict_not_dirty;

  // prefetch page hit
  unsigned long long pf_page_hit;
  // prefetch page miss
  unsigned long long pf_page_miss;
  // prefetch fault page size, large page and latency
  std::vector<std::pair<unsigned long, unsigned long long>> pf_fault_latency;

  // for each page, how many time is it being accessed by each shader
  std::map<mem_addr_t, unsigned> *page_access_times;
  bool collect_access_trace;
  bool collect_memory_access_stats;
  // for each timestamp, which page is being accessed
  std::list<access_info> time_and_page_access;

  // ready lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_read_utilization;
  // write lanes utilization
  std::list<std::pair<unsigned long long, float>> pcie_write_utilization;

  // page and its partern
  std::map<mem_addr_t, std::vector<bool>> page_thrashing;
  // tlb and its partern
  std::map<mem_addr_t, std::vector<bool>> page_compressed;
  // tlb and its partern
  std::map<mem_addr_t, std::vector<bool>> *tlb_thrashing;

  // for each shader, the memory access latency
  std::map<unsigned, std::pair<bool, unsigned long long>> *ma_latency;

  // for mf when it is fault(not pending to prefetch), the latency
  std::map<mem_addr_t, std::list<unsigned long long>> mf_page_fault_latency;

  // for prefetch each small page latency
  std::map<mem_addr_t, std::list<unsigned long long>> pf_page_fault_latency;
  std::map<mem_addr_t, std::list<unsigned long long>> comp_fault_latency;
  std::map<mem_addr_t, std::list<unsigned long long>> pf_comp_fault_latency;

  const gpgpu_sim_config &m_config;

  unsigned long long num_dma;
  unsigned long long dma_page_transfer_read;
  unsigned long long dma_page_transfer_write;
  unsigned long long comp_page_tansfer;
  unsigned long long comp_evcit_num;
  unsigned long long comp_fault_num;
  unsigned long long comp_evict_clean;
  unsigned long long inefficient_comp;
  unsigned long long comp_evict_dirty;
  // compress number
  unsigned long long compress;
  // decompress number
  unsigned long long decompress;
  std::list<size_t> comp_size_list;
};

// this class simulate the gmmu unit on chip

class gmmu_t {
 public:
  gmmu_t(class gpgpu_sim *gpu, const gpgpu_sim_config &config,
         class gpgpu_new_stats *new_stats);
  unsigned long long calculate_transfer_time(size_t data_size);
  unsigned long long calculate_decompres_time(size_t num_pages, float comp_ratio);
  void calculate_devicesync_time(size_t data_size);
  void cycle();
  void processWriteBackLatencyQueue();
  double get_total_compressedsize();
  void scheduleWriteBackTransfer();
  void processCompressionLatencyQueue();
  void scheduleCompressionTasks();
  void processReadLatencyQueue();
  void scheduleReadTransfer();
  void processDecompressionTransfers();
  void scheduleDecompressionTransfers();
  void register_tlbflush_callback(std::function<void(mem_addr_t)> cb_tlb);
  void tlb_flush(mem_addr_t page_num);
  void old_page_eviction_procedure();
  bool page_eviction_procedure();
  bool page_comp_eviction_procedure();
  bool page_compression_procedure(bool evict_enabled);
  bool is_block_evictable(mem_addr_t bb_addr, size_t size,bool print,bool large_page_check);
  bool is_block_precomp(mem_addr_t bb_addr, size_t size);
  bool is_block_evictable_print(mem_addr_t bb_addr, size_t size);
  bool is_block_compable(mem_addr_t bb_addr, size_t size);
  bool is_block_comp_evict(mem_addr_t bb_addr, size_t size,bool print=false);
  int check_invalid_pages(mem_addr_t bb_addr, size_t size, bool err);
  int check_compressed_pages(mem_addr_t bb_addr, size_t size, bool err);
  void update_lp_tree_prefetch_threshold(struct lp_tree_node *node, float prefetch_threshold);
  eviction_t *two_level_LRU_find_candidate(int type, bool comp_only);
  eviction_t *get_LRU_leaf_node(lp_tree_node *root,int type,bool comp_only);//type 0 is find valid list, 1 is find comp list
  LCP_CompressedPage find_LCP_compressed_page(mem_addr_t page_no);
  Compressor* get_compressor(){return compressor;}
  // add a new accessed page or refresh the position of the page in the LRU page
  // list being called on detecting tlb hit or when memory fetch comes back from
  // the upward (gmmu to cu) queue
  void refresh_valid_pages(mem_addr_t page_addr, bool check=false);
  void sort_valid_pages();
  void sort_compressed_pages();
  size_t get_decomp_stage_queue_size(){return decomp_stage_queue.size();};
  size_t get_comp_stage_queue_size(){return comp_stage_queue.size();};
  // check whether the page to be accessed is already in pci-e write stage queue
  // being called on tlb hit or on tlb miss but no page fault
  bool check_write_stage_queue(mem_addr_t page_num, bool refresh);
  bool check_write_latency_queue(mem_addr_t page_num, bool refresh);
  bool check_write_stage_queue_2MB(mem_addr_t page_num, bool refresh);
  void set_decomp_mf_wait_for_comp(mem_addr_t addr);
  //void check_comp_evict_stage_queue(mem_addr_t page_num, bool refresh);
  void print_debug_all(bool asser);
  bool print_all = false;
  void compressed_pages_decompressing(mem_addr_t pagenum, size_t leaf_size);
  void queue_compressed_pages_decomp(mem_addr_t pagenum);
  bool compressed_page_find(mem_addr_t page_num);
  //void compressed_pages_decomp_addr(mem_addr_t page_addr);
  void compressing_pages_done(mem_addr_t page_num, size_t leaf_size);
  void decompressing_pages_done(mem_addr_t page_num, size_t leaf_size);
//  void cancel_decompressing_pages(mem_addr_t page_num);
  bool cancel_compressing_pages(mem_addr_t page_num, size_t leaf_size, bool change_size=false);
  void valid_pages_compressing(mem_addr_t pagenum,size_t comp_size, size_t leaf_size);
  void valid_pages_erase(mem_addr_t pagenum, size_t leaf_size);
  void evicting_pages_erase(mem_addr_t pagenum, size_t leaf_size);
  void cancel_evicting_pages(mem_addr_t pagenum, size_t leaf_size);
  void valid_pages_clear();
  void compressed_pages_clear();
  void comp_evict_pages_clear();
  void LCP_compressed_pages_clear();
  //bool valid_pages_check(mem_addr_t page_num);
  void compress_pages_erase(mem_addr_t pagenum, size_t leaf_size);
  bool compress_pages_check(mem_addr_t pagenum);
  void comp_evict_erase(mem_addr_t pagenum, size_t leaf_size);
  void compress_pages_comp_evict(mem_addr_t page_num, size_t leaf_size);
  void comp_evict_erase_recover(mem_addr_t page_num, size_t leaf_size);
  void register_prefetch(mem_addr_t m_device_addr,
                         mem_addr_t m_device_allocation_ptr, size_t m_cnt,
                         struct CUstream_st *m_stream);
  void activate_prefetch(mem_addr_t m_device_addr, size_t m_cnt,
                         struct CUstream_st *m_stream);
  struct lp_tree_node *make_4MB_lp_tree(struct lp_tree_node *node_left,struct lp_tree_node *node_right);
  struct lp_tree_node *build_lp_tree(mem_addr_t addr, size_t size,uint8_t tree_type,uint8_t alloc_count);  //tree type 0 is default TBN and LC 2MB root 64K leaf 1 is LS tree 4MB root 256K leaf 2 is HCHI tree 512K root 64K leaf 3 is HCLI tree 512K root 16K leaf 4 is range 2MB root 2MB leaf
  void reset_lp_tree_type(struct lp_tree_node *node,uint8_t tree_type);
  void reset_large_page_info(struct lp_tree_node *node);
  void reset_lp_tree_node(struct lp_tree_node *node);
  struct lp_tree_node *get_lp_node(mem_addr_t addr);
  void evict_whole_tree(struct lp_tree_node *root);
  void evict_whole_tree_comp(struct lp_tree_node *root);
  void evict_whole_tree_comp_evict(struct lp_tree_node *root);
  mem_addr_t update_basic_block(struct lp_tree_node *root, mem_addr_t addr,
                                size_t size, bool prefetch);
  mem_addr_t update_basic_block_comp(struct lp_tree_node *root, mem_addr_t addr,
                                     size_t size, bool prefetch);
  mem_addr_t update_basic_block_comp_evict(struct lp_tree_node *root,
                                           mem_addr_t addr, size_t size,
                                           bool prefetch);
  mem_addr_t get_basic_block(struct lp_tree_node *root, mem_addr_t addr);
  // list/comp/decomp granularity: 64KB when MICRO_barrier_Range (range tree), else the tree leaf_size
  size_t micro_leaf(struct lp_tree_node *root);

  void fill_lp_tree(struct lp_tree_node *node,
                    std::set<mem_addr_t> &scheduled_basic_blocks);
  void fill_lp_tree_comp(struct lp_tree_node *node,
                         std::set<mem_addr_t> &scheduled_basic_blocks);
  void remove_lp_tree(struct lp_tree_node *node,
                      std::vector<mem_addr_t> &scheduled_basic_blocks);
  void remove_lp_tree_comp_evict(struct lp_tree_node *node,
                                 std::vector<mem_addr_t> &scheduled_basic_blocks);
  void remove_lp_tree_comp(struct lp_tree_node *node,
                                 std::vector<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_remove_lp_tree_comp(struct lp_tree_node *node,
                                 std::vector<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_fill_lp_tree(struct lp_tree_node *node,
                                 std::set<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_fill_lp_tree_comp(
      struct lp_tree_node *node, std::set<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_remove_lp_tree_comp_evict(
      struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks);
  void traverse_and_remove_lp_tree(
      struct lp_tree_node *node, std::vector<mem_addr_t> &scheduled_basic_blocks);

  bool pcie_transfers_completed();
  bool compression_completed();
  bool check_pages(mem_addr_t addr, size_t size, bool should_be_compressed,
                   bool check_validity = true);
  bool check_reserved_pages(mem_addr_t addr, size_t size);
  void initialize_large_page(mem_addr_t start_addr, size_t size,uint8_t tree_type,uint8_t alloc_count);
  void erase_page_from_list(std::list<eviction_t *> &page_list,
                            mem_addr_t page_addr, size_t leaf_size);
  void move_page_between_lists(std::list<eviction_t *> &from_list,
                               std::list<eviction_t *> &to_list,
                               mem_addr_t page_addr, size_t leaf_size);
  void move_page_between_lists_comp(std::list<eviction_t *> &from_list,
                               std::list<eviction_t *> &to_list,
                               mem_addr_t page_addr, size_t comp_size, size_t leaf_size);
  bool check_page_in_list(std::list<eviction_t *> &page_list,
                          mem_addr_t page_addr);
  bool check_page_in_list_comp(std::list<eviction_t *> &page_list,
                            mem_addr_t page_addr);
  void sort_eviction_list_by_access(std::list<eviction_t *> &eviction_list);
  void sort_eviction_list_by_cycle(std::list<eviction_t *> &eviction_list,
                                   bool reverse);
  unsigned long long get_ready_cycle(unsigned num_pages);
  unsigned long long get_ready_cycle_bytes(size_t bytes);
  unsigned long long get_ready_cycle_transfer(unsigned num_pages);
  unsigned long long get_ready_cycle_compress(unsigned num_pages);
  unsigned long long get_ready_cycle_pages_compress(unsigned num_pages,
                                                    float pages_band);
  unsigned long long get_compress_cycle(unsigned num_pages,
                                                    float pages_band);
  unsigned long long get_ready_cycle_with_decomp(unsigned num_pages,
                                                    float band, unsigned num_compressed_pages);
  double get_size_comp_occupancy(size_t num_pages, float comp_ratio);
  unsigned long long get_ready_cycle_dma(unsigned size);

  float get_pcie_utilization(unsigned num_pages);

  void do_hardware_prefetch(
      std::map<mem_addr_t, std::list<mem_fetch *>> &page_fault_this_turn);
  void do_hardware_precomp(
      std::map<mem_addr_t, std::list<mem_fetch *>> &page_comp_this_turn);
  void reserve_pages_insert(mem_addr_t addr, unsigned mem_access_uid);
  bool get_full_comp();
  void reserve_pages_remove(mem_addr_t addr, unsigned mem_access_uid);
  bool reserve_pages_check(mem_addr_t addr);
  int invalid_pages_check(bool err);
  //LCP_CompressedPage is_LCP_compress_page(mem_addr_t page_addr);
  std::map<mem_addr_t, std::list<unsigned>> reserve_pages;
  void update_hardware_prefetcher_oversubscribed();

  // update paging, pinning, and eviction decision based on memory access
  // pattern under oversubscription
  void update_memory_management_policy();
  void log_kernel_info(unsigned kernel_id, unsigned long long time,
                       bool finish);
  std::list<struct lp_tree_node *> large_page_info;
  void reset_large_page_info();

  mem_addr_t get_eviction_base_addr(mem_addr_t page_addr);
  size_t get_eviction_granularity(mem_addr_t page_addr, unsigned int leaf_size);

  int get_bb_access_counter(struct lp_tree_node *node, mem_addr_t addr);
  int get_bb_round_trip(struct lp_tree_node *node, mem_addr_t addr);
  void inc_bb_access_counter(mem_addr_t addr);
  void inc_bb_round_trip(struct lp_tree_node *root);
  void traverse_and_reset_access_counter(struct lp_tree_node *root);
  void reset_bb_access_counter();
  void traverse_and_reset_round_trip(struct lp_tree_node *root);
  void reset_bb_round_trip();
  void update_access_type(mem_addr_t addr, int type);
  bool is_sampled(mem_addr_t addr);
  size_t get_tree_comp_size(mem_addr_t addr); 
  void set_tree_comp_size(mem_addr_t addr, size_t comp_size);
  void set_sampled(mem_addr_t addr);
  void clear_sampled(mem_addr_t addr);
  int get_comp_policy();
  int get_comp_lat();
  bool should_cause_page_migration(mem_addr_t addr, bool is_write);
  bool get_tbn_threshold(){return m_config.tbn_threshold;};
  int need_throttle_SM=0;
  bool fault_occured=false;
  size_t maximum_used_compression_unit=0;
  std::list<comp_ratio_t> comp_ratio_info;
  bool process_print_on=false;
  bool WB_schedule_on=false;
  // Stalled SM 추적 관련 공개 함수들
  int Clock_LRU_index=0;
  size_t get_memory_stalled_sm_count() const { return memory_stalled_sms.get_stalled_count(); }
  bool is_sm_memory_stalled(unsigned sm_id) const { return memory_stalled_sms.is_stalled(sm_id); }
  unsigned get_sm_stall_intensity(unsigned sm_id) const { return memory_stalled_sms.get_stall_intensity(sm_id); }
  void clear_stalled_sm_tracker() { memory_stalled_sms.clear(); }
  
  //  std::map<mem_addr_t, LCP_compressed_chunk> LCP_compressed_pages_map; // for LCP compression with double mapping
  std::map<mem_addr_t, LCP_CompressedPage> LCP_compressed_pages_map;
  size_t get_comp_latency_queue_size(){return comp_latency_queue.size();};
  // 동시에 처리할 수 있는 최대 요청 수를 설정하는 변수 추가
  size_t max_parallel_comp_requests;
  //  size_t max_parallel_decomp_requests; integrated to max_parallel_comp_requests
 size_t current_stalled_SM=0;
  uint8_t alloc_count=0;
 // Stalled SM 통합 추적을 위한 구조체
 bool oversubscription_on=false;
 unsigned long long last_page_eviction_cycle=0;
 struct stalled_sm_tracker {
   std::map<unsigned, unsigned> sm_count;  // SM ID -> stalled 횟수
   size_t total_stalled_sms = 0;           // 현재 stalled된 SM 총 개수
   
   void add_stalled_sm(unsigned sm_id) {
     if (sm_count[sm_id]++ == 0) {
       total_stalled_sms++;
     }
   }
   
   void remove_stalled_sm(unsigned sm_id) {
     auto it = sm_count.find(sm_id);
     if (it != sm_count.end() && --it->second == 0) {
       sm_count.erase(it);
       total_stalled_sms--;
     }
   }
   
   bool is_stalled(unsigned sm_id) const {
     return sm_count.find(sm_id) != sm_count.end();
   }
   
   size_t get_stalled_count() const {
     return total_stalled_sms;
   }
   
   unsigned get_stall_intensity(unsigned sm_id) const {
     auto it = sm_count.find(sm_id);
     return it != sm_count.end() ? it->second : 0;
   }
   
   void clear() {
     sm_count.clear();
     total_stalled_sms = 0;
   }
 };

 // 통합 stalled SM 추적기
 stalled_sm_tracker memory_stalled_sms;

 private:
  // data structure to wrap memory fetch and page table walk delay
  struct page_table_walk_latency_t {
    mem_fetch *mf;
    unsigned long long ready_cycle;
  };

  // page table walk delay queue
  std::list<page_table_walk_latency_t> page_table_walk_queue;
  std::list<page_table_walk_latency_t> page_decomp_queue;
  std::list<page_table_walk_latency_t> full_comp_queue;
  enum class latency_type {
    PCIE_READ,
    PCIE_WRITE_BACK,
    INVALIDATE,
    PAGE_FAULT,
    COMP_FAULT,
    DMA,
    COMP,
    DECOMP,
    COMP_EVICT,
    COMP_INV_EVICT
  };

  // data structure to wrap a memory page and delay to transfer over PCI-E
  struct pcie_latency_t {
    mem_addr_t start_addr;
    unsigned long long size;
    std::list<mem_addr_t> page_list;
    unsigned long long ready_cycle;
    double comp_ratio;
    size_t comp_size;
    float comp_band;
    float decomp_band;
    unsigned long long start_cycle;
    mem_fetch *mf;
    latency_type type;
    bool mf_wait_for_comp=false;
    bool comp_fault_checked=false;
    // Separate control readiness from bandwidth-dependent data completion.
    unsigned long long crm_ready_cycle = 0;
    bool crm_handling_reserved = false;
    size_t leaf_size;
    bool first_block=false;
  };
  void reserve_crm_handling(const std::shared_ptr<pcie_latency_t> &request);
  bool is_in_queue_range(
      mem_addr_t addr, size_t size,
      const std::list<std::shared_ptr<pcie_latency_t>> &queue);
  // staging queue to hold the PCI-E requests waiting for scheduling
  std::list<std::shared_ptr<pcie_latency_t>> pcie_read_stage_queue;
  std::list<std::shared_ptr<pcie_latency_t>> pcie_write_stage_queue;
  std::list<std::shared_ptr<pcie_latency_t>> decomp_stage_queue;
  std::list<std::shared_ptr<pcie_latency_t>> comp_stage_queue;
  std::list<std::shared_ptr<pcie_latency_t>> comp_evict_stage_queue;
  bool process_stage_queue(
      std::list<std::shared_ptr<pcie_latency_t>> &stage_queue,
      mem_addr_t page_num, bool refresh, bool &check_cont);
  double decomp_demand_size;
  size_t comp_error_count;
  size_t page_eviction_count;
  // read queue for fetching the page from host side
  // the request may be global memory's read (load)/ write (store)
  std::shared_ptr<pcie_latency_t> pcie_read_latency_queue;
  std::list<std::shared_ptr<pcie_latency_t>> decomp_latency_queue;
  std::list<std::shared_ptr<pcie_latency_t>> comp_latency_queue;
  // write back queue for page eviction requests over PCI-E
  std::shared_ptr<pcie_latency_t> pcie_write_latency_queue;
  // loosely represent MSHRs to hold all memory fetches
  // corresponding to a PCI-E read requests, i.e., a common page number
  // to replay the memory fetch back upon completion
  std::map<mem_addr_t, std::list<mem_fetch *>> req_info;
  std::map<mem_addr_t, std::list<mem_fetch *>> comp_info;

  // need the gpu to do address traslation, validate page
  class gpgpu_sim *m_gpu;
  class memory_space *g_memory;
  int try_error;
  double num_read_stage_queue;//this is the number of pages
  double num_write_stage_queue;//this is the number of pages
  double num_comp_stage_queue;//this is the number of pages
  double num_decomp_stage_queue;//this is the number of pages
  double num_comp_evict_stage_queue;//this is the number of pages
  // config file
  const gpgpu_sim_config &m_config;
  const struct shader_core_config *m_shader_config;
  bool skip_compression_selection;
  // callback functions to invalidate the tlb in ldst unit
  std::list<std::function<void(mem_addr_t)>> callback_tlb_flush;
  Compressor* compressor;
  // list of valid pages (valid = 1, accessed = 1/0, dirty = 1/0) ordered as LRU
  // LRU
  std::list<eviction_t *> valid_pages;
  std::list<eviction_t *> compressing_pages;
  std::list<eviction_t *> decompressing_pages;
  std::list<eviction_t *> evicting_pages;
  std::list<eviction_t *> all_pages;
  std::list<mem_addr_t> valid_pages_dedup;
  std::list<eviction_t *> compressed_pages;
  int not_found_count;
  std::list<eviction_t *> comp_evict_pages;
  bool comp_evict;
  // page eviction policy
  enum class eviction_policy { LRU, TBN, SEQUENTIAL_LOCAL, RANDOM, LFU, LRU4K };
  enum class compression_policy_struct { FULL, PARTIAL, NONE };
  bool selective_compression;
  bool evict_form;
  // types of hardware prefetcher
  enum class hwardware_prefetcher { DISBALED, TBN, SEQUENTIAL_LOCAL, RANDOM };

  // types of hardware prefetcher under over-subscription
  enum class hwardware_prefetcher_oversub {
    DISBALED,
    TBN,
    SEQUENTIAL_LOCAL,
    RANDOM
  };

  // type of DMA
  enum class dma_type { DISABLED, ADAPTIVE, ALWAYS, OVERSUB };

  // type of memory access pattern per data structure
  enum class ds_pattern {
    UNDECIDED,
    RANDOM,
    LINEAR,
    MIXED,
    RANDOM_REUSE,
    LINEAR_REUSE,
    MIXED_REUSE
  };

  // list of scheduled basic blocks by their timestamps
  std::list<std::pair<unsigned long long, mem_addr_t>> block_access_list;

  // list of launch and finish cycle of kernels keyed by id
  std::map<unsigned, std::pair<unsigned long long, unsigned long long>>
      kernel_info;

  eviction_policy evict_policy;
  compression_policy_struct compression_policy;
  bool m_ignore_reserved_pages;
  float tbn_threshold_float;
  bool full_comp_policy;
  bool comp_evict_policy;
  float comp_ratio;
  hwardware_prefetcher prefetcher;
  hwardware_prefetcher_oversub oversub_prefetcher;

  dma_type dma_mode;

  struct prefetch_req {
    // starting address (rolled up and down for page alignment) for the prefetch
    mem_addr_t start_addr;

    // current address from the start up to which PCI-e has already processed
    mem_addr_t cur_addr;

    // starting address of the current variable allocation
    mem_addr_t allocation_addr;

    // total size (rolled up and down for page alignment) for the prefetch
    size_t size;

    // stream associated to the prefetch
    CUstream_st *m_stream;

    // memory fetches, which are created upon page fault and are depending on
    // current prefetch, aggreagted before the prefetch is actually scheduled
    std::map<mem_addr_t, std::list<mem_fetch *>> incoming_replayable_nacks;

    // memory fetches that are finished PCI-e transfer are aggregated to be
    // replayed together upon completion of the prefetch
    std::map<mem_addr_t, std::list<mem_fetch *>> outgoing_replayable_nacks;

    // list of pages (max upto 2MB) from the current prefetch request which are
    // being served by PCI-e
    std::list<mem_addr_t> pending_prefetch;

    // stream manager upon reaching to this entry of the queue sets it to active
    bool active;
  };

  std::list<prefetch_req> prefetch_req_buffer;

  std::list<event_stats *> fault_stats;
  std::list<event_stats *> writeback_stats;
  std::list<event_stats *> comp_fault_stats;
  std::list<event_stats *> compress_stats;

  size_t total_allocation_size;
  unsigned long long cur_cycle;
  // Shared control timeline for all ReclaimX representation transitions.
  unsigned long long last_crm_handling_cycle = 0;
  std::map<mem_addr_t, unsigned long long> crm_host_restore_ready;
  bool over_sub;

  class gpgpu_new_stats *m_new_stats;
  size_t failed_read_count;
  bool pcie_read_failed=false;
  unsigned int skip_eviction; // 페이지 제거(eviction) 및 압축 관련 제한 플래그
                              // EVICT_ENABLED(0x1): 페이지 제거 가능
                              // COMP_ENABLED(0x2): 압축 가능
                              // COMP_EVICT_ENABLED(0x4): 압축 후 제거 가능
  bool ignore_reserved_pages=false;
  // 비트 플래그 체크 함수들
  bool is_evict_enabled() const { return !(skip_eviction & 0x1); }
  bool is_comp_enabled() const { return !(skip_eviction & 0x2); }
  bool is_comp_evict_enabled() const { return !(skip_eviction & 0x4); }

  // 비트 플래그 설정 함수들
  void enable_evict() { skip_eviction &= ~0x1; }
  void disable_evict() { if(m_config.compression_policy) skip_eviction |= 0x1; }
  void enable_comp() { skip_eviction &= ~0x2; }
  void disable_comp() { skip_eviction |= 0x2; }
  void enable_comp_evict() { skip_eviction &= ~0x4; }
  void disable_comp_evict() { skip_eviction |= 0x4; }
  void clear_skip_eviction() { skip_eviction = 0; }
  void clear_valid_eviction_compression() { enable_comp(); enable_evict(); }
};

struct lp_tree_node {
  mem_addr_t addr;
  size_t size;
  size_t valid_size;
  struct lp_tree_node *left;
  struct lp_tree_node *right;
  struct lp_tree_node *parent;
  uint8_t tree_type;
  uint32_t access_counter;
  size_t compress_size;//for page compression and prefetch...
  bool large_pages_reserved;
  bool sampled;
  bool inefficient_compression;//if set skip compression selection
  uint8_t RW;
  float comp_ratio;
  size_t comp_size;//for memory allocation
  unsigned int leaf_size; //for page prefetch 
  unsigned int root_size; //for page prefetch 
  uint8_t alloc_count; //index of allocated tree type
  bool leaf_will_be_inefficient=false;
  unsigned int fault_from[32];//circular buffer for fault from
  unsigned int fault_index=0; //index of fault from
  float prefetch_threshold=100.0;
  bool accessed=false;
};
struct occupancy_stats {
  occupancy_stats()
      : aggregate_warp_slot_filled(0), aggregate_theoretical_warp_slots(0) {}
  occupancy_stats(unsigned long long wsf, unsigned long long tws)
      : aggregate_warp_slot_filled(wsf),
        aggregate_theoretical_warp_slots(tws) {}

  unsigned long long aggregate_warp_slot_filled;
  unsigned long long aggregate_theoretical_warp_slots;

  float get_occ_fraction() const {
    return float(aggregate_warp_slot_filled) /
           float(aggregate_theoretical_warp_slots);
  }

  occupancy_stats &operator+=(const occupancy_stats &rhs) {
    aggregate_warp_slot_filled += rhs.aggregate_warp_slot_filled;
    aggregate_theoretical_warp_slots += rhs.aggregate_theoretical_warp_slots;
    return *this;
  }

  occupancy_stats operator+(const occupancy_stats &rhs) const {
    return occupancy_stats(
        aggregate_warp_slot_filled + rhs.aggregate_warp_slot_filled,
        aggregate_theoretical_warp_slots +
            rhs.aggregate_theoretical_warp_slots);
  }
};

class gpgpu_context;
class ptx_instruction;

class watchpoint_event {
 public:
  watchpoint_event() {
    m_thread = NULL;
    m_inst = NULL;
  }
  watchpoint_event(const ptx_thread_info *thd, const ptx_instruction *pI) {
    m_thread = thd;
    m_inst = pI;
  }
  const ptx_thread_info *thread() const { return m_thread; }
  const ptx_instruction *inst() const { return m_inst; }

 private:
  const ptx_thread_info *m_thread;
  const ptx_instruction *m_inst;
};

class gpgpu_sim : public gpgpu_t {
 public:
  gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx);

  void set_prop(struct cudaDeviceProp *prop);

  void launch(kernel_info_t *kinfo);
  bool can_start_kernel();
  unsigned finished_kernel();
  void set_kernel_done(kernel_info_t *kernel);
  void stop_all_running_kernels();

  void init();
  void cycle();
  bool active();
  bool cycle_insn_cta_max_hit() {
    return (m_config.gpu_max_cycle_opt && (gpu_tot_sim_cycle + gpu_sim_cycle) >=
                                              m_config.gpu_max_cycle_opt) ||
           (m_config.gpu_max_insn_opt &&
            (gpu_tot_sim_insn + gpu_sim_insn) >= m_config.gpu_max_insn_opt) ||
           (m_config.gpu_max_cta_opt &&
            (gpu_tot_issued_cta >= m_config.gpu_max_cta_opt)) ||
           (m_config.gpu_max_completed_cta_opt &&
            (gpu_completed_cta >= m_config.gpu_max_completed_cta_opt));
  }
  void print_stats();
  void update_stats();
  void deadlock_check();
  bool is_progress_print_set();
  void inc_completed_cta() { gpu_completed_cta++; }
  void get_pdom_stack_top_info(unsigned sid, unsigned tid, unsigned *pc,
                               unsigned *rpc);

  int shared_mem_size() const;
  int shared_mem_per_block() const;
  int compute_capability_major() const;
  int compute_capability_minor() const;
  int num_registers_per_core() const;
  int num_registers_per_block() const;
  int wrp_size() const;
  int shader_clock() const;
  int max_cta_per_core() const;
  int get_max_cta(const kernel_info_t &k) const;
  const struct cudaDeviceProp *get_prop() const;
  enum divergence_support_t simd_model() const;

  unsigned threads_per_core() const;
  bool get_more_cta_left() const;
  bool kernel_more_cta_left(kernel_info_t *kernel) const;
  bool hit_max_cta_count() const;
  kernel_info_t *select_kernel();
  PowerscalingCoefficients *get_scaling_coeffs();
  void decrement_kernel_latency();

  const gpgpu_sim_config &get_config() const { return m_config; }
  void gpu_print_stat();
  void dump_pipeline(int mask, int s, int m) const;

  void perf_memcpy_to_gpu(size_t dst_start_addr, size_t count);

  // The next three functions added to be used by the functional simulation
  // function

  //! Get shader core configuration
  /*!
   * Returning the configuration of the shader core, used by the functional
   * simulation only so far
   */
  const shader_core_config *getShaderCoreConfig();

  //! Get shader core Memory Configuration
  /*!
   * Returning the memory configuration of the shader core, used by the
   * functional simulation only so far
   */
  const memory_config *getMemoryConfig();

  //! Get shader core SIMT cluster
  /*!
   * Returning the cluster of of the shader core, used by the functional
   * simulation so far
   */
  simt_core_cluster *getSIMTCluster(int index);

  gmmu_t *getGmmu();
  void flush_l2_page(unsigned page_num);
  void hit_watchpoint(unsigned watchpoint_num, ptx_thread_info *thd,
                      const ptx_instruction *pI);

  // backward pointer
  class gpgpu_context *gpgpu_ctx;
  unsigned num_throttled_SM;
  
  // CURE throttle용 이전 워프 수 저장 변수들
  std::vector<unsigned long long> prev_active_warps_per_cluster;
  std::vector<unsigned long long> prev_total_warps_per_cluster;
  
  // throttle 조절의 마지막 실행 시점을 저장하는 변수들
  unsigned long long last_throttle_adjustment_cycle;
  unsigned long long print_throttle_count;
  class gmmu_t *m_gmmu;

 private:
  // clocks
  void reinit_clock_domains(void);
  int next_clock_domain(void);
  void issue_block2core();
  void throttle_SM();
  void print_dram_stats(FILE *fout) const;
  void shader_print_runtime_stat(FILE *fout);
  void shader_print_l1_miss_stat(FILE *fout) const;
  void shader_print_cache_stats(FILE *fout) const;
  void shader_print_scheduler_stat(FILE *fout, bool print_dynamic_info) const;
  void visualizer_printstat();
  void print_shader_cycle_distro(FILE *fout) const;

  void gpgpu_debug();

 protected:
  ///// data /////
  class simt_core_cluster **m_cluster;
  class memory_partition_unit **m_memory_partition_unit;
  class memory_sub_partition **m_memory_sub_partition;

  std::vector<kernel_info_t *> m_running_kernels;
  unsigned m_last_issued_kernel;

  std::list<unsigned> m_finished_kernel;
  // m_total_cta_launched == per-kernel count. gpu_tot_issued_cta == global
  // count.
  unsigned long long m_total_cta_launched;
  unsigned long long gpu_tot_issued_cta;
  unsigned gpu_completed_cta;
  unsigned m_last_cluster_issue;
  unsigned m_last_fetched_cluster;
  float *average_pipeline_duty_cycle;
  float *active_sms;
  // time of next rising edge
  double core_time;
  double icnt_time;
  double dram_time;
  double l2_time;
  double gmmu_time;

  // debug
  bool gpu_deadlock;

  //// configuration parameters ////
  const gpgpu_sim_config &m_config;

  const struct cudaDeviceProp *m_cuda_properties;
  const shader_core_config *m_shader_config;
  const memory_config *m_memory_config;

  // stats
  class shader_core_stats *m_shader_stats;
  class memory_stats_t *m_memory_stats;
  class power_stat_t *m_power_stats;
  class gpgpu_sim_wrapper *m_gpgpusim_wrapper;
  unsigned long long last_gpu_sim_insn;

  unsigned long long last_liveness_message_time;

  std::map<std::string, FuncCache> m_special_cache_config;

  std::vector<std::string>
      m_executed_kernel_names;  //< names of kernel for stat printout
  std::vector<unsigned>
      m_executed_kernel_uids;  //< uids of kernel launches for stat printout
  std::map<unsigned, watchpoint_event> g_watchpoint_hits;

  std::string executed_kernel_info_string();  //< format the kernel information
                                              // into a string for stat printout
  std::string executed_kernel_name();
  void clear_executed_kernel_info();  //< clear the kernel information after
                                      // stat printout
  virtual void createSIMTCluster() = 0;

 public:
  class gpgpu_new_stats *m_new_stats;

 public:
  unsigned long long gpu_sim_insn;
  unsigned long long gpu_tot_sim_insn;
  unsigned long long gpu_sim_insn_last_update;
  unsigned gpu_sim_insn_last_update_sid;
  occupancy_stats gpu_occupancy;
  occupancy_stats gpu_tot_occupancy;

  // performance counter for stalls due to congestion.
  unsigned int gpu_stall_dramfull;
  unsigned int gpu_stall_icnt2sh;
  unsigned long long partiton_reqs_in_parallel;
  unsigned long long partiton_reqs_in_parallel_total;
  unsigned long long partiton_reqs_in_parallel_util;
  unsigned long long partiton_reqs_in_parallel_util_total;
  unsigned long long gpu_sim_cycle_parition_util;
  unsigned long long gpu_tot_sim_cycle_parition_util;
  unsigned long long partiton_replys_in_parallel;
  unsigned long long partiton_replys_in_parallel_total;

  FuncCache get_cache_config(std::string kernel_name);
  void set_cache_config(std::string kernel_name, FuncCache cacheConfig);
  bool has_special_cache_config(std::string kernel_name);
  void change_cache_config(FuncCache cache_config);
  void set_cache_config(std::string kernel_name);

  // Jin: functional simulation for CDP
 private:
  // set by stream operation every time a functoinal simulation is done
  bool m_functional_sim;
  kernel_info_t *m_functional_sim_kernel;

 public:
  bool is_functional_sim() { return m_functional_sim; }
  kernel_info_t *get_functional_kernel() { return m_functional_sim_kernel; }
  void functional_launch(kernel_info_t *k) {
    m_functional_sim = true;
    m_functional_sim_kernel = k;
  }
  void finish_functional_sim(kernel_info_t *k) {
    assert(m_functional_sim);
    assert(m_functional_sim_kernel == k);
    m_functional_sim = false;
    m_functional_sim_kernel = NULL;
  }
};

class exec_gpgpu_sim : public gpgpu_sim {
 public:
  exec_gpgpu_sim(const gpgpu_sim_config &config, gpgpu_context *ctx)
      : gpgpu_sim(config, ctx) {
    createSIMTCluster();
  }

  virtual void createSIMTCluster();
};

#endif
