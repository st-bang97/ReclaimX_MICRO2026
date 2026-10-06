// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
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

#ifndef memory_h_INCLUDED
#define memory_h_INCLUDED

#include "../abstract_hardware_model.h"

#include "../tr1_hash_map.h"
#define mem_map tr1_hash_map
#if tr1_hash_map_ismap == 1
#define MEM_MAP_RESIZE(hash_size)
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
#endif

#include <assert.h>
#include <list>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <string>
#include "../gpgpu-sim/Compressor.h"
typedef address_type mem_addr_t;

typedef struct _eviction_t {
  mem_addr_t addr;
  size_t size;
  unsigned long long cycle;
  uint32_t access_counter;
  uint8_t RW;
  size_t comp_size;
} eviction_t;
typedef struct _comp_ratio_t {
  mem_addr_t addr;
  size_t size;
  double comp_ratio;
  double comp_bandwidth;
  double decomp_bandwidth;
} comp_ratio_t;
#define MEM_BLOCK_SIZE (4 * 1024)

template <unsigned BSIZE>
class mem_storage {
 public:
  mem_storage(const mem_storage &another) {
    m_data = (unsigned char *)calloc(1, BSIZE);
    memcpy(m_data, another.m_data, BSIZE);
    // initialize page as unmanaged
    managed = false;

    // initialize page flags to default value
    valid = false;
    dirty = false;
    access = false;
    compressed = false;    // bst edit
    evicted_compressed = false;
    sampled = false;
    comp_ratio = 0;        // bst edit
    comp_bandwidth = 1;    // bst edit
    decomp_bandwidth = 1;  // bst edit
    counter = 0;
    bpc_compressor = new BPCompressor("BPCompressor");
    bdi_compressor = new BDICompressor("BDICompressor");
  }
  
  mem_storage() { 
    m_data = (unsigned char *)calloc(1, BSIZE); 

    // initialize page as unmanaged
    managed = false;

    // initialize page flags to default value
    valid = false;
    dirty = false;
    access = false;
    evicted_compressed = false;
    sampled = false;
    compressed = false;  // bst edit
    comp_ratio = 0;      // bst edit
    counter = 0;
    bpc_compressor = new BPCompressor("BPCompressor");
    bdi_compressor = new BDICompressor("BDICompressor");
  }
  ~mem_storage() { free(m_data); delete bpc_compressor; delete bdi_compressor; }

  void write(unsigned offset, size_t length, const unsigned char *data) {
    assert(offset + length <= BSIZE);
    memcpy(m_data + offset, data, length);
  }

  void read(unsigned offset, size_t length, unsigned char *data) const {
    //printf("bst debug read\n");
    //printf("offset: %d, length: %d, BSIZE: %d\n", offset, length, BSIZE);
    assert(offset + length <= BSIZE);
    memcpy(data, m_data + offset, length);//data is the pointer to copy address, m_data is the pointer to the memory address
  }
  //comp_type: 1 for BPCompressor, 2 for BDICompressor
  //prev_allocated_size: the previous allocated size of the page
  //page_no: the page number of the memory address
  //return : LCP compressed page
  LCP_CompressedPage *compress_page_with_LCP(int comp_type, mem_addr_t page_no, unsigned prev_allocated_size) {
    unsigned char temp_data[BSIZE];//data is the pointer to the memory address
    memcpy(temp_data, m_data+page_no, BSIZE);//read the data from the memory address
    LCP_CompressedPage *compressed_page;
    if (comp_type == 1) {
      compressed_page = compressPageLCP(bpc_compressor, temp_data, prev_allocated_size, page_no);
    } else if (comp_type == 2) {
      compressed_page = compressPageLCP(bdi_compressor, temp_data, prev_allocated_size, page_no);
    }
    free(temp_data);//we only need compression size and LCP information, so we can free the temp_data(real data)
    return compressed_page;
  }

  void print(const char *format, FILE *fout) const {
    unsigned int *i_data = (unsigned int *)m_data;
    for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) {
      if (d % 1 == 0) {
        fprintf(fout, "\n");
      }
      fprintf(fout, format, i_data[d]);
      fprintf(fout, " ");
    }
    fprintf(fout, "\n");
    fflush(fout);
  }
  
  // set the flag of managed into true in order to distinguish it from the
  // unmanaged allocation
  void set_managed() { managed = true; }
  bool is_managed() { return managed; }
  bool is_compressed() { return compressed; }
  void set_comp_bandwidth(float band_val) {
    comp_bandwidth = band_val;
  }  // consider curv_a
  void set_comp_ratio(float ratio) { comp_ratio = ratio; }
  float get_comp_ratio() { return comp_ratio; }
  void set_decomp_bandwidth(float band_val) {
    decomp_bandwidth = band_val;
  }  // consider curv_a
  float get_decomp_bandwidth() { return decomp_bandwidth;}//decomp_bandwidth; }
  float get_comp_bandwidth() { return comp_bandwidth;}//comp_bandwidth; }
  //bst edit for compressed
  // bst edit for compressed
  void set_compressed() { compressed = true; }
  void set_decompressed() {
    compressed = false;
  }
  // methods to query and modify page table flags
  bool is_valid() { return valid; }
  void validate_page() { valid = true; }
  void set_evicted_compressed() { evicted_compressed = true; }
  void clear_evicted_compressed() { evicted_compressed = false; }
  bool is_evicted_compressed() { return evicted_compressed; }
  bool is_sampled() { return sampled; }
  void set_sampled() { sampled = true; }
  void clear_sampled() { sampled = false; }
  void invalidate_page() { valid = false; }
  void set_dirty() { dirty = true; }
  void clear_dirty() { dirty = false; }
  bool is_dirty() { return dirty; }

  void set_access() { access = true; }
  void clear_access() { access = false; }
  bool is_access() { return access; }

 private:
  unsigned m_nbytes;
  unsigned char *m_data;
  BPCompressor* bpc_compressor;
  BDICompressor* bdi_compressor;
  // flag to differentiate whether a page is a managed allocation or traditional
  // unmanged by deafult it is false to denote cudaMalloc and cudaMallocArray on
  // managed allocation set this to true check this at the generation of
  // mem_fetch to determine which path to take managed may take the longer
  // latency path
  bool managed;
  bool compressed;         // bst edit
  float comp_ratio;        // bst edit
  float comp_bandwidth;    // bst edit
  float decomp_bandwidth;  // bst edit
  // flags for page table
  bool sampled;
  bool valid;
  bool evicted_compressed;
  bool dirty;

  bool access;

  unsigned counter;
};

class ptx_thread_info;
class ptx_instruction;

class memory_space {
 public:
  virtual ~memory_space() {}
  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI) = 0;
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data) = 0;
  virtual void read(mem_addr_t addr, size_t length, void *data) const = 0;
  virtual void print(const char *format, FILE *fout) const = 0;
  virtual void set_watch(addr_t addr, unsigned watchpoint) = 0;
    // method to find out whether or not to follow the managed time simulation
  virtual bool is_page_managed(mem_addr_t addr, size_t length) = 0;
  // method to set the pages as managed allocation
 virtual void set_pages_managed(mem_addr_t addr, size_t length) = 0;
  virtual void set_pages_compressed(mem_addr_t addr, size_t length) = 0;
  virtual void set_pages_decompressed(mem_addr_t addr, size_t length) = 0;
  virtual float get_pages_comp_ratio(mem_addr_t addr) = 0;
  virtual void set_page_comp_ratio(mem_addr_t addr, float comp_ratio) = 0;
  virtual void set_pages_comp_band(mem_addr_t addr, size_t length) = 0;
  virtual void set_pages_decomp_band(mem_addr_t addr, size_t length) = 0;
  virtual float get_pages_comp_band(mem_addr_t addr) = 0;
  virtual float get_pages_decomp_band(mem_addr_t addr) = 0;
  //virtual void set_page_compress(mem_addr_t pg_index) = 0;
  //virtual void set_page_uncompress(mem_addr_t pg_index) = 0;
  virtual bool is_page_compressed(mem_addr_t addr, size_t length) = 0;
  // method to allocate page(s) from free pages and change the count of free
  // pages
  virtual bool alloc_page_by_byte(size_t size) = 0;
  virtual void alloc_pages(double num) = 0;
  virtual void free_pages(double num) = 0;
  //virtual size_t get_free_pages() = 0;
  virtual bool comp_evict_free_pages(size_t num, float comp_ratio) = 0;
  virtual bool comp_free_pages(size_t num, float comp_ratio) = 0;
  virtual bool change_size_comp_evict_with_comp_size(size_t tree_size, size_t comp_size,bool increase) = 0;
  virtual bool change_size_comp_with_comp_size(size_t comp_size, bool increase) = 0;
  virtual bool change_size(size_t num, bool increase, float comp_ratio) = 0;
  virtual double get_free_pages() = 0;
  virtual double get_total_pages() = 0;
  virtual double get_init_pages() = 0;
  virtual double get_comp_free_pages()= 0;
  virtual void set_page_dirty(mem_addr_t pg_index) = 0;
  virtual bool is_page_dirty(mem_addr_t pg_index) = 0;
  virtual void clear_page_dirty(mem_addr_t pg_index) = 0;

  virtual void set_page_access(mem_addr_t pg_index) = 0;
  virtual bool is_page_access(mem_addr_t pg_index) = 0;
  virtual void clear_page_access(mem_addr_t pg_index) = 0;

  // methods to query page table
  virtual void validate_page(mem_addr_t pg_index) = 0;
  virtual void invalidate_page(mem_addr_t pg_index) = 0;
  virtual void set_evicted_compressed(mem_addr_t pg_index) = 0;
  virtual void clear_evicted_compressed(mem_addr_t pg_index) = 0;
  virtual bool is_evicted_compressed(mem_addr_t pg_index) = 0;
  virtual bool is_sampled(mem_addr_t pg_index) = 0;
  virtual void set_sampled(mem_addr_t pg_index) = 0;
  virtual void clear_sampled(mem_addr_t pg_index) = 0;
  virtual void set_iter_compressed(mem_addr_t pg_index) = 0;
  virtual void set_iter_decompressed(mem_addr_t pg_index) = 0;
  virtual bool is_iter_compressed(mem_addr_t pg_index) = 0;

  virtual std::list<mem_addr_t> get_faulty_pages(mem_addr_t addr,
                                                 size_t length) = 0;
  virtual mem_addr_t get_page_num(mem_addr_t addr) = 0;
  virtual mem_addr_t get_page_addr(mem_addr_t pg_index) = 0;

  virtual std::list<mem_addr_t> get_compressed_pages(mem_addr_t addr,
                                                 size_t length) = 0;
  virtual size_t get_data_size(mem_addr_t addr) = 0;
  virtual size_t get_page_size() = 0;
  virtual mem_addr_t get_mem_addr(mem_addr_t pg_index) = 0;
  virtual size_t get_cache_line_index(mem_addr_t addr) = 0;
  virtual bool is_valid(mem_addr_t pg_index) = 0;
  virtual bool should_evict_page(double read_stage_queue_size,
                                 double write_stage_queue_size,
                                 double decomp_stage_queue_size,
                                 double comp_stage_queue_size,
                                 float eviction_buffer_percentage) = 0;
  virtual bool should_comp_page(double read_stage_queue_size,
                                double write_stage_queue_size,
                                double decomp_stage_queue_size,
                                double comp_stage_queue_size,
                                float  comp_buffer_percentage) = 0;
  virtual int should_page_action(double read_stage_queue_size,
                                double write_stage_queue_size,
                                double decomp_stage_queue_size,
                                double comp_stage_queue_size,
                                double comp_evict_stage_queue_size,
                                float eviction_buffer_percentage,
                                float comp_buffer_percentage,
                                float comp_evict_start_percentage) = 0;
  virtual float get_projected_occupancy(double read_stage_queue_size,
                                        double write_stage_queue_size,
                                        float eviction_buffer_percentage) = 0;
  virtual void read_single_page(mem_addr_t page_num, void *data) = 0;
  virtual void reset() = 0;
};

template <unsigned BSIZE>
class memory_space_impl : public memory_space {
 public:
  memory_space_impl(std::string name, unsigned hash_size,
                    unsigned long long gddr_size = 0,
                    std::vector<float> comp_ratio = {},
                    std::vector<float> comp_band = {},
                    std::vector<float> decomp_band = {},
                    float comp_band_s = 4.5,
                    float decomp_band_s = 4.5);

  virtual void write(mem_addr_t addr, size_t length, const void *data,
                     ptx_thread_info *thd, const ptx_instruction *pI);
  virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                          const void *data);
  virtual void read(mem_addr_t addr, size_t length, void *data) const;
  virtual void print(const char *format, FILE *fout) const;

  virtual void set_watch(addr_t addr, unsigned watchpoint);

  // method to find out whether or not to follow the managed time simulation
  virtual bool is_page_managed(mem_addr_t addr, size_t length);
  // method to set the pages as managed allocation
  virtual void set_pages_managed(mem_addr_t addr, size_t length);
  virtual void set_pages_compressed(mem_addr_t addr, size_t length);
  virtual void set_pages_decompressed(mem_addr_t addr, size_t length);
  virtual bool is_page_compressed(mem_addr_t addr, size_t length);
  virtual float get_pages_comp_ratio(mem_addr_t addr);
  virtual void set_page_comp_ratio(mem_addr_t addr, float comp_ratio);
  virtual void set_pages_comp_band(mem_addr_t addr, size_t length);
  virtual void set_pages_decomp_band(mem_addr_t addr, size_t length);
  virtual float get_pages_comp_band(mem_addr_t addr);
  virtual float get_pages_decomp_band(mem_addr_t addr);
  // methods to query page table
  virtual void validate_page(mem_addr_t pg_index);
  virtual void invalidate_page(mem_addr_t pg_index);
  virtual void set_evicted_compressed(mem_addr_t pg_index);
  virtual void clear_evicted_compressed(mem_addr_t pg_index);
  virtual bool is_evicted_compressed(mem_addr_t pg_index);
  virtual bool is_sampled(mem_addr_t pg_index);
  virtual void set_sampled(mem_addr_t pg_index);
  virtual void clear_sampled(mem_addr_t pg_index);
  virtual void set_iter_compressed(mem_addr_t pg_index);
  virtual void set_iter_decompressed(mem_addr_t pg_index);
  virtual bool is_iter_compressed(mem_addr_t pg_index);
  virtual std::list<mem_addr_t> get_faulty_pages(mem_addr_t addr,
                                                 size_t length);
  virtual mem_addr_t get_page_num(mem_addr_t addr);
  virtual mem_addr_t get_page_addr(mem_addr_t pg_index);

  // methods to implement gddr size constraint
  virtual std::list<mem_addr_t> get_compressed_pages(mem_addr_t addr,
                                                 size_t length);  
  virtual bool alloc_page_by_byte(size_t size);
  virtual void alloc_pages(double num);
  virtual void free_pages(double num);
  virtual bool comp_evict_free_pages(size_t num, float comp_ratio);
  virtual bool comp_free_pages(size_t num, float comp_ratio);
  virtual bool change_size_comp_evict_with_comp_size(size_t tree_size, size_t comp_size, bool increase);
  virtual bool change_size_comp_with_comp_size(size_t comp_size, bool increase);
  virtual bool change_size(size_t num, bool increase, float comp_ratio);
  virtual double get_free_pages();
  virtual double get_init_pages();
  virtual double get_total_pages();
  virtual double get_comp_free_pages();

  virtual void set_page_dirty(mem_addr_t pg_index);
  virtual bool is_page_dirty(mem_addr_t pg_index);
  virtual void clear_page_dirty(mem_addr_t pg_index);

  virtual void set_page_access(mem_addr_t pg_index);
  virtual bool is_page_access(mem_addr_t pg_index);
  virtual void clear_page_access(mem_addr_t pg_index);

  virtual size_t get_data_size(mem_addr_t addr);
  virtual size_t get_page_size();
  virtual mem_addr_t get_mem_addr(mem_addr_t pg_index);
  virtual size_t get_cache_line_index(mem_addr_t addr);
  virtual bool is_valid(mem_addr_t pg_index);
  virtual bool should_evict_page(double read_stage_queue_size,
                                 double write_stage_queue_size,
                                 double decomp_stage_queue_size,
                                 double comp_stage_queue_size,
                                 float eviction_buffer_percentage);
  virtual bool should_comp_page(double read_stage_queue_size,
                                double write_stage_queue_size,
                                double decomp_stage_queue_size,
                                double comp_stage_queue_size,
                                float comp_buffer_percentage);
  virtual int should_page_action(double read_stage_queue_size,
                                double write_stage_queue_size,
                                double decomp_stage_queue_size,
                                double comp_stage_queue_size,
                                double comp_evict_stage_queue_size,
                                float eviction_buffer_percentage,
                                float comp_buffer_percentage,
                                float comp_evict_start_percentage);
  virtual float get_projected_occupancy(double read_stage_queue_size,
                                        double write_stage_queue_size,
                                        float eviction_buffer_percentage);
  virtual void read_single_page(mem_addr_t page_num, void *data);
  virtual void reset();
  int band_count;

 private:
  void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                         void *data) const;
  std::string m_name;
  unsigned m_log2_block_size;//shift right to get the page number
  typedef mem_map<mem_addr_t, mem_storage<BSIZE> > map_t;
  map_t m_data;
  
  // variable to store total number of 8KB pages in global memory
  // calculated based on the GDDR5 size specified in config
  // it is used to enforce size restriction on both managed and unmanaged malloc
  // it should be decremented on every allocation either managed or unmanaged
  // i.e., gpu_malloc, gpu_mallocmanaged, gpu_mallocarray
  double num_free_pages;//edited the number of free bytes
  double comp_added_pages;//edited the number of compressed bytes
  std::vector<float> comp_ratio_float;
  std::vector<float> comp_band_float;
  std::vector<float> decomp_band_float;
  float comp_band_scalar;
  float decomp_band_scalar;
  bool band_on;
  // the size of gddr in number of pages
  double num_gddr_pages;//edited the number of pages of gddr
    double init_num_gddr;//edited the number of bytes of gddr

  std::map<unsigned, mem_addr_t> m_watchpoints;
};

#endif
