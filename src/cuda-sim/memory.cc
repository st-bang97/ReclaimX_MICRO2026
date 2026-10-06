// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung,
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

#include "memory.h"
#include <stdlib.h>
#include "../../libcuda/gpgpu_context.h"
#include "math.h"
#include "../debug.h"
std::vector<float> valid_compression_ratios = {
    1.0f, 1.03f, 1.07f, 1.10f, 1.14f, 1.19f, 1.23f, 1.28f, 1.33f, 1.39f,
    1.46f, 1.52f, 1.6f, 1.68f, 1.78f, 1.88f, 2.0f, 2.13f, 2.29f, 2.46f,
    2.67f, 2.91f, 3.2f, 3.56f, 4.0f, 4.57f, 5.33f, 6.4f, 8.0f, 10.67f, 16.0f, 32.0f
};
// 가장 가까운 압축률 찾는 함수
float find_nearest_compression_ratio(float input_ratio) {
    auto it = std::lower_bound(valid_compression_ratios.begin(), valid_compression_ratios.end(), input_ratio);
    
    // lower_bound는 input_ratio보다 크거나 같은 첫 번째 요소를 반환
    if (it == valid_compression_ratios.begin()) {
        return *it;  // 리스트의 최소값보다 작으면 최소값 반환
    }
    
    if (it == valid_compression_ratios.end()) {
        return valid_compression_ratios.back();  // 리스트의 최대값보다 크면 최대값 반환
    }
    
    // 이전 값이 input_ratio보다 작거나 같은 가장 가까운 값
    return *(it - 1);
}
const int64_t MULTIPLIER = 100000000;
template <unsigned BSIZE>
memory_space_impl<BSIZE>::memory_space_impl(std::string name,
                                            unsigned hash_size,
                                            unsigned long long gddr_size,
                                            std::vector<float> comp_ratio,
                                            std::vector<float> comp_band,
                                            std::vector<float> decomp_band,
                                            float comp_band_s,
                                            float decomp_band_s)
    : num_gddr_pages(gddr_size), comp_band_scalar(comp_band_s), decomp_band_scalar(decomp_band_s) {
  m_name = name;
  MEM_MAP_RESIZE(hash_size);

  m_log2_block_size = -1;
  for (unsigned n = 0, mask = 1; mask != 0; mask <<= 1, n++) {
    if (BSIZE & mask) {
      assert(m_log2_block_size == (unsigned)-1);
      m_log2_block_size = n;
    }
  }
  assert(m_log2_block_size != (unsigned)-1);

  // initialize the number of free pages based on size of GDDR5 and page size
  num_free_pages = num_gddr_pages;
  band_count = 0;
  init_num_gddr = num_gddr_pages;
  comp_added_pages = 0;
  comp_ratio_float = comp_ratio;
  comp_band_float = comp_band;
  decomp_band_float = decomp_band;
  band_on = !comp_band_float.empty() && !decomp_band_float.empty() &&
            !comp_ratio_float.empty() ;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write_only(mem_addr_t offset, mem_addr_t index,
                                          size_t length, const void *data) {
  m_data[index].write(offset, length, (const unsigned char *)data);
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::write(mem_addr_t addr, size_t length,
                                     const void *data,
                                     class ptx_thread_info *thd,
                                     const ptx_instruction *pI) {
  mem_addr_t index = addr >> m_log2_block_size;

  if ((addr + length) <= (index + 1) * BSIZE) {
    // fast route for intra-block access
    unsigned offset = addr & (BSIZE - 1);
    unsigned nbytes = length;
    m_data[index].write(offset, nbytes, (const unsigned char *)data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned src_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      m_data[page].write(offset, tx_bytes,
                         &((const unsigned char *)data)[src_offset]);

      // advance pointers
      src_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
  if (!m_watchpoints.empty()) {
    std::map<unsigned, mem_addr_t>::iterator i;
    for (i = m_watchpoints.begin(); i != m_watchpoints.end(); i++) {
      mem_addr_t wa = i->second;
      if (((addr <= wa) && ((addr + length) > wa)) ||
          ((addr > wa) && (addr < (wa + 4))))
        thd->get_gpu()->gpgpu_ctx->the_gpgpusim->g_the_gpu->hit_watchpoint(
            i->first, thd, pI);
    }
  }
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::reset() {
  num_free_pages = num_gddr_pages;
  comp_added_pages = 0;
  band_count = 0;
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_managed(mem_addr_t addr, size_t length) {
  mem_addr_t page_index = get_page_num(addr + length - 1);
  return m_data[page_index].is_managed();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_managed(mem_addr_t addr,
                                                 size_t length) {
  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  while (start_page <= end_page) {
    m_data[start_page].set_managed();
    start_page++;
  }
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_compressed(mem_addr_t addr,
                                                  size_t length) {
  mem_addr_t page_index = get_page_num(addr + length - 1);
  return m_data[page_index].is_compressed();
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_compressed(mem_addr_t addr,
                                                    size_t length) {
  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  while (start_page <= end_page) {
    m_data[start_page].set_compressed();
    start_page++;
  }
}
template <unsigned BSIZE>
float memory_space_impl<BSIZE>::get_pages_comp_ratio(mem_addr_t addr) {
  mem_addr_t start_page = get_page_num(addr);
  assert(0&&"this function is not used in current version\n");
  return m_data[start_page].get_comp_ratio();
}
//set compression ratio for a page address;
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_page_comp_ratio(mem_addr_t addr, float comp_ratio) {
  mem_addr_t start_page = get_page_num(addr);
  m_data[start_page].set_comp_ratio(comp_ratio);
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_comp_band(mem_addr_t addr,
                                                   size_t length) {
  if (band_on) {
    mem_addr_t start_page = get_page_num(addr);
    mem_addr_t end_page = get_page_num(addr + length - 1);
    while (start_page <= end_page) {
        m_data[start_page].set_comp_bandwidth(comp_band_scalar);
      m_data[start_page].set_decomp_bandwidth(decomp_band_scalar);
      start_page++;
    }
    band_count++;
  }
}
template <unsigned BSIZE>
float memory_space_impl<BSIZE>::get_pages_comp_band(mem_addr_t addr) {
  if (band_on) {
    mem_addr_t start_page = get_page_num(addr);
    return m_data[start_page].get_comp_bandwidth();
  }
}
template <unsigned BSIZE>
float memory_space_impl<BSIZE>::get_pages_decomp_band(mem_addr_t addr) {
  if (band_on) {
    mem_addr_t start_page = get_page_num(addr);
    return m_data[start_page].get_decomp_bandwidth();
  }
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_decomp_band(
    mem_addr_t addr,
    size_t length) {  // deprecated
  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  float temp = 16;
  while (start_page <= end_page) {
    m_data[start_page].set_decomp_bandwidth(temp);
    start_page++;
  }
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_pages_decompressed(mem_addr_t addr,
                                                      size_t length) {
  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  while (start_page <= end_page) {
    m_data[start_page].set_decompressed();
    start_page++;
  }
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_block(mem_addr_t blk_idx,
                                                 mem_addr_t addr, size_t length,
                                                 void *data) const {//blk_idx is page number
  if ((addr + length) > (blk_idx + 1) * BSIZE) {//if the address + length is greater than the page size
    printf(
        "GPGPU-Sim PTX: ERROR * access to memory \'%s\' is unaligned : "
        "addr=0x%x, length=%zu\n",
        m_name.c_str(), addr, length);
    printf(
        "GPGPU-Sim PTX: (addr+length)=0x%lx > 0x%x=(index+1)*BSIZE, "
        "index=0x%x, BSIZE=0x%x\n",
        (addr + length), (blk_idx + 1) * BSIZE, blk_idx, BSIZE);
    throw 1;
  }
  typename map_t::const_iterator i = m_data.find(blk_idx);//find the page number in the map
  if (i == m_data.end()) {//if the page number is not found in the map
    for (size_t n = 0; n < length; n++)
      ((unsigned char *)data)[n] = (unsigned char)0;//if the page number is not found in the map, then set the data to 0
  } else {
    unsigned offset = addr & (BSIZE - 1);//get the offset of the page
    unsigned nbytes = length;//length is the number of bytes to read
    i->second.read(offset, nbytes, (unsigned char *)data);//read the data from the page, data is the destination pointer of the data
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read(mem_addr_t addr, size_t length,
                                    void *data) const {
  mem_addr_t index = addr >> m_log2_block_size;//same as get_page_num
  if ((addr + length) <= (index + 1) * BSIZE) {//BSIZE = page size, 
    // fast route for intra-block access
    read_single_block(index, addr, length, data);
  } else {
    // slow route for inter-block access
    unsigned nbytes_remain = length;
    unsigned dst_offset = 0;
    mem_addr_t current_addr = addr;

    while (nbytes_remain > 0) {
      unsigned offset = current_addr & (BSIZE - 1);
      mem_addr_t page = current_addr >> m_log2_block_size;
      mem_addr_t access_limit = offset + nbytes_remain;
      if (access_limit > BSIZE) {
        access_limit = BSIZE;
      }

      size_t tx_bytes = access_limit - offset;
      read_single_block(page, current_addr, tx_bytes,
                        &((unsigned char *)data)[dst_offset]);

      // advance pointers
      dst_offset += tx_bytes;
      current_addr += tx_bytes;
      nbytes_remain -= tx_bytes;
    }
    assert(nbytes_remain == 0);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::print(const char *format, FILE *fout) const {
  typename map_t::const_iterator i_page;

  for (i_page = m_data.begin(); i_page != m_data.end(); ++i_page) {
    fprintf(fout, "%s %08x:", m_name.c_str(), i_page->first);
    i_page->second.print(format, fout);
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_watch(addr_t addr, unsigned watchpoint) {
  m_watchpoints[watchpoint] = addr;
}
// get page number from a virtual address
template <unsigned BSIZE>
mem_addr_t memory_space_impl<BSIZE>::get_page_num(mem_addr_t addr) {//shift right to get the page number
  return addr >> m_log2_block_size;
}
template <unsigned BSIZE>
mem_addr_t memory_space_impl<BSIZE>::get_page_addr(mem_addr_t pg_index) {//shift left to get the page address
  return pg_index << m_log2_block_size;
}
// check whether the valid flag of corresponding physical page is set or not
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_valid(mem_addr_t pg_index) {
  // asserts whether the physical page is allocated.
  // should never happen as they are allocated while memcpy.
  assert(m_data.find(pg_index) != m_data.end());
  return m_data[pg_index].is_valid();
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_evicted_compressed(mem_addr_t pg_index) {
  // asserts whether the physical page is allocated.
  // should never happen as they are allocated while memcpy.
  assert(m_data.find(pg_index) != m_data.end());
  return m_data[pg_index].is_evicted_compressed();
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_evicted_compressed(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].set_evicted_compressed();
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_evicted_compressed(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].clear_evicted_compressed();
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_sampled(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  return m_data[pg_index].is_sampled();
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_sampled(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].set_sampled();
} 
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_sampled(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].clear_sampled();
}
// set the valid flag of corresponding physical page
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::validate_page(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].validate_page();
}

// clear the valid flag of corresponding physical page
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::invalidate_page(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].invalidate_page();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_iter_compressed(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].set_compressed();
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_iter_decompressed(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  m_data[pg_index].set_decompressed();
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_iter_compressed(mem_addr_t pg_index) {
  assert(m_data.find(pg_index) != m_data.end());
  return m_data[pg_index].is_compressed();
}
// a variable accessed by a memory address and the datatype size may exceed a
// page boundary method returns list of page numbers if at all they are faulty
// or invalid
template <unsigned BSIZE>
std::list<mem_addr_t> memory_space_impl<BSIZE>::get_faulty_pages(
    mem_addr_t addr, size_t length) {
  std::list<mem_addr_t> page_list;

  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);

  while (start_page <= end_page) {
    if (!is_valid(start_page)) {//find invalid page
      if (!is_iter_compressed(start_page)) {//pass compressed page only doing uncompressed
        page_list.push_back(start_page);
      }
    }
    start_page++;
  }

  return page_list;
}
template <unsigned BSIZE>
std::list<mem_addr_t>
memory_space_impl<BSIZE>::get_compressed_pages(  // bst edit for predecomp
    mem_addr_t addr, size_t length) {
  std::list<mem_addr_t> page_list;

  mem_addr_t start_page = get_page_num(addr);
  mem_addr_t end_page = get_page_num(addr + length - 1);
  while (start_page <= end_page) {
    if (is_iter_compressed(start_page)) {
      page_list.push_back(start_page);
    }
    start_page++;
  }
  return page_list;
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::alloc_page_by_byte(size_t size) {
  if (num_free_pages >= size) {
    num_free_pages -= size;
    return true;
  } else {
    return false;
  }
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::alloc_pages(double num) {
  assert(num_free_pages >= num*4096);
  assert(!isinf(num_free_pages));
  num_free_pages -= num*4096;
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::free_pages(double num) {
  num_free_pages += num*4096;
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::comp_free_pages(size_t num, float comp_ratio) {
  return change_size(num, true, comp_ratio);
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::change_size_comp_evict_with_comp_size(size_t tree_size, size_t comp_size, bool increase) {//increase is false mean recover_comp_size, true mean compressed to invalid
  double  change_size_from_gddr = (double)tree_size-(double)comp_size; 
  num_free_pages += increase? (double)comp_size : -(double)comp_size;
  num_gddr_pages -= increase? change_size_from_gddr : -change_size_from_gddr;
  comp_added_pages -= increase? change_size_from_gddr : -change_size_from_gddr;
  assert(num_free_pages >= (double)-100000000);
  if(comp_added_pages < 0){
    printf("bst debug change_size_comp_evict_with_comp_size comp_added_pages < 0\n");
    return false;
    }
  return true;
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::change_size_comp_with_comp_size(size_t comp_size, bool increase) {//increase is false mean decompress, true mean compress comp_size = tree_size - comp_size
  // As in prior UVM models, assume compaction makes compression-released space available as free pages.
  double change_size_from_gddr = (double)comp_size;
  num_free_pages += increase ? change_size_from_gddr : -change_size_from_gddr;
  num_gddr_pages += increase ? change_size_from_gddr : -change_size_from_gddr;
  comp_added_pages += increase ? change_size_from_gddr : -change_size_from_gddr;
  if(comp_added_pages < 0){
    printf("bst debug change_size_comp_with_comp_size comp_added_pages < 0\n");
    return false;
  }
  return true;
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::comp_evict_free_pages(size_t num, float comp_ratio) {//we change num to real size not page number
  bool ret = true;

  assert(comp_ratio >= 1.0f);
  // 감소량 계산
  float change = num * 4096 / comp_ratio;

  // 오차 보정을 위해 roundf() 적용 (소수점 3자리까지 유지)
  change = roundf(change * 1000) / 1000;

  // 값 변경
  num_free_pages += change;
  num_gddr_pages -= change;
  comp_added_pages -= change;

  return ret;
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::change_size(
    size_t num, bool increase, float comp_ratio) {  
      if(comp_ratio <= 1) {
        return true;
      }
  bool ret = true;

  float change = num * 4096 * (1.0f - 1.0f / comp_ratio);

  // 오차 보정을 위해 roundf 적용
  change = roundf(change * 1000) / 1000;  // 소수점 3자리까지 유지

  num_free_pages += (increase ? change : -change);
  num_gddr_pages += (increase ? change : -change);
  comp_added_pages += (increase ? change : -change);

  if (!increase && (comp_added_pages < 0 || num_gddr_pages < init_num_gddr)) {
    ret = false;
  }

  return ret;
}

template <unsigned BSIZE>
double memory_space_impl<BSIZE>::get_free_pages() {
  return num_free_pages/4096;
}
template <unsigned BSIZE>
double memory_space_impl<BSIZE>::get_total_pages() {
  return num_gddr_pages/4096;
}
template <unsigned BSIZE>
double memory_space_impl<BSIZE>::get_init_pages() {
  return init_num_gddr/4096;
}
template <unsigned BSIZE>
double memory_space_impl<BSIZE>::get_comp_free_pages() {
  return comp_added_pages/4096;
}
// if the already allocated pages and about to allocate pages(in read stage
// queue) reaches the buffer size in gddr, then it should start eviction
// procedure
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::should_evict_page(
    double read_stage_queue_size, double write_stage_queue_size,
    double decomp_stage_queue_size, double comp_stage_queue_size,
    float eviction_buffer_percentage) {
  return ((double)(write_stage_queue_size +
                   comp_stage_queue_size)) <
         ((((double)num_gddr_pages/4096) * eviction_buffer_percentage / 100) +
          ((double)(read_stage_queue_size - num_free_pages/4096 +
                    decomp_stage_queue_size)));
}
template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::should_comp_page(
    double read_stage_queue_size, double write_stage_queue_size,
    double decomp_stage_queue_size, double comp_stage_queue_size,
    float comp_buffer_percentage) {
  return ((double)(write_stage_queue_size +
                   comp_stage_queue_size)) <
         ((((double)num_gddr_pages/4096) * comp_buffer_percentage / 100) +
          ((double)(read_stage_queue_size - num_free_pages/4096 +
                    decomp_stage_queue_size)));
}

template <unsigned BSIZE>
float memory_space_impl<BSIZE>::get_projected_occupancy(
    double read_stage_queue_size, double write_stage_queue_size,
    float eviction_buffer_percentage) {
  return ((((double)num_gddr_pages/4096) * eviction_buffer_percentage / 100) +
          ((double)(num_gddr_pages/4096 - num_free_pages/4096 + read_stage_queue_size))) /
         ((double)(write_stage_queue_size + num_gddr_pages/4096));
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::read_single_page(mem_addr_t page_num, void *data) {
  m_data[page_num].read(0, BSIZE, (unsigned char*)data);
}
template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_page_dirty(mem_addr_t pg_index) {
  m_data[pg_index].set_dirty();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_page_dirty(mem_addr_t pg_index) {
  m_data[pg_index].clear_dirty();
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_dirty(mem_addr_t pg_index) {
  return m_data[pg_index].is_dirty();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::set_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].set_access();
}

template <unsigned BSIZE>
bool memory_space_impl<BSIZE>::is_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].is_access();
}

template <unsigned BSIZE>
void memory_space_impl<BSIZE>::clear_page_access(mem_addr_t pg_index) {
  return m_data[pg_index].clear_access();
}

// get size in bytes starting from the addr to the end of the page
// first get starting address of the page containing the address
// then get how many bytes are there starting from the page to the given address
// then subtract it from the total page size
template <unsigned BSIZE>
size_t memory_space_impl<BSIZE>::get_data_size(mem_addr_t addr) {
  return BSIZE - (addr - (mem_addr_t)((addr >> m_log2_block_size)
                                      << m_log2_block_size));
}

template <unsigned BSIZE>
size_t memory_space_impl<BSIZE>::get_page_size() {
  return BSIZE;
}

template <unsigned BSIZE>
mem_addr_t memory_space_impl<BSIZE>::get_mem_addr(mem_addr_t pg_index) {
  return pg_index << m_log2_block_size;
}

template <unsigned BSIZE>
size_t memory_space_impl<BSIZE>::get_cache_line_index(mem_addr_t addr) {
  return (addr - get_mem_addr(get_page_num(addr)))/128;
}

template <unsigned BSIZE>//1=evict+comp_evict 2=evict 3=comp_evict 4=comp
int memory_space_impl<BSIZE>::should_page_action(
    double read_stage_queue_size, double write_stage_queue_size,
    double decomp_stage_queue_size, double comp_stage_queue_size,
    double comp_evict_stage_queue_size,
    float eviction_buffer_percentage, float comp_buffer_percentage,
    float comp_evict_start_percentage) {
  
  // 공통 계산 부분을 한 번만 수행
  double unallocated = write_stage_queue_size + comp_stage_queue_size + num_free_pages/4096 ;
  double requested = read_stage_queue_size + decomp_stage_queue_size;
  
  // 각 조건 계산
  double evict_request_and_buffer = (init_num_gddr/4096 * eviction_buffer_percentage / 100) + requested;
  bool should_evict = unallocated < evict_request_and_buffer;//evict buffer보다 낮은 경우
  
  double comp_request_and_buffer = (init_num_gddr/4096 * comp_buffer_percentage / 100) + requested;
  bool should_comp = unallocated < comp_request_and_buffer;//comp buffer보다 낮은 경우
  
  bool should_comp_evict = init_num_gddr /4096 * comp_evict_start_percentage / 100 < (comp_added_pages /4096)- comp_evict_stage_queue_size;//30%이상이 압축되서 확보된 상황에서는 너무 압축이 많이 됐으므로 압축페이지삭제
  
  // 조건에 따른 액션 결정 (우선순위: evict > comp_evict > comp)
  if (should_evict && should_comp_evict) {//evict buffer(현재는 0)보다 낮아지고, 30%이상이 압축되서 컴프레션 버퍼가 적은경우
      return 1; // evict + comp_evict (1순위 - 가장 긴급)
  }
  else if (should_evict) {//evict buffer(현재는 0)보다 낮아졌으나, compression 버퍼는 아직남은경우
    return 2; // evict만 (2순위)
  }
  else if (should_comp_evict&&should_comp) {///5%이하로 프리페이지가 적은경우 + 압축이 많이 된 경우
    return 3; // comp_evict만 (3순위)
  }
  else if (should_comp) {//compression 버퍼가 남아있는 경우, 5%이하인경우
    return 4; // comp만 (5순위 - 가장 낮은 우선순위)
  }
  else {
    return 0; // 아무 액션 없음
  }
}

template class memory_space_impl<32>;
template class memory_space_impl<64>;
// template class memory_space_impl<8192>;
template class memory_space_impl<4096>;
template class memory_space_impl<1024 * 1024 * 2>;
template class memory_space_impl<16 * 1024>;

void g_print_memory_space(memory_space *mem, const char *format = "%08x",
                          FILE *fout = stdout) {
  mem->print(format, fout);
}

#ifdef UNIT_TEST

int main(int argc, char *argv[]) {
  int errors_found = 0;
  memory_space *mem = new memory_space_impl<32>("test", 4);
  // write address to [address]
  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4)
    mem->write(addr, 4, &addr, NULL, NULL);

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 4) {
    unsigned tmp = 0;
    mem->read(addr, 4, &tmp);
    if (tmp != addr) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp, addr);
    }
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned char val = (addr + 128) % 256;
    mem->write(addr, 1, &val, NULL, NULL);
  }

  for (mem_addr_t addr = 0; addr < 16 * 1024; addr += 1) {
    unsigned tmp = 0;
    mem->read(addr, 1, &tmp);
    unsigned char val = (addr + 128) % 256;
    if (tmp != val) {
      errors_found = 1;
      printf("ERROR ** mem[0x%x] = 0x%x, expected 0x%x\n", addr, tmp,
             (unsigned)val);
    }
  }

  if (errors_found) {
    printf("SUMMARY:  ERRORS FOUND\n");
  } else {
    printf("SUMMARY: UNIT TEST PASSED\n");
  }
}

#endif
