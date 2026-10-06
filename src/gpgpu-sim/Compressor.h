#ifndef __COMPRESSOR_HH__
#define __COMPRESSOR_HH__
#pragma once


#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <cstdio>
#include <climits>
#include <string>
#include <cmath>
#include <cassert>
//--------------------------------------------------------------------
//#define LSIZE (512)  // in bits
#define LSIZE (1024)  // in bits

//--------------------------------------------------------------------
using namespace std;

typedef bool                BOOL;
typedef uint8_t             UINT8;
typedef uint16_t            UINT16;
typedef uint32_t            UINT32;
typedef uint64_t            UINT64;
typedef int8_t              INT8;
typedef int16_t             INT16;
typedef int32_t             INT32;
typedef int64_t             INT64;
typedef float               FLT32;
typedef double              FLT64;

typedef unsigned long long  CNT;
typedef unsigned            LENGTH;
typedef unsigned long long  KEY;

static const UINT32 _MAX_BYTES_PER_LINE     = LSIZE/8;
static const UINT32 _MAX_WORDS_PER_LINE     = LSIZE/16;
static const UINT32 _MAX_DWORDS_PER_LINE    = LSIZE/32;
static const UINT32 _MAX_QWORDS_PER_LINE    = LSIZE/64;
static const UINT32 _MAX_FLOATS_PER_LINE    = LSIZE/32;
static const UINT32 _MAX_DOUBLES_PER_LINE   = LSIZE/64;

typedef struct { UINT64 m : 52; UINT64 e : 11; UINT64 s : 1; } FLT64_P;
typedef struct { UINT32 m : 23; UINT32 e : 8;  UINT32 s : 1; } FLT32_P;
// LCP 관련 상수
static const unsigned UNCOMPRESSED_SIZE = _MAX_BYTES_PER_LINE;             // 128바이트
static const unsigned NUM_LINES_PER_PAGE = 4096 / UNCOMPRESSED_SIZE;         // 32라인
static const unsigned METADATA_OVERHEAD = 64;                                // 메타데이터 영역
#define METADATA_PER_EXCEPTION 4 // 4 bytes per exception
#define BASE_METADATA_OVERHEAD 64 // Fixed metadata overhead (e.g., for page table entry)
// BDI 후보 타겟 슬롯 크기
static const unsigned candidateTargetSizes_BDI[] = {24, 40, 72};
static const unsigned numCandidates_BDI = sizeof(candidateTargetSizes_BDI) / sizeof(candidateTargetSizes_BDI[0]);

// OS에서 지원하는 물리 페이지 크기 후보
static const unsigned allowedPageSizes[] = {
    512,1024,2048,4096
};
static const unsigned numAllowed = sizeof(allowedPageSizes) / sizeof(allowedPageSizes[0]);


//--------------------------------------------------------------------
typedef union CACHELINE_DATA {
    UINT8   byte[_MAX_BYTES_PER_LINE];
    UINT16  word[_MAX_WORDS_PER_LINE];
    UINT32  dword[_MAX_DWORDS_PER_LINE];
    UINT64  qword[_MAX_QWORDS_PER_LINE];

    INT8    s_byte[_MAX_BYTES_PER_LINE];
    INT16   s_word[_MAX_WORDS_PER_LINE];
    INT32   s_dword[_MAX_DWORDS_PER_LINE];
    INT64   s_qword[_MAX_QWORDS_PER_LINE];

    FLT32   flt[_MAX_FLOATS_PER_LINE];
    FLT64   dbl[_MAX_DOUBLES_PER_LINE];
    FLT32_P flt_p[_MAX_FLOATS_PER_LINE];
    FLT64_P dbl_p[_MAX_DOUBLES_PER_LINE];
} CACHELINE_DATA;

//--------------------------------------------------------------------
class Compressor {
public:
    // constructor / destructor        
    Compressor(const string _name) : name(_name), totalPatternCnt(0ull), totalLineCnt(0ull) { }
    ~Compressor() {}
public:
    // methods
    string getName() const { return name; }
    CNT getPatternCnt(INT64 pattern) { auto it = patternCounterMap.find(pattern); return (it==patternCounterMap.end()) ? 0 : it->second; }

    virtual LENGTH compressLine(CACHELINE_DATA* line, UINT64 line_addr) = 0;
    virtual void reset() {
        totalPatternCnt = 0ull;
        totalLineCnt = 0ull;
        patternCounterMap.clear();
        lengthMap.clear();
    }

protected:
    void compressFile(FILE *fd);
    virtual void countPattern(INT64 pattern);
    virtual void countLineResult(LENGTH length);
    double getCoverage(int thresholdBitSize);
    
public:
    virtual void printSummary(FILE* fd);
    virtual void printDetails(FILE* fd, string bench_name) const;

protected:
    string name;

    CNT totalPatternCnt;
    CNT totalLineCnt;
    map<INT64, CNT> patternCounterMap;
    map<LENGTH, CNT> lengthMap;
};

bool sign_extended(UINT64 value, UINT8 bit_size);
bool zero_extended(UINT64 value, UINT8 bit_size);

class BPCompressor : public Compressor {
public:
    BPCompressor(const string name) : Compressor(name) {}
    unsigned compressLine(CACHELINE_DATA* line, UINT64 line_addr);
    virtual unsigned encodeFirst(INT32 sym);
    virtual unsigned encodeDeltas(INT32* DBP, INT32* DBX);
};

// 절대값 계산 함수 선언만 남기고 구현 제거
unsigned long long my_llabs(long long x);
unsigned my_abs(int x);

// 버퍼를 배열로 변환하는 함수 선언만 남기고 구현 제거
long long unsigned * convertBuffer2Array(char * buffer, unsigned size, unsigned step);

// 캐시라인 검사 함수 선언만 남기고 구현 제거
int isZeroPackable(long long unsigned * values, unsigned size);
int isSameValuePackable(long long unsigned * values, unsigned size);

// 다중 베이스 압축 함수 선언만 남기고 구현 제거
unsigned multBaseCompression(long long unsigned * values, unsigned size, unsigned blimit, unsigned bsize);

// BDI 압축 함수 선언만 남기고 구현 제거
unsigned BDICompress(char * buffer, unsigned _blockSize);
size_t CUREMEM_compress(Compressor* comp, CACHELINE_DATA* pageData);//압축된 페이지 사이즈 반환.
class BDICompressor : public Compressor {
public:
    BDICompressor(const string _name) : Compressor(_name) { }
    virtual uint32_t compressLine(CACHELINE_DATA* line, uint64_t line_addr) {//return 압축된 바이트 단위 사이즈 
        return BDICompress((char*)line->byte, _MAX_BYTES_PER_LINE);//BDICompress 사용 byte(char) 타입 사용, 캐시라인 크기 128바이트
    }
};

// LCP 압축 페이지 메트릭스 구조체
typedef struct {
    unsigned long long pageNumber;   // 페이지 번호
    unsigned targetSize;             // 선택된 고정 압축 슬롯 크기 (바이트)
    unsigned allocatedPageSize;      // 할당된 물리 페이지 크기 (128,256,512,1024,2048,4096 중 하나)
    unsigned totalUsedSize;          // 압축 데이터 영역 + 메타데이터 + 예외 영역으로 사용된 전체 크기
    unsigned remainingCapacity;      // 할당된 페이지 내 남은 용량 (바이트)
    unsigned numExceptions;          // 현재 예외 캐시라인 수
    unsigned maxExceptions;          // 현재 할당된 예외 슬롯 수
    int penalty;                     // 재압축 결과가 이전 할당 크기보다 커졌거나 초기 접근이면 1, 아니면 0
    bool exceptionLines[NUM_LINES_PER_PAGE]; // 각 캐시라인의 예외 상태 추적
    unsigned cacheSizes[NUM_LINES_PER_PAGE]; // 각 캐시라인의 압축 크기
    int exceptionSlots[NUM_LINES_PER_PAGE];  // 사용안함
} LCP_CompressedPage;
/* we do not use this class because we only consider 4KB level compression
class LCP_compressed_chunk{
    public:
        map<unsigned long long, LCP_CompressedPage> LCP_compressed_pages_map; // for LCP compression with double mapping
        size_t allocated_size;
    public:
        LCP_compressed_chunk(){
            LCP_compressed_pages_map = map<unsigned long long, LCP_CompressedPage>();
            allocated_size = 0;
        }
        size_t allocated_chunk(unsigned long long chunk_addr_t){
            size_t current_size=0;
            for(size_t i=0; i<64*1024;i+=4096){
                LCP_compressed_pages_map.insert(make_pair(chunk_addr_t+i, LCP_CompressedPage()));
                current_size+=LCP_compressed_pages_map[chunk_addr_t+i].allocatedPageSize;
            }
            allocated_size=current_size;
            return current_size;
        }
        size_t get_allocated_size(){
            return allocated_size;
        }
        void set_allocated_size(size_t size){
            allocated_size = size;
        }
        void add_LCP_compressed_page(unsigned long long page_num, LCP_CompressedPage page){
            LCP_compressed_pages_map[page_num] = page;
        }
        void remove_LCP_compressed_page(unsigned long long page_num){
            LCP_compressed_pages_map.erase(page_num);
        }
        void clear_LCP_compressed_pages_map(){
            LCP_compressed_pages_map.clear();
        }
};*/
// LCP 압축 함수 선언
LCP_CompressedPage compressPageLCP(Compressor* comp, CACHELINE_DATA* pageData, unsigned prevAllocatedSize, unsigned long long pageNum,int updatedLine = -1, LCP_CompressedPage* prevCP = NULL);
LCP_CompressedPage updateCompressedCacheLine(Compressor* comp, CACHELINE_DATA* pageData, const LCP_CompressedPage& prevState, unsigned cacheLineIndex);

#endif /* __COMPRESSOR_HH__ */