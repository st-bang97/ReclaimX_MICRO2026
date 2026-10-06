#ifndef __LCP_HH__
#define __LCP_HH__

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <climits>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <sstream>

//--------------------------------------------------------------------
//#define LSIZE (512)  // in bits
#define LSIZE (1024)  // in bits

//--------------------------------------------------------------------
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
    Compressor(const std::string _name) : name(_name) { }
    ~Compressor() {}
public:
    // methods
    std::string getName() const { return name; }
    CNT getPatternCnt(INT64 pattern) { auto it = patternCounterMap.find(pattern); return (it==patternCounterMap.end()) ? 0 : it->second; }

    virtual LENGTH compressLine(CACHELINE_DATA* line, UINT64 line_addr) = 0;
    virtual void reset() {
        totalPatternCnt = 0ull;
        totalLineCnt = 0ull;
        patternCounterMap.clear();
        lengthMap.clear();
    }

protected:
    void compressFile(FILE *fd) {
        CACHELINE_DATA line;

        reset();

        while (true) {
            if (fread(&line, _MAX_BYTES_PER_LINE, 1, fd)!=1) {
                break;
            }
            compressLine(&line, 0);
        }
    }
    virtual void countPattern(INT64 pattern) {
        totalPatternCnt++;
        auto it = patternCounterMap.find(pattern);
        if (it==patternCounterMap.end()) {
            patternCounterMap.insert(std::pair<INT64, CNT>(pattern, 1ull));
        } else {
            it->second++;
        }
    }
    virtual void countLineResult(LENGTH length) {
        totalLineCnt++;
        auto it = lengthMap.find(length);
        if (it==lengthMap.end()) {
            lengthMap.insert(std::pair<LENGTH, CNT>(length, 1ull));
        } else {
            it->second++;
        }
    }
    double getCoverage(int thresholdBitSize) {
        CNT accumLineCnt = 0ull;
        for (int i=0; i<=thresholdBitSize; i++) {
            auto it = lengthMap.find(i);
            if (it!=lengthMap.end()) {
                accumLineCnt += it->second;
            }
        }
        return accumLineCnt*1./totalLineCnt;
    }
public:
    virtual void printSummary(FILE* fd) {
        CNT accumCnt;

        fprintf(fd, "Comp\t%s\n", name.c_str());
        // Input bench data
        fprintf(fd, "dataCnt32\t%lld\n", totalLineCnt*_MAX_DWORDS_PER_LINE);

        // Compression results
        accumCnt = 0ull;
        for (auto it = lengthMap.cbegin(); it != lengthMap.cend(); ++it) {
            accumCnt += (it->first * it->second);
        }
        fprintf(fd, "CompRatio  \t%f\n", totalLineCnt*_MAX_DWORDS_PER_LINE*32./accumCnt);
        fprintf(fd, "Comp32b    \t%f\n", accumCnt*1./totalLineCnt/_MAX_DWORDS_PER_LINE);
        fprintf(fd, "Comp8b     \t%f\n", accumCnt*1./totalLineCnt/_MAX_BYTES_PER_LINE);
        fprintf(fd, "CompDist   \t");
        accumCnt = 0ull;
        for (int i=0; i<600; i++) {
            auto it = lengthMap.find(i);
            if (it!=lengthMap.end()) {
                accumCnt += it->second;
            }
            if (i%8==0) {
                fprintf(fd, "%f\t", accumCnt*1./totalLineCnt);
            }
        }
        fprintf(fd, "\n");
    }
    virtual void printDetails(FILE* fd, std::string bench_name) const {
        fprintf(fd, "Pattern frequency\n");
        // if too huge
        if (patternCounterMap.size() > (1<<16)) {
            std::map<INT64, CNT> patternGroupCounterMap;
            // group into 65536 groups
            int offset = 48;
            for (auto it = patternCounterMap.cbegin(); it != patternCounterMap.cend(); ++it) {
                INT64 patternGroup = it->first>>offset;
                auto it2 = patternGroupCounterMap.find(patternGroup);
                if (it2 == patternGroupCounterMap.end()) {
                    patternGroupCounterMap.insert(std::pair<INT64, CNT>(patternGroup, it->second));
                } else {
                    it2->second += it->second;
                }
            }
            if (true) {     // sorted based on pattern
                std::vector<std::pair<INT64, CNT>>v;
                for (auto it = patternGroupCounterMap.cbegin(); it != patternGroupCounterMap.cend(); ++it) {
                    v.push_back(std::pair<INT64, CNT>(it->first, it->second));
                }
                // sorting
                std::sort(v.begin(), v.end());

                for (auto it = v.cbegin(); it != v.cend(); ++it) {
                    fprintf(fd, "%16lx\t%f\t%f\t%lld\n", (UINT64) it->first, it->second*1./totalPatternCnt, -log2(it->second*1./totalPatternCnt), it->second);
                }
            } else {        // sorted based on frequency
                std::vector<std::pair<CNT, INT64>>v;
                for (auto it = patternGroupCounterMap.cbegin(); it != patternGroupCounterMap.cend(); ++it) {
                    v.push_back(std::pair<INT64, CNT>(it->second, it->first));
                }
                // sorting
                std::sort(v.begin(), v.end());

                for (auto it = v.cbegin(); it != v.cend(); ++it) {
                    fprintf(fd, "%16lx\t%f\t%f\n", (UINT64) it->second, it->first*1./totalPatternCnt, -log2(it->first*1./totalPatternCnt));
                }
            }
        } else {
            if (true) {     // sorted based on pattern
                std::vector<std::pair<INT64, CNT>>v;
                for (auto it = patternCounterMap.cbegin(); it != patternCounterMap.cend(); ++it) {
                    v.push_back(std::pair<INT64, CNT>(it->first, it->second));
                }
                // sorting
                std::sort(v.begin(), v.end());

                for (auto it = v.cbegin(); it != v.cend(); ++it) {
                    fprintf(fd, "%16lx\t%f\t%f\t%lld\n", (UINT64) it->first, it->second*1./totalPatternCnt, -log2(it->second*1./totalPatternCnt), it->second);
                }
            } else {        // sorted based on frequency
                std::vector<std::pair<CNT, INT64>>v;
                for (auto it = patternCounterMap.cbegin(); it != patternCounterMap.cend(); ++it) {
                    v.push_back(std::pair<INT64, CNT>(it->second, it->first));
                }
                // sorting
                std::sort(v.begin(), v.end());

                for (auto it = v.cbegin(); it != v.cend(); ++it) {
                    fprintf(fd, "%16lx\t%f\t%f\n", (UINT64) it->second, it->first*1./totalPatternCnt, -log2(it->first*1./totalPatternCnt));
                }
            }
        }

        fprintf(fd, "Compressed line size\n");
        for (auto it=lengthMap.cbegin(); it!=lengthMap.cend(); it++) {
            fprintf(fd, "%02d\t%f\t%f\n", it->first, it->second*1./ totalLineCnt, -log2(it->second*1./totalLineCnt));
        }
    }
protected:
    std::string name;

    CNT totalPatternCnt;
    CNT totalLineCnt;
    std::map<INT64, CNT> patternCounterMap;
    std::map<LENGTH, CNT> lengthMap;
};

bool sign_extended(UINT64 value, UINT8 bit_size) {
    UINT64 max = (1ULL << (bit_size-1)) - 1;    // bit_size: 4 -> ...00000111
    UINT64 min = ~max;                          // bit_size: 4 -> ...11111000
    return (value <= max) | (value >= min);
}
bool zero_extended(UINT64 value, UINT8 bit_size);

class BPCompressor : public Compressor {
public:
    BPCompressor(const std::string name) : Compressor(name) {}
    unsigned compressLine(CACHELINE_DATA* line, UINT64 line_addr) {
        INT64 deltas[31];
        for (int i=1; i<_MAX_DWORDS_PER_LINE; i++) {
            deltas[i-1] = ((INT64) line->s_dword[i]) - ((INT64) line->s_dword[i-1]);
        }

        INT32 prevDBP;
        INT32 DBP[33];
        INT32 DBX[33];
        for (int j=63; j>=0; j--) {
            INT32 buf = 0;
            for (int i=30; i>=0; i--) {
                buf <<= 1;
                buf |= ((deltas[i]>>j)&1);
            }
            if (j==63) {
                DBP[32] = buf;
                DBX[32] = buf;
                prevDBP = buf;
            } else if (j<32) {
                DBP[j] = buf;
                DBX[j] = buf^prevDBP;
                prevDBP = buf;
            } else {
                assert(buf==prevDBP);
                prevDBP = buf;
            }
        }
        
        // first 32-bit word in original form
        unsigned blkLength = encodeFirst(line->dword[0]);
        blkLength += encodeDeltas(DBP, DBX);

        countLineResult(blkLength);
        return blkLength/8;//바이트 단위 사이즈 반환
    }

    virtual unsigned encodeFirst(INT32 sym) {
        if (sym==0) {
            countPattern(256);
            return 3;
        } else if (sign_extended(sym, 4)) {
            countPattern(257);
            return (3+4);
        } else if (sign_extended(sym, 8)) {
            countPattern(258);
            return (3+8);
        } else if (sign_extended(sym, 16)) {
            countPattern(259);
            return (3+16);
        } else {
            countPattern(263);
            return (1+32);
        }
    }
    virtual unsigned encodeDeltas(INT32* DBP, INT32* DBX) {
        static const unsigned ZRL_CODE_SIZE[34] = {0, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
        static const unsigned singleOneSize = 10;
        static const unsigned consecutiveDoubleOneSize = 10;
        static const unsigned allOneSize = 5;
        static const unsigned zeroDBPSize = 5;

        unsigned length = 0;
        unsigned run_length = 0;
        bool firstNZDBX = false;
        bool secondNZDBX = false;
        for (int i=32; i>=0; i--) {
            if (DBX[i]==0) {
                run_length++;
            }
            else {
                if (run_length>0) {
                    countPattern(run_length-1);
                    assert(run_length!=33);
                    length += ZRL_CODE_SIZE[run_length];
                }
                run_length = 0;

                if (DBP[i]==0) {
                    length += zeroDBPSize;
                    countPattern(34);
                } else if (DBX[i]==0x7fffffff) {
                    length += allOneSize;
                    countPattern(35);
                } else {
                    int oneCnt = 0;
                    for (int j=0; j<32; j++) {
                        if ((DBX[i]>>j)&1) {
                            oneCnt++;
                        }
                    }
                    unsigned two_distance = 0;
                    int firstPos = -1;
                    if (oneCnt<=2) {
                        for (int j=0; j<32; j++) {
                            if ((DBX[i]>>j)&1) {
                                if (firstPos==-1) {
                                    firstPos = j;
                                } else {
                                    two_distance = j - firstPos;
                                }
                            }
                        }
                    }
                    if (oneCnt==1) {
                        length += singleOneSize;
                        countPattern(37+firstPos);
                    } else if ((oneCnt==2) && (two_distance==1)) {
                        length += consecutiveDoubleOneSize;
                        countPattern(69+firstPos);
                    } else {
                        length += 32;
                        countPattern(36);
                    }
                }
            }
        }
        if (run_length>0) {
            length += ZRL_CODE_SIZE[run_length];
            countPattern(run_length-1);
            assert(run_length<=33);
        }
        return length;
    }
};

// ---------------------- BDI 기반 압축 관련 함수 ----------------------

// 절대값 계산 (long long 버전)
static unsigned long long my_llabs(long long x) {
   unsigned long long t = x >> 63;
   return (x ^ t) - t;
}

// 절대값 계산 (int 버전)
static unsigned my_abs(int x) {
   unsigned t = x >> 31;
   return (x ^ t) - t;
}

// 버퍼를 step 단위로 배열로 변환 (예: 8바이트씩 읽어 64비트 정수 배열로)
long long unsigned * convertBuffer2Array(char * buffer, unsigned size, unsigned step) {
   long long unsigned * values = (long long unsigned *) malloc(sizeof(long long unsigned) * (size/step));
   unsigned int i, j;
   for (i = 0; i < size / step; i++) {
       values[i] = 0; // 모든 요소 0으로 초기화
   }
   for (i = 0; i < size; i += step) {
       for (j = 0; j < step; j++) {
           values[i / step] += ((long long unsigned)((unsigned char)buffer[i + j])) << (8 * j);
       }
   }
   return values;
}

// 캐시라인이 모두 0인지 검사
int isZeroPackable(long long unsigned * values, unsigned size) {
   unsigned int i;
   for (i = 0; i < size; i++) {
      if (values[i] != 0)
         return 0;
   }
   return 1;
}

// 캐시라인의 모든 값이 동일한지 검사
int isSameValuePackable(long long unsigned * values, unsigned size) {
   unsigned int i;
   for (i = 0; i < size; i++) {
      if (values[0] != values[i])
         return 0;
   }
   return 1;
}

/// 다중 베이스 압축: 여러 베이스를 고려하여 압축 가능성을 평가
unsigned multBaseCompression(long long unsigned * values, unsigned size, unsigned blimit, unsigned bsize) {
   unsigned long long limit = 0;
   unsigned BASES = 2; // 최대 2개의 베이스 사용
   switch(blimit) {
      case 1: limit = 0xFF; break;
      case 2: limit = 0xFFFF; break;
      case 4: limit = 0xFFFFFFFF; break;
      default:
         exit(1);
   }
   unsigned long long mbases[64];
   unsigned baseCount = 1;
   mbases[0] = 0;
   unsigned int i, j;
   for (i = 0; i < size; i++) {
      for (j = 0; j < baseCount; j++) {
         if ( my_llabs((long long)(mbases[j] - values[i])) > limit ) {
            mbases[baseCount++] = values[i];
         }
      }
      if (baseCount >= BASES)
         break;
   }
   unsigned compCount = 0;
   for (i = 0; i < size; i++) {
      for (j = 0; j < baseCount; j++) {
         if ( my_llabs((long long)(mbases[j] - values[i])) <= limit ) {
            compCount++;
            break;
         }
      }
   }
   unsigned mCompSize = blimit * compCount + bsize * BASES + (size - compCount) * bsize;
   if (compCount < size)
      return size * bsize;
   return mCompSize;
}

/// BDICompress: 주어진 _blockSize(예: 128바이트) 데이터 블록에 대해 BDI 기반 압축 크기를 산출
unsigned BDICompress(char * buffer, unsigned _blockSize) {
   long long unsigned * values = convertBuffer2Array(buffer, _blockSize, 8);
   unsigned bestCSize = _blockSize;
   unsigned currCSize = _blockSize;
   
   if (isZeroPackable(values, _blockSize / 8))
      bestCSize = 1;
   if (isSameValuePackable(values, _blockSize / 8))
      currCSize = 8;
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   
   currCSize = multBaseCompression(values, _blockSize / 8, 1, 8);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   currCSize = multBaseCompression(values, _blockSize / 8, 2, 8);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   currCSize = multBaseCompression(values, _blockSize / 8, 4, 8);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   
   free(values);
   values = convertBuffer2Array(buffer, _blockSize, 4);
   if (isSameValuePackable(values, _blockSize / 4))
      currCSize = 4;
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   currCSize = multBaseCompression(values, _blockSize / 4, 1, 4);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   currCSize = multBaseCompression(values, _blockSize / 4, 2, 4);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   free(values);
   
   values = convertBuffer2Array(buffer, _blockSize, 2);
   currCSize = multBaseCompression(values, _blockSize / 2, 1, 2);
   bestCSize = (bestCSize > currCSize) ? currCSize : bestCSize;
   free(values);
   
   return bestCSize;
}

class BDICompressor : public Compressor {
public:
    BDICompressor(const std::string _name) : Compressor(_name) { }
    virtual uint32_t compressLine(CACHELINE_DATA* line, uint64_t line_addr) {//return 압축된 바이트 단위 사이즈 
        return BDICompress((char*)line->byte, _MAX_BYTES_PER_LINE);//BDICompress 사용 byte(char) 타입 사용, 캐시라인 크기 128바이트
    }
};

//--------------------------------------------------------------------
// LCP 프레임워크 관련 상수
static const unsigned UNCOMPRESSED_SIZE = _MAX_BYTES_PER_LINE;             // 128바이트
static const unsigned NUM_LINES_PER_PAGE = 4096 / UNCOMPRESSED_SIZE;         // 32라인
static const unsigned METADATA_OVERHEAD = 64;                                // 메타데이터 영역

// BDI 후보 타겟 슬롯 크기
static const unsigned candidateTargetSizes_BDI[] = {24, 40, 72};
static const unsigned numCandidates_BDI = sizeof(candidateTargetSizes_BDI) / sizeof(candidateTargetSizes_BDI[0]);

// OS에서 지원하는 물리 페이지 크기 후보
static const unsigned allowedPageSizes[] = {512, 1024, 2048, 4096};
static const unsigned numAllowed = sizeof(allowedPageSizes) / sizeof(allowedPageSizes[0]);

// allowedPageSizes 중 totalSize를 수용할 수 있는 최소 페이지 크기를 반환 (수용 불가면 0)
unsigned getAllocatedPageSize(unsigned totalSize) {
    for (unsigned i = 0; i < numAllowed; i++) {
        if (totalSize <= allowedPageSizes[i])
            return allowedPageSizes[i];
    }
    return 0;
}

// LCP 압축 페이지 메트릭스 구조체 (페이지 번호 포함)
typedef struct {
    unsigned long long pageNumber;   // 페이지 번호
    unsigned targetSize;             // 선택된 고정 압축 슬롯 크기 (바이트)
    unsigned allocatedPageSize;      // 할당된 물리 페이지 크기 (512,1024,2048,4096 중 하나)
    unsigned totalUsedSize;          // 압축 데이터 영역 + 메타데이터 + 예외 영역으로 사용된 전체 크기
    unsigned remainingCapacity;      // 할당된 페이지 내 남은 용량 (바이트)
    unsigned numExceptions;          // 압축 실패(예외) 캐시라인 수
    int penalty;                     // 재압축 결과가 이전 할당 크기보다 커졌거나 초기 접근이면 1, 아니면 0
} LCP_CompressedPage;

//--------------------------------------------------------------------
// compressPageLCP 함수
// comp: 사용할 Compressor 인스턴스 (BDICompressor 또는 BPCompressor)
// pageData: 4KB 페이지 데이터를 담은 CACHELINE_DATA 배열 (32라인)
// prevAllocatedSize: 이전에 할당받은 물리 페이지 크기 (초기 접근이면 0)
// pageNum: 페이지 번호
LCP_CompressedPage compressPageLCP(Compressor* comp, CACHELINE_DATA* pageData, unsigned prevAllocatedSize, unsigned long long pageNum) {
    LCP_CompressedPage cp;
    cp.pageNumber = pageNum;
    
    // 각 캐시라인에 대해 compressLine()을 호출하여 압축 길이(LENGTH)를 구함.
    unsigned cacheLineSizes[NUM_LINES_PER_PAGE];
    for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
        cacheLineSizes[i] = comp->compressLine(&pageData[i], 0);
    }
    
    // Compressor 종류에 따라 후보 타겟 배열 결정:
    // - BPCompressor(BPC)의 경우: 압축 성공(압축 길이 < 128) 캐시라인 중 가장 큰 압축 길이를 후보로 선택.
    //   모두 실패하면 128바이트로 처리.
    // - 그 외 (예: BDICompressor)는 고정 후보 배열 {24, 40, 72} 사용.
    std::vector<unsigned> candidateArray;
    if (comp->getName() == "BPCompressor") {
        unsigned candidateBP = 0;
        for (unsigned j = 0; j < NUM_LINES_PER_PAGE; j++) {
            if (cacheLineSizes[j] < UNCOMPRESSED_SIZE && cacheLineSizes[j] > candidateBP) {
                candidateBP = cacheLineSizes[j];
            }
        }
        if (candidateBP == 0)
            candidateBP = UNCOMPRESSED_SIZE;
        candidateArray.push_back(candidateBP);
    } else {
        for (unsigned i = 0; i < numCandidates_BDI; i++) {
            candidateArray.push_back(candidateTargetSizes_BDI[i]);
        }
    }
    
    // 각 후보에 대해 전체 사용 크기를 계산:
    // totalUsed = (NUM_LINES_PER_PAGE × candidate) + METADATA_OVERHEAD + (예외 수 × UNCOMPRESSED_SIZE)
    struct CandidateInfo {
        unsigned candidate;
        unsigned numExceptions;
        unsigned totalUsed;
        unsigned allocatedSize;
    };
    std::vector<CandidateInfo> candidates;
    for (unsigned c : candidateArray) {
        unsigned exceptions = 0;
        for (unsigned j = 0; j < NUM_LINES_PER_PAGE; j++) {
            if (cacheLineSizes[j] > c)
                exceptions++;
        }
        unsigned totalUsed = NUM_LINES_PER_PAGE * c + METADATA_OVERHEAD + exceptions * UNCOMPRESSED_SIZE;
        unsigned allocSize = getAllocatedPageSize(totalUsed);
        if (allocSize != 0) {
            candidates.push_back({c, exceptions, totalUsed, allocSize});
        }
    }
    
    // 후보 선택: 두 그룹으로 분류
    // Group1: allocatedSize <= prevAllocatedSize (이전 할당 내에서 가능한 후보)
    // Group2: allocatedSize > prevAllocatedSize
    CandidateInfo bestCandidate;
    bool group1Found = false;
    for (auto &cand : candidates) {
        if (prevAllocatedSize != 0 && cand.allocatedSize <= prevAllocatedSize) {
            if (!group1Found || cand.allocatedSize < bestCandidate.allocatedSize) {
                bestCandidate = cand;
                group1Found = true;
            }
        }
    }
    
    int penaltyOccurred = 0;
    unsigned finalAllocSize = 0;
    unsigned finalCandidate = 0;
    unsigned finalNumExceptions = 0;
    unsigned finalTotalUsed = 0;
    if (group1Found) {
        // Group1이 존재하면, allocatedPageSize는 이전 값을 유지하고 새 totalUsed를 업데이트.
        finalAllocSize = prevAllocatedSize;
        finalCandidate = bestCandidate.candidate;
        finalNumExceptions = bestCandidate.numExceptions;
        finalTotalUsed = bestCandidate.totalUsed;
        penaltyOccurred = 0;
    } else {
        // Group1이 없으면, 모든 후보 중 최소 allocatedSize를 선택하고 penalty 발생.
        if (!candidates.empty()) {
            bestCandidate = candidates[0];
            for (auto &cand : candidates) {
                if (cand.allocatedSize < bestCandidate.allocatedSize) {
                    bestCandidate = cand;
                }
            }
            finalAllocSize = bestCandidate.allocatedSize;
            finalCandidate = bestCandidate.candidate;
            finalNumExceptions = bestCandidate.numExceptions;
            finalTotalUsed = bestCandidate.totalUsed;
            penaltyOccurred = 1;
        } else {
            // 후보가 하나도 없으면, 기본 4096 사용.
            finalAllocSize = 4096;
            finalCandidate = UNCOMPRESSED_SIZE;
            finalNumExceptions = NUM_LINES_PER_PAGE;
            finalTotalUsed = 4096;
            penaltyOccurred = 1;
        }
    }
    
    // 초기 접근인 경우(prevAllocatedSize == 0)에도 penalty 적용
    if (prevAllocatedSize == 0)
        penaltyOccurred = 1;
    
    // 여기서는 allocatedPageSize는 위에서 정한 finalAllocSize로 결정하고,
    // 재압축 결과가 이전보다 작더라도 allocatedPageSize는 변경하지 않고 totalUsed만 업데이트합니다.
    cp.allocatedPageSize = finalAllocSize;
    cp.targetSize = finalCandidate;
    cp.numExceptions = finalNumExceptions;
    cp.totalUsedSize = finalTotalUsed;
    cp.remainingCapacity = (finalAllocSize > finalTotalUsed) ? (finalAllocSize - finalTotalUsed) : 0;
    cp.penalty = penaltyOccurred;
    
    return cp;
}

#endif // __LCP_HH__