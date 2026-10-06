#include "Compressor.h"
// Compressor 클래스 메서드 구현
void Compressor::compressFile(FILE *fd) {
    CACHELINE_DATA line;

    reset();

    while (true) {
        if (fread(&line, _MAX_BYTES_PER_LINE, 1, fd) != 1) {
            break;
        }
        compressLine(&line, 0);
    }
}

void Compressor::countPattern(INT64 pattern) {
    totalPatternCnt++;
    auto it = patternCounterMap.find(pattern);
    if (it == patternCounterMap.end()) {
        patternCounterMap.insert(std::pair<INT64, CNT>(pattern, 1ull));
    } else {
        it->second++;
    }
}

void Compressor::countLineResult(LENGTH length) {
    totalLineCnt++;
    auto it = lengthMap.find(length);
    if (it == lengthMap.end()) {
        lengthMap.insert(std::pair<LENGTH, CNT>(length, 1ull));
    } else {
        it->second++;
    }
}

double Compressor::getCoverage(int thresholdBitSize) {
    CNT accumLineCnt = 0ull;
    for (int i = 0; i <= thresholdBitSize; i++) {
        auto it = lengthMap.find(i);
        if (it != lengthMap.end()) {
            accumLineCnt += it->second;
        }
    }
    return accumLineCnt * 1. / totalLineCnt;
}

void Compressor::printSummary(FILE* fd) {
    CNT accumCnt;

    fprintf(fd, "Comp\t%s\n", name.c_str());
    fprintf(fd, "dataCnt32\t%lld\n", totalLineCnt * _MAX_DWORDS_PER_LINE);

    accumCnt = 0ull;
    for (auto it = lengthMap.cbegin(); it != lengthMap.cend(); ++it) {
        accumCnt += (it->first * it->second);
    }
    fprintf(fd, "CompRatio  \t%f\n", totalLineCnt * _MAX_DWORDS_PER_LINE * 32. / accumCnt);
    fprintf(fd, "Comp32b    \t%f\n", accumCnt * 1. / totalLineCnt / _MAX_DWORDS_PER_LINE);
    fprintf(fd, "Comp8b     \t%f\n", accumCnt * 1. / totalLineCnt / _MAX_BYTES_PER_LINE);
    fprintf(fd, "CompDist   \t");
    accumCnt = 0ull;
    for (int i = 0; i < 600; i++) {
        auto it = lengthMap.find(i);
        if (it != lengthMap.end()) {
            accumCnt += it->second;
        }
        if (i % 8 == 0) {
            fprintf(fd, "%f\t", accumCnt * 1. / totalLineCnt);
        }
    }
    fprintf(fd, "\n");
}

void Compressor::printDetails(FILE* fd, std::string bench_name) const {
    fprintf(fd, "Pattern frequency\n");
    if (patternCounterMap.size() > (1 << 16)) {
        std::map<INT64, CNT> patternGroupCounterMap;
        int offset = 48;
        for (auto it = patternCounterMap.cbegin(); it != patternCounterMap.cend(); ++it) {
            INT64 patternGroup = it->first >> offset;
            auto it2 = patternGroupCounterMap.find(patternGroup);
            if (it2 == patternGroupCounterMap.end()) {
                patternGroupCounterMap.insert(std::pair<INT64, CNT>(patternGroup, it->second));
            } else {
                it2->second += it->second;
            }
        }
        std::vector<std::pair<INT64, CNT>> v;
        for (auto it = patternGroupCounterMap.cbegin(); it != patternGroupCounterMap.cend(); ++it) {
            v.push_back(std::pair<INT64, CNT>(it->first, it->second));
        }
        std::sort(v.begin(), v.end());
        for (auto it = v.cbegin(); it != v.cend(); ++it) {
            fprintf(fd, "%16lx\t%f\t%f\t%lld\n", (UINT64)it->first, it->second * 1. / totalPatternCnt, -log2(it->second * 1. / totalPatternCnt), it->second);
        }
    } else {
        std::vector<std::pair<INT64, CNT>> v;
        for (auto it = patternCounterMap.cbegin(); it != patternCounterMap.cend(); ++it) {
            v.push_back(std::pair<INT64, CNT>(it->first, it->second));
        }
        std::sort(v.begin(), v.end());
        for (auto it = v.cbegin(); it != v.cend(); ++it) {
            fprintf(fd, "%16lx\t%f\t%f\t%lld\n", (UINT64)it->first, it->second * 1. / totalPatternCnt, -log2(it->second * 1. / totalPatternCnt), it->second);
        }
    }

    fprintf(fd, "Compressed line size\n");
    for (auto it = lengthMap.cbegin(); it != lengthMap.cend(); it++) {
        fprintf(fd, "%02d\t%f\t%f\n", it->first, it->second * 1. / totalLineCnt, -log2(it->second * 1. / totalLineCnt));
    }
}

// BPCompressor 클래스 메서드 구현 128B압축
unsigned BPCompressor::compressLine(CACHELINE_DATA* line, UINT64 line_addr) {
    INT64 deltas[31];
    for (unsigned int i = 1; i < _MAX_DWORDS_PER_LINE; i++) {
        deltas[i - 1] = ((INT64)line->s_dword[i]) - ((INT64)line->s_dword[i - 1]);
    }

    INT32 prevDBP;
    INT32 DBP[33];
    INT32 DBX[33];
    for (int j = 63; j >= 0; j--) {
        INT32 buf = 0;
        for (int i = 30; i >= 0; i--) {
            buf <<= 1;
            buf |= ((deltas[i] >> j) & 1);
        }
        if (j == 63) {
            DBP[32] = buf;
            DBX[32] = buf;
            prevDBP = buf;
        } else if (j < 32) {
            DBP[j] = buf;
            DBX[j] = buf ^ prevDBP;
            prevDBP = buf;
        } else {
            assert(buf == prevDBP);
            prevDBP = buf;
        }
    }

    unsigned blkLength = encodeFirst(line->dword[0]);
    blkLength += encodeDeltas(DBP, DBX);

    countLineResult(blkLength);
    return blkLength / 8; // Return size in bytes
}

unsigned BPCompressor::encodeFirst(INT32 sym) {
    if (sym == 0) {
        countPattern(256);
        return 3;
    } else if (sign_extended(sym, 4)) {
        countPattern(257);
        return (3 + 4);
    } else if (sign_extended(sym, 8)) {
        countPattern(258);
        return (3 + 8);
    } else if (sign_extended(sym, 16)) {
        countPattern(259);
        return (3 + 16);
    } else {
        countPattern(263);
        return (1 + 32);
    }
}

unsigned BPCompressor::encodeDeltas(INT32* DBP, INT32* DBX) {
    static const unsigned ZRL_CODE_SIZE[34] = {0, 3, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
    static const unsigned singleOneSize = 10;
    static const unsigned consecutiveDoubleOneSize = 10;
    static const unsigned allOneSize = 5;
    static const unsigned zeroDBPSize = 5;

    unsigned length = 0;
    unsigned run_length = 0;
    for (int i = 32; i >= 0; i--) {
        if (DBX[i] == 0) {
            run_length++;
        } else {
            if (run_length > 0) {
                countPattern(run_length - 1);
                assert(run_length != 33);
                length += ZRL_CODE_SIZE[run_length];
            }
            run_length = 0;

            if (DBP[i] == 0) {
                length += zeroDBPSize;
                countPattern(34);
            } else if (DBX[i] == 0x7fffffff) {
                length += allOneSize;
                countPattern(35);
            } else {
                int oneCnt = 0;
                for (int j = 0; j < 32; j++) {
                    if ((DBX[i] >> j) & 1) {
                        oneCnt++;
                    }
                }
                unsigned two_distance = 0;
                int firstPos = -1;
                if (oneCnt <= 2) {
                    for (int j = 0; j < 32; j++) {
                        if ((DBX[i] >> j) & 1) {
                            if (firstPos == -1) {
                                firstPos = j;
                            } else {
                                two_distance = j - firstPos;
                            }
                        }
                    }
                }
                if (oneCnt == 1) {
                    length += singleOneSize;
                    countPattern(37 + firstPos);
                } else if ((oneCnt == 2) && (two_distance == 1)) {
                    length += consecutiveDoubleOneSize;
                    countPattern(69 + firstPos);
                } else {
                    length += 32;
                    countPattern(36);
                }
            }
        }
    }
    if (run_length > 0) {
        length += ZRL_CODE_SIZE[run_length];
        countPattern(run_length - 1);
        assert(run_length <= 33);
    }
    return length;
}

// BDI 압축 관련 함수 구현
bool sign_extended(UINT64 value, UINT8 bit_size) {
    UINT64 max = (1ULL << (bit_size - 1)) - 1;
    UINT64 min = ~max;
    return (value <= max) | (value >= min);
}

bool zero_extended(UINT64 value, UINT8 bit_size) {
    UINT64 max = (1ULL << bit_size) - 1;
    return (value <= max);
}

unsigned long long my_llabs(long long x) {
    unsigned long long t = x >> 63;
    return (x ^ t) - t;
}

unsigned my_abs(int x) {
    unsigned t = x >> 31;
    return (x ^ t) - t;
}

long long unsigned* convertBuffer2Array(char* buffer, unsigned size, unsigned step) {
    long long unsigned* values = (long long unsigned*)malloc(sizeof(long long unsigned) * (size / step));
    unsigned int i, j;
    for (i = 0; i < size / step; i++) {
        values[i] = 0;
    }
    for (i = 0; i < size; i += step) {
        for (j = 0; j < step; j++) {
            values[i / step] += ((long long unsigned)((unsigned char)buffer[i + j])) << (8 * j);
        }
    }
    return values;
}

int isZeroPackable(long long unsigned* values, unsigned size) {
    for (unsigned i = 0; i < size; i++) {
        if (values[i] != 0)
            return 0;
    }
    return 1;
}

int isSameValuePackable(long long unsigned* values, unsigned size) {
    for (unsigned i = 0; i < size; i++) {
        if (values[0] != values[i])
            return 0;
    }
    return 1;
}

unsigned multBaseCompression(long long unsigned* values, unsigned size, unsigned blimit, unsigned bsize) {
    unsigned long long limit = 0;
    unsigned BASES = 2;
    switch (blimit) {
        case 1: limit = 0xFF; break;
        case 2: limit = 0xFFFF; break;
        case 4: limit = 0xFFFFFFFF; break;
        default: exit(1);
    }
    unsigned long long mbases[64];
    unsigned baseCount = 1;
    mbases[0] = 0;
    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = 0; j < baseCount; j++) {
            if (my_llabs((long long)(mbases[j] - values[i])) > limit) {
                mbases[baseCount++] = values[i];
            }
        }
        if (baseCount >= BASES)
            break;
    }
    unsigned compCount = 0;
    for (unsigned i = 0; i < size; i++) {
        for (unsigned j = 0; j < baseCount; j++) {
            if (my_llabs((long long)(mbases[j] - values[i])) <= limit) {
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

unsigned BDICompress(char* buffer, unsigned _blockSize) {
    long long unsigned* values = convertBuffer2Array(buffer, _blockSize, 8);
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

unsigned getAllocatedPageSize(unsigned totalUsed) {
    if (totalUsed > 4096) return 4096;
    for (unsigned i = 0; i < numAllowed; i++) {
        if (totalUsed <= allowedPageSizes[i])
            return allowedPageSizes[i];
    }
    return 4096; // Fallback to uncompressed size
}
size_t CUREMEM_compress(Compressor* comp, CACHELINE_DATA* pageData){//CURE-MEM으로 압축된 페이지 사이즈 반환
    size_t cur_comp_size =0;
    for(int i =0; i < NUM_LINES_PER_PAGE; i++){//페이지는 128B가 32개 존재
        cur_comp_size += comp->compressLine(&pageData[i], i * UNCOMPRESSED_SIZE);
    }
    return cur_comp_size;
}  
LCP_CompressedPage compressPageLCP(Compressor* comp, void* pageData, 
                                   unsigned prevAllocatedSize, unsigned long long pageNum,
                                   int updatedLine, LCP_CompressedPage* prevCP) {//updatedLine은 -1이면 전체
    LCP_CompressedPage cp;
    cp.pageNumber = pageNum;

    // 배열 초기화
    for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
        cp.exceptionLines[i] = false;
        cp.cacheSizes[i] = UNCOMPRESSED_SIZE; // 기본값: 비압축
    }
    cp.numExceptions = 0;
    cp.maxExceptions = 0;
    cp.targetSize = UNCOMPRESSED_SIZE;
    cp.allocatedPageSize = 4096;
    cp.totalUsedSize = 4096;
    cp.remainingCapacity = 0;
    cp.penalty = 1;

    // 캐시 라인 압축 크기 계산
    if (prevCP == nullptr || updatedLine <= 0) {
        // 전체 페이지 압축
        for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
            cp.cacheSizes[i] = comp->compressLine((CACHELINE_DATA*)((char*)pageData + i * UNCOMPRESSED_SIZE), i * UNCOMPRESSED_SIZE);
        }
    } else {
        // 이전 페이지 복사 및 업데이트된 라인만 재압축
        memcpy(&cp, prevCP, sizeof(LCP_CompressedPage));
        if (updatedLine >= 0 && static_cast<unsigned>(updatedLine) < NUM_LINES_PER_PAGE) {
            cp.cacheSizes[updatedLine] = comp->compressLine((CACHELINE_DATA*)((char*)pageData + updatedLine * UNCOMPRESSED_SIZE), updatedLine * UNCOMPRESSED_SIZE);
        }
    }

    // 대상 압축 크기 후보 설정
    std::vector<unsigned> candidateArray;
    if (comp->getName() == "BPC") {
        candidateArray = {8,16,32,64};
        /*unsigned maxCompressedSize = 0;
        for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
            if (cp.cacheSizes[i] < UNCOMPRESSED_SIZE && cp.cacheSizes[i] > maxCompressedSize) {
                maxCompressedSize = cp.cacheSizes[i];
            }
        }
        if (maxCompressedSize > 0) {
            candidateArray.push_back(maxCompressedSize);
        }*/
    } else {
        // BDI 후보 (가정: 문서에 명시되지 않음)
        candidateArray = {4, 8, 16, 32};
    }
    /*if (prevCP != nullptr && updatedLine >= 0) {
        candidateArray.push_back(prevCP->targetSize);
    }*/
    //std::sort(candidateArray.begin(), candidateArray.end());

    // 후보 평가
    struct CandidateInfo {
        unsigned candidate; // 대상 압축 크기
        unsigned numExceptions; // 예외 개수
        unsigned totalUsed; // 총 사용 크기
        unsigned allocatedSize; // 할당 페이지 크기
    };
    std::vector<CandidateInfo> candidates;

    for (unsigned c : candidateArray) {
        unsigned numExceptions = 0;
        for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
            if (cp.cacheSizes[i] > c) {
                numExceptions++;
            }
        }

        // Compressed Data Region: 모든 캐시 라인에 대해 슬롯 준비
        unsigned compressedDataSize = NUM_LINES_PER_PAGE * c;
        // Exception Storage Region: 예외 캐시 라인 (비압축)
        unsigned exceptionStorageSize = numExceptions * UNCOMPRESSED_SIZE;
        // 메타데이터
        unsigned metadataSize = BASE_METADATA_OVERHEAD;
        // 총 사용 크기
        unsigned totalUsed = compressedDataSize + exceptionStorageSize + metadataSize;
        // 할당 페이지 크기
        unsigned allocSize = getAllocatedPageSize(totalUsed);

        if (allocSize != 0 && totalUsed <= 4096) {
            candidates.push_back({c, numExceptions, totalUsed, allocSize});
        }
    }

    // 최적 후보 선택
    CandidateInfo bestCandidate = {UNCOMPRESSED_SIZE, NUM_LINES_PER_PAGE, 4096, 4096};
    bool group1Found = false;

    for (const auto& cand : candidates) {
        if (prevAllocatedSize != 0 && cand.allocatedSize <= prevAllocatedSize) {
            if (!group1Found || cand.totalUsed < bestCandidate.totalUsed ||
                (cand.totalUsed == bestCandidate.totalUsed && cand.numExceptions < bestCandidate.numExceptions)) {
                bestCandidate = cand;
                group1Found = true;
            }
        }
    }

    if (!group1Found && !candidates.empty()) {
        bestCandidate = candidates[0];
        for (const auto& cand : candidates) {
            if (cand.totalUsed < bestCandidate.totalUsed) {
                bestCandidate = cand;
            }
        }
    }

    // 최종 값 설정
    cp.targetSize = bestCandidate.candidate;
    cp.numExceptions = bestCandidate.numExceptions;
    cp.maxExceptions = (bestCandidate.allocatedSize-bestCandidate.candidate*NUM_LINES_PER_PAGE-BASE_METADATA_OVERHEAD)/UNCOMPRESSED_SIZE;
    cp.totalUsedSize = bestCandidate.totalUsed;
    cp.allocatedPageSize = bestCandidate.allocatedSize;
    cp.remainingCapacity = bestCandidate.allocatedSize - bestCandidate.totalUsed;
    cp.penalty = (prevAllocatedSize == 0 || bestCandidate.allocatedSize > prevAllocatedSize || 
                  candidates.empty()) ? 1 : 0;

    // 예외 슬롯 할당

    for (unsigned i = 0; i < NUM_LINES_PER_PAGE; i++) {
        cp.exceptionLines[i] = (cp.cacheSizes[i] > cp.targetSize);
    }

    return cp;
}

LCP_CompressedPage updateCompressedCacheLine(
    Compressor* comp,
    CACHELINE_DATA* pageData,
    const LCP_CompressedPage& prevState,
    unsigned cacheLineIndex) {
    if (cacheLineIndex >= NUM_LINES_PER_PAGE) {
        fprintf(stderr, "Error: Cache line index (%u) out of range (0-%u).\n",
                cacheLineIndex, NUM_LINES_PER_PAGE - 1);
        return prevState;
    }

    LCP_CompressedPage newState = prevState;
    
    // 해당 캐시 라인만 압축
    unsigned newSize = comp->compressLine((CACHELINE_DATA*)((char*)pageData + cacheLineIndex * UNCOMPRESSED_SIZE), 0);
    
    newState.cacheSizes[cacheLineIndex] = newSize;

    // 이전 상태에서 해당 라인이 예외였는지 확인
    bool wasException = prevState.exceptionLines[cacheLineIndex];
    bool isException = (newSize > prevState.targetSize);

    // 예외 상태가 변경된 경우에만 처리
    if (wasException != isException) {
        if (isException) {
            // 새로운 예외가 필요한 경우
            if (prevState.numExceptions < prevState.maxExceptions) {
                // 사용 가능한 예외 슬롯이 있는 경우
                newState.exceptionLines[cacheLineIndex] = true;
                newState.numExceptions++;
                
                // 총 사용 크기 업데이트
                newState.totalUsedSize = prevState.totalUsedSize - prevState.targetSize + UNCOMPRESSED_SIZE;
                newState.remainingCapacity = newState.allocatedPageSize - newState.totalUsedSize;
            } else {
                // 예외 슬롯이 부족한 경우
                newState.penalty = 1;
            }
        } else {
            // 예외가 제거된 경우
            newState.exceptionLines[cacheLineIndex] = false;
            newState.numExceptions--;
            
            // 총 사용 크기 업데이트
            newState.totalUsedSize = prevState.totalUsedSize - UNCOMPRESSED_SIZE + prevState.targetSize;
        }
    }

    return newState;
}