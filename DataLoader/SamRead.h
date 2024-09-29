//
// Created by dell on 2023/8/7.
//

#ifndef BAMCLASS_SAMREAD_H
#define BAMCLASS_SAMREAD_H
#pragma once

#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>
#include "../3rdparty/htslib/include/sam.h"

namespace Yao {


    struct SamRead {

        SamRead() = delete;

        SamRead(samFile* bam_in, bam_hdr_t* bam_header, bam1_t *aln);
        SamRead(std::string sam_str);
        void print() {
            for (const auto &[a, b]: cigar_pair) {
                std::cout << "(" << a << ", " << b << ") ";
            }
            std::cout << std::endl;
        }

        bool is_forward;
        bool is_mapped;
        bool is_split;

        // Basuc alignment section
        std::string query_name;
        int32_t flag;
        std::string reference_name;
        int32_t mapping_quality;
        int32_t reference_start;
        int32_t reference_end;
        int32_t reference_length;
        int32_t query_alignment_start;
        int32_t query_alignment_end;
        int32_t query_alignment_length;
        std::string cigar_string;
        std::vector<std::pair<uint8_t , int32_t>> cigar_pair;
        std::string rnext;
        int32_t pnext;
        int64_t tlen;
        std::string query_sequence;
        std::string query_qualities;
        std::string reference_seq;

        // tags
        std::vector<uint64_t> movetable;
        int32_t stride;
        std::string MD_str;
        int32_t NM;
        std::string file_name;
        std::string parent_read;
        int32_t split;
        int num_samples;
        int32_t trimed_start;
    };

}
#endif //BAMCLASS_SAMREAD_H
