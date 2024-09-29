//
// Created by dell on 2023/8/10.
//

#ifndef BAMCLASS_UTILS_FUNC_H
#define BAMCLASS_UTILS_FUNC_H

#include <tuple>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <unordered_map>
#include <torch/torch.h>
#include <torch/torch.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <unordered_set>
#include <fstream>
#include <spdlog/spdlog.h>
#include "../3rdparty/htslib/include/sam.h"


namespace fs = std::filesystem;
namespace Yao {
    const std::map<char, std::int32_t> cigar_convert{
            {'M', 0},
            {'I', 1},
            {'D', 2},
            {'N', 3},
            {'S', 4},
            {'H', 5},
            {'P', 6},
            {'=', 7},
            {'X', 8}
    };

    const std::unordered_set<char> op_set{'M', 'I', 'D', 'N',
                                          'S', 'H', 'P', '=', 'X'};

    std::set<std::string> get_motif_set(std::string &motif_type);

    std::unordered_map<char, int> get_base2code_dna();

    std::unordered_map<char, int> get_base2code_rna();

    std::unordered_map<char, char> get_basepairs_dna();

    std::unordered_map<char, char> get_basepairs_rna();

    char alphabed(char letter, const std::map<char, char> &basepairs);

    void get_movetable(std::vector<uint64_t> &movetable, std::string &movetable_str);

    char fourbits2base(uint8_t val) ;

    std::string getSeq(const bam1_t *b);

    std::string getQual(const bam1_t *b);

    std::vector<std::pair<uint8_t , int32_t>> get_cigar_pair(const bam1_t *b);

    std::vector<std::pair<uint8_t, int32_t>> get_cigar_pair_from_str(const std::string &cigar_str);


    std::vector<uint64_t> get_movetable(const bam1_t *b, int32_t &stride);

    std::string get_aux_tag_str(const bam1_t *b, const char tag[2]);

    int32_t count_reference_length(std::vector<std::pair<uint8_t , int32_t>> &cigar_pair);

    void get_refloc_of_methysite_in_motif(std::vector<int32_t> &refloc_of_methysite,
                                          std::string &seqstr,
                                          std::set<std::string> &motifset,
                                          size_t loc_in_motif);

    std::vector<int32_t> group_signal_by_movetable(at::Tensor &signal_group,
                                   at::Tensor &normed_signal,
                                   std::vector<uint64_t> &movetable,
                                   size_t stride,
                                   int64_t signal_len=15);

    std::string get_complement_seq(std::string &ref_seq,
                                   std::string seq_type);

    void get_cigar_pair(std::vector<std::pair<int32_t, int32_t>> &cigar_pair,
                        const std::string &cigar_str);

    void parse_cigar(std::vector<std::pair<uint8_t, int32_t>> &cigar_pair,
                     std::vector<int32_t> &r_to_q_poss,
                     bool is_forward, int32_t ref_len);

//    int32_t count_reference_length(std::vector<std::pair<int32_t, int32_t>> &cigar_pair);

    float compute_pct_identity(std::vector<std::pair<uint8_t , int32_t>> &cigar_pair, int32_t nm);

    float compute_coverage(std::vector<std::pair<uint8_t, int32_t>> &cigar_pair, int32_t query_alignment_length);

    void parse_str_by_delimiter(std::vector<std::string> &tokens,
                                std::string &str,
                                std::string delimiter);

    at::Tensor get_query_qualities(std::string &query_qualities, bool is_forward);

    at::Tensor get_signals_rect(std::vector<at::Tensor> &k_signals,
                                int32_t signals_len);

    std::map<std::string, int32_t> get_chr_to_id(const fs::path &path);

    std::map<std::string, fs::path> get_filename_to_path(const fs::path &path);

    std::set<std::string> get_hc_set(const fs::path &path);

    std::vector<std::string> generate_keys(at::Tensor &site_info);

//std::string get_num_to_str(int32_t i);

}
#endif //BAMCLASS_UTILS_FUNC_H
