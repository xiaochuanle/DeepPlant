//
// Created by dell on 2023/8/11.
//

#include <fstream>
#include <boost/algorithm/string.hpp>
#include "Reference_Reader.h"
#include "../utils/utils_func.h"

Yao::Reference_Reader::Reference_Reader(fs::path reference_path_, std::string seq_type_) :
        reference_path(reference_path_),
        seq_type(seq_type_) {
    if (!(seq_type == "DNA" || seq_type == "RNA")) {
        std::cerr << "Getting wrong ref_seq type\n";
    }
    std::ifstream ref_file(reference_path);
    std::string line, chr_key, ref_seq;
    while (std::getline(ref_file, line)) {
        if (line[0] == '>') {
            if (chr_key.length() != 0 && ref_seq.length() != 0) {
                chr_to_ref[chr_key] = ref_seq;
                chr_set.insert(chr_key);
                ref_seq.clear();
            }
            chr_key = line.substr(1, line.find(' ') - 1);
        } else
            // turn to upper case
            ref_seq += boost::to_upper_copy(line);
    }
    chr_to_ref[chr_key] = ref_seq;
    chr_set.insert(chr_key);
}

std::string
Yao::Reference_Reader::get_reference_seq(std::string chr_key,
                                         bool is_forward,
                                         int64_t start,
                                         int64_t end) {
    if (chr_set.find(chr_key) != chr_set.end())
        return chr_to_ref[chr_key].substr(start, end - start);
    else
        return "";
}
