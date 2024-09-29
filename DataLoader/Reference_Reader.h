//
// Created by dell on 2023/8/11.
//

#ifndef BAMCLASS_REFERENCE_READER_H
#define BAMCLASS_REFERENCE_READER_H

#include <map>
#include <unordered_map>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
namespace Yao {
    class Reference_Reader {
    public:
        Reference_Reader(fs::path reference_path_, std::string seq_type_);

        Reference_Reader();

        std::string get_reference_seq(std::string chr_key,
                                      bool is_forward,
                                      int64_t start,
                                      int64_t end);

        void display() {
            for (const auto &[key, value]: chr_to_ref) {
                std::cout << key << " " << value.length() << std::endl;
            }
        }

        std::map<std::string, std::string> reformat_chr() {
            size_t len = 0;
            std::map<std::string, std::string> map;
            for (const auto &key: chr_set) {
                len = std::max(len, key.length());
            }
            for (auto key: chr_set) {
                std::string value = key;
                if (value.length() < len) {
                    size_t l = value.length();
                    for (size_t i = 0; i < len - l; i++) {
                        value = "_" + value;
                    }
                }
                map[key] = value;
            }
            return map;
        }

    private:
        fs::path reference_path;
        std::map<std::string, std::string> chr_to_ref;
        std::set<std::string> chr_set;
        std::string seq_type;
    };
}
#endif //BAMCLASS_REFERENCE_READER_H
