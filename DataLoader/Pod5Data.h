//
// Created by dell on 2023/8/9.
//

#ifndef BAMCLASS_POD5DATA_H
#define BAMCLASS_POD5DATA_H

#include <string>
#include <vector>
#include <filesystem>
#include <torch/torch.h>
#include <map>

namespace fs = std::filesystem;

namespace Yao {
    struct pod5read {
        std::string read_id;
        at::Tensor sample_data;
        float offset;
        float scale;
    };

    class Pod5Data {
    private:
        std::map<std::string, std::shared_ptr<pod5read>> id_to_read;
        int32_t read_count;
        fs::path data_path;
    public:
        Pod5Data(fs::path data_path_);

        at::Tensor get_normalized_signal_by_read_id(std::string & id);

        at::Tensor get_raw_signal_by_read_id(std::string id);

        std::tuple<float, float> get_offset_and_scale_by_read_id(std::string id);

        std::string get_filename();

        bool contain_read(std::string & id);

        void release();

        ~Pod5Data();
    };

//    void
//    get_features(Pod5Data p5, std::vector<Yao::SamRead *> inputs, fs::path writefile,
//                 std::set<std::string> &pos_hc_sites,
//                 std::set<std::string> &neg_hc_sites, int32_t kmer_size, int32_t mapq_thresh_hold,
//                 float coverage_thresh_hold, float identity_thresh_hole, std::string motifset, size_t loc_in_motif);
}
#endif //BAMCLASS_POD5DATA_H
