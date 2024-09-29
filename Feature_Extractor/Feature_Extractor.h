//
// Created by dell on 2023/8/11.
//

#ifndef BAMCLASS_FEATURE_EXTRACTOR_H
#define BAMCLASS_FEATURE_EXTRACTOR_H

#include <filesystem>
#include <torch/torch.h>
#include <set>
#include "../DataLoader/Pod5Data.h"
#include "../DataLoader/SamRead.h"
#include "../DataLoader/Reference_Reader.h"

namespace fs = std::filesystem;
namespace Yao {
    class Feature_Extractor {

    public:
        Feature_Extractor(fs::path pod5_dir_,
                          fs::path reference_path,
                          std::string ref_type);

        void extract_hc_sites(size_t num_workers,
                              size_t sub_thread_per_worker,
                              fs::path &bam_file,
                              fs::path &write_dir,
                              std::set<std::string> &pos_hc_sites,
                              std::set<std::string> &neg_hc_sites,
                              std::set<std::string> &motifset,
                              int32_t loc_in_motif,
                              int32_t kmer_size);

    private:
        fs::path pod5_dir;

        fs::path reference_path;
        std::vector<SamRead *> inputs;
        Yao::Reference_Reader ref;
        std::map<std::string, fs::path> filename_to_path;
        std::string file_name_hold;
        std::mutex mtx;
        std::condition_variable cv;

        int mapq_thresh_hold;
        float coverage_thresh_hold;
        float identity_thresh_hold;
    };

}

#endif //BAMCLASS_FEATURE_EXTRACTOR_H
