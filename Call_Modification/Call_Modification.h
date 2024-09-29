//
// Created by dell on 2023/8/17.
//

#ifndef BAMCLASS_CALL_MODIFICATION_H
#define BAMCLASS_CALL_MODIFICATION_H

#include <torch/torch.h>
#include "../DataLoader/Reference_Reader.h"

namespace Yao {
    class Call_Modification {
    public:
        Call_Modification(size_t batch_size_,
                          size_t kmer_size_,
                          size_t kmer_size_2,
                          size_t kmer_size_3,
                          fs::path reference_path_,
                          std::string ref_type,
                          fs::path module_dir);

        void call_mods(fs::path &pod5_dir,
                       fs::path &bam_path,
                       fs::path &write_dir,
                       size_t num_workers,
                       size_t num_sub_thread,
                       std::set<std::string> &motifset,
                       std::set<std::string> &motifset_2,
                       std::set<std::string> &motifset_3,
                       size_t &loc_in_motif);

    private:
        std::queue<std::tuple<std::vector<std::string>, std::vector<std::string>, at::Tensor, at::Tensor>> dataQueue;
        std::queue<std::vector<std::string>> site_key_Queue;
        std::queue<std::vector<std::string>> site_info_Queue;
        std::queue<std::vector<int64_t>> pred_Queue;
        std::queue<std::vector<float>> p_rate_Queue;
        std::mutex mtx1;
        std::mutex mtx2;
        std::condition_variable cv1;
        std::condition_variable cv2;
        torch::jit::script::Module module;
        size_t kmer_size;

        std::queue<std::tuple<std::vector<std::string>, std::vector<std::string>, at::Tensor, at::Tensor>> dataQueue_2;
        std::queue<std::vector<std::string>> site_key_Queue_2;
        std::queue<std::vector<std::string>> site_info_Queue_2;
        std::queue<std::vector<int64_t>> pred_Queue_2;
        std::queue<std::vector<float>> p_rate_Queue_2;
        std::mutex mtx1_2;
        std::mutex mtx2_2;
        std::condition_variable cv1_2;
        std::condition_variable cv2_2;
        torch::jit::script::Module module_2;
        size_t kmer_size_2;

        std::queue<std::tuple<std::vector<std::string>, std::vector<std::string>, at::Tensor, at::Tensor>> dataQueue_3;
        std::queue<std::vector<std::string>> site_key_Queue_3;
        std::queue<std::vector<std::string>> site_info_Queue_3;
        std::queue<std::vector<int64_t>> pred_Queue_3;
        std::queue<std::vector<float>> p_rate_Queue_3;
        std::mutex mtx1_3;
        std::mutex mtx2_3;
        std::condition_variable cv1_3;
        std::condition_variable cv2_3;
        torch::jit::script::Module module_3;
        size_t kmer_size_3;


//        std::map<std::string, std::vector<float>> site_dict;
        size_t batch_size;
        fs::path reference_path;
        Yao::Reference_Reader ref;
        std::string file_name_hold;
        int32_t mapq_thresh_hold;
        float coverage_thresh_hold;
        float identity_thresh_hold;
    };

}


#endif //BAMCLASS_CALL_MODIFICATION_H
