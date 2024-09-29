//
// Created by dell on 2023/8/17.
//
#include <torch/script.h>
#include <spdlog/spdlog.h>
#include "Call_Modification.h"
#include "../utils/utils_thread.h"
#include "../utils/utils_func.h"

bool isModuleEmpty(const torch::jit::script::Module &module) {
    // 检查模块是否包含任何参数或子模块
    return module.parameters().size() == 0 && module.children().size() == 0;
}


Yao::Call_Modification::Call_Modification(size_t batch_size_,
                                          size_t kmer_size_,
                                          size_t kmer_size_2,
                                          size_t kmer_size_3,
                                          fs::path reference_path_,
                                          std::string ref_type,
                                          fs::path module_dir) :
        reference_path(reference_path_),
        ref(std::move(Yao::Reference_Reader(reference_path, ref_type))),
        batch_size(batch_size_),
        kmer_size(kmer_size_),
        kmer_size_2(kmer_size_2),
        kmer_size_3(kmer_size_3) {

    fs::path module_path = "";
    fs::path module_path_2 = "";
    fs::path module_path_3 = "";

    for (const auto &entry: fs::directory_iterator(module_dir)) {
        if (entry.path().extension() == ".cpg") {
            module_path = entry.path();
        } else if (entry.path().extension() == ".chg") {
            module_path_2 = entry.path();
        } else if (entry.path().extension() == ".chh") {
            module_path_3 = entry.path();
        }
    }

    try {
        if (module_path == "") {
            spdlog::info("Cannot find cpg model!");
        } else {
            module = torch::jit::load(module_path);
        }
        if (module_path_2 == "") {
            spdlog::info("Cannot find chg model!");
        } else {
            module_2 = torch::jit::load(module_path_2);
        }
        if (module_path_3 == "") {
            spdlog::info("Cannot find chh model!");
        } else {
            module_3 = torch::jit::load(module_path_3);
        }
        if (module_path == "" && module_path_2 == "" && module_path_3 == "") {
            return;
        }
        spdlog::info("Successfully load module!");
    }
    catch (const c10::Error &e) {
        spdlog::error("Error loading the module");
    }
    try {
        at::Tensor kmer = torch::randint(0, 4, {512, static_cast<long>(kmer_size_)}, torch::kLong);
        at::Tensor signal = torch::rand({512, 1, static_cast<long>(kmer_size_), 19}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(kmer.to(torch::kCUDA));
        inputs.push_back(signal.to(torch::kCUDA));

        at::Tensor kmer_2 = torch::randint(0, 4, {512, static_cast<long>(kmer_size_2)}, torch::kLong);
        at::Tensor signal_2 = torch::rand({512, 1, static_cast<long>(kmer_size_2), 19}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs_2;
        inputs_2.push_back(kmer_2.to(torch::kCUDA));
        inputs_2.push_back(signal_2.to(torch::kCUDA));

        at::Tensor kmer_3 = torch::randint(0, 4, {512, static_cast<long>(kmer_size_3)}, torch::kLong);
        at::Tensor signal_3 = torch::rand({512, 1, static_cast<long>(kmer_size_3), 19}, torch::kFloat32);
        std::vector<torch::jit::IValue> inputs_3;
        inputs_3.push_back(kmer_3.to(torch::kCUDA));
        inputs_3.push_back(signal_3.to(torch::kCUDA));

        if (!isModuleEmpty(module)) {
            module.to(torch::kCUDA);
            auto result = module.forward(inputs);
        }
        if (!isModuleEmpty(module_2)) {
            module_2.to(torch::kCUDA);
            auto result_2 = module_2.forward(inputs_2);
        }
        if (!isModuleEmpty(module_3)) {
            module_3.to(torch::kCUDA);
            auto result_3 = module_3.forward(inputs_3);
        }
        spdlog::info("module test successfully!");
    }
    catch (const c10::Error &e) {
        spdlog::error("error loading the module");
        std::cout << e.what() << std::endl;
    }

    // default parameter to filter reads
    mapq_thresh_hold = 20;
    coverage_thresh_hold = 0.8;
    identity_thresh_hold = 0.8;
}

void Yao::Call_Modification::call_mods(fs::path &pod5_dir,
                                       fs::path &bam_path,
                                       fs::path &write_dir,
                                       size_t num_workers,
                                       size_t num_sub_thread,
                                       std::set<std::string> &motifset,
                                       std::set<std::string> &motifset_2,
                                       std::set<std::string> &motifset_3,
                                       size_t &loc_in_motif) {

    if (isModuleEmpty(module) && isModuleEmpty(module_2) && isModuleEmpty(module_3)) {
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();
    fs::path write_file = write_dir / "cpg_result.txt";
    fs::path write_file_2 = write_dir / "chg_result.txt";
    fs::path write_file_3 = write_dir / "chh_result.txt";

    std::thread threads[9];
    int i = 0;

    if (!isModuleEmpty(module)) {

        threads[i] = std::thread(Yao::get_feature_for_model_with_thread_pool,
                                 num_workers,
                                 num_sub_thread,
                                 std::ref(pod5_dir),
                                 std::ref(bam_path),
                                 std::ref(ref),
                                 std::ref(dataQueue),
                                 std::ref(mtx1),
                                 std::ref(cv1),
                                 batch_size,
                                 kmer_size,
                                 mapq_thresh_hold,
                                 coverage_thresh_hold,
                                 identity_thresh_hold,
                                 std::ref(motifset),
                                 loc_in_motif);
        threads[i + 1] = std::thread(Yao::Model_Inference,
                                     std::ref(module),
                                     std::ref(dataQueue),
                                     std::ref(site_key_Queue),
                                     std::ref(site_info_Queue),
                                     std::ref(pred_Queue),
                                     std::ref(p_rate_Queue),
                                     std::ref(mtx1),
                                     std::ref(cv1),
                                     std::ref(mtx2),
                                     std::ref(cv2),
                                     batch_size);
        threads[i + 2] = std::thread(Yao::count_modification_thread,
//                        std::ref(site_dict),
                                     std::ref(site_key_Queue),
                                     std::ref(site_info_Queue),
                                     std::ref(pred_Queue),
                                     std::ref(p_rate_Queue),
                                     std::ref(write_file),
                                     std::ref(mtx2),
                                     std::ref(cv2));
        i += 3;
    }
    if (!isModuleEmpty(module_2)) {

        threads[i] = std::thread(Yao::get_feature_for_model_with_thread_pool,
                                 num_workers,
                                 num_sub_thread,
                                 std::ref(pod5_dir),
                                 std::ref(bam_path),
                                 std::ref(ref),
                                 std::ref(dataQueue_2),
                                 std::ref(mtx1_2),
                                 std::ref(cv1_2),
                                 batch_size,
                                 kmer_size_2,
                                 mapq_thresh_hold,
                                 coverage_thresh_hold,
                                 identity_thresh_hold,
                                 std::ref(motifset_2),
                                 loc_in_motif);
        threads[i + 1] = std::thread(Yao::Model_Inference,
                                     std::ref(module_2),
                                     std::ref(dataQueue_2),
                                     std::ref(site_key_Queue_2),
                                     std::ref(site_info_Queue_2),
                                     std::ref(pred_Queue_2),
                                     std::ref(p_rate_Queue_2),
                                     std::ref(mtx1_2),
                                     std::ref(cv1_2),
                                     std::ref(mtx2_2),
                                     std::ref(cv2_2),
                                     batch_size);
        threads[i + 2] = std::thread(Yao::count_modification_thread,
                                     std::ref(site_key_Queue_2),
                                     std::ref(site_info_Queue_2),
                                     std::ref(pred_Queue_2),
                                     std::ref(p_rate_Queue_2),
                                     std::ref(write_file_2),
                                     std::ref(mtx2_2),
                                     std::ref(cv2_2));
        i += 3;
    }
    if (!isModuleEmpty(module_3)) {

        threads[i] = std::thread(Yao::get_feature_for_model_with_thread_pool,
                                 num_workers,
                                 num_sub_thread,
                                 std::ref(pod5_dir),
                                 std::ref(bam_path),
                                 std::ref(ref),
                                 std::ref(dataQueue_3),
                                 std::ref(mtx1_3),
                                 std::ref(cv1_3),
                                 batch_size,
                                 kmer_size_3,
                                 mapq_thresh_hold,
                                 coverage_thresh_hold,
                                 identity_thresh_hold,
                                 std::ref(motifset_3),
                                 loc_in_motif);
        threads[i + 1] = std::thread(Yao::Model_Inference,
                                     std::ref(module_3),
                                     std::ref(dataQueue_3),
                                     std::ref(site_key_Queue_3),
                                     std::ref(site_info_Queue_3),
                                     std::ref(pred_Queue_3),
                                     std::ref(p_rate_Queue_3),
                                     std::ref(mtx1_3),
                                     std::ref(cv1_3),
                                     std::ref(mtx2_3),
                                     std::ref(cv2_3),
                                     batch_size);
        threads[i + 2] = std::thread(Yao::count_modification_thread,
                                     std::ref(site_key_Queue_3),
                                     std::ref(site_info_Queue_3),
                                     std::ref(pred_Queue_3),
                                     std::ref(p_rate_Queue_3),
                                     std::ref(write_file_3),
                                     std::ref(mtx2_3),
                                     std::ref(cv2_3));
        i += 3;
    }

    for (int j = 0; j < i; ++j) {
        threads[j].join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    spdlog::info("Total time taken by extract feature and call modification: {} seconds", duration.count());
    spdlog::info("Write result finished");
}
