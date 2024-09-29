//
// Created by dell on 2023/8/11.
//

#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <spdlog/spdlog.h>
#include "Feature_Extractor.h"
#include "../utils/utils_func.h"
#include "../3rdparty/threadpool/threadpool.h"
#include "../utils/utils_thread.h"
#include "../DataLoader/Reference_Reader.h"

Yao::Feature_Extractor::Feature_Extractor(fs::path pod5_dir_,
                                          fs::path reference_path_,
                                          std::string ref_type) :
        pod5_dir(pod5_dir_),
        reference_path(reference_path_),
        ref(Yao::Reference_Reader(reference_path, ref_type)) {
    filename_to_path = Yao::get_filename_to_path(pod5_dir);
    file_name_hold = "";

    mapq_thresh_hold = 20;
    coverage_thresh_hold = 0.8;
    identity_thresh_hold = 0.8;
}


void Yao::Feature_Extractor::extract_hc_sites(size_t num_workers,
                                              size_t num_sub_thread,
                                              fs::path &bam_file,
                                              fs::path &write_dir,
                                              std::set<std::string> &pos_hc_sites,
                                              std::set<std::string> &neg_hc_sites,
                                              std::set<std::string> &motifset,
                                              int32_t loc_in_motif,
                                              int32_t kmer_size) {
    auto st = std::chrono::high_resolution_clock::now();
    auto reformat_chr = ref.reformat_chr();
    spdlog::info("num_workers: {}, num_sub_thread: {}", num_workers, num_sub_thread);

    samFile *bam_in = sam_open(bam_file.c_str(), "r");
    if (hts_set_threads(bam_in, 2)) {
        fprintf(stderr, "Error setting threads.\n");
        sam_close(bam_in);
    }
    bam_hdr_t *bam_header = sam_hdr_read(bam_in);
    bam1_t  *aln = NULL;
    aln = bam_init1();

    bool get_new_p5 = true;
    std::packaged_task<Yao::Pod5Data(const std::map<std::string, fs::path>&, std::string )> task;


    std::atomic<int64_t> thread_cnt(0);
    {

        ThreadPool pool(num_workers );
        int32_t file_cnt = 0;

        std::vector<std::shared_ptr<Yao::SamRead>> inputs;

        while (sam_read1(bam_in, bam_header, aln) >= 0) {
            if (get_new_p5 && file_name_hold != "") {
                std::function<Yao::Pod5Data(const std::map<std::string, fs::path>&, std::string )>
                        getP5 = [&](const std::map<std::string, fs::path>& filename_to_path, std::string file_name_hold)
                {
                    fs::path p5_file = filename_to_path.at(file_name_hold);
                    return Yao::Pod5Data(p5_file);
                };
                task = std::packaged_task<Yao::Pod5Data(const std::map<std::string, fs::path>&, std::string )>(getP5);

                std::thread thread(std::ref(task), std::ref(filename_to_path), file_name_hold);
                thread.join();
                get_new_p5 = false;
            }

            std::shared_ptr<Yao::SamRead> sam_ptr = std::make_shared<Yao::SamRead>(bam_in, bam_header, aln);

            if (file_name_hold != sam_ptr->file_name && !file_name_hold.empty()) {
                fs::path p5_file;
                get_new_p5 = true;
                try {
                    p5_file = filename_to_path.at(file_name_hold);
                    fs::path write_file = write_dir / p5_file.filename();
                    write_file.replace_extension(".npz");
                    auto p5 = task.get_future().get();
                    uint64_t sub_th_mut = (fs::file_size(p5_file) + 100 * 1024 * 1024 - 1) / (100 * 1024 * 1024);
                    while (thread_cnt >= (int64_t)(num_workers * num_sub_thread)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds (50));
                    }
                    pool.enqueue(Yao::get_hc_features,
                                 p5,
                                 inputs,
                                 std::ref(reformat_chr),
                                 write_file,
                                 std::ref(pos_hc_sites),
                                 std::ref(neg_hc_sites),
                                 kmer_size,
                                 mapq_thresh_hold,
                                 coverage_thresh_hold,
                                 identity_thresh_hold,
                                 std::ref(motifset),
                                 loc_in_motif,
                                 std::ref(thread_cnt),
                                 num_sub_thread * sub_th_mut);
                    inputs.clear();
                    file_cnt++;
                    spdlog::info("File {} enter to"\
                         " thread pool, progress [{}/{}], current work thread: {}",
                                 p5.get_filename(), file_cnt, filename_to_path.size(), thread_cnt);
                }
                catch (...) {
                    spdlog::error("Couldn't find file: {}", file_name_hold);
                    file_name_hold = sam_ptr->file_name;
                    inputs.clear();
                }
            }

            file_name_hold = sam_ptr->file_name;
            if (!sam_ptr->is_mapped) continue;
            int32_t st = sam_ptr->reference_start;
            int32_t ed = sam_ptr->reference_end;
            std::string chr_key = sam_ptr->reference_name;
            sam_ptr->reference_seq = ref.get_reference_seq(chr_key, sam_ptr->is_forward, st, ed);
            if (sam_ptr->reference_seq.length() == 0) continue;
            if (sam_ptr->query_sequence.empty()) continue;
            inputs.push_back(sam_ptr);
        }
        if (inputs.size() > 0) {
            fs::path p5_file;
            try {
                p5_file = filename_to_path.at(file_name_hold);
                fs::path write_file = write_dir / p5_file.filename();
                write_file.replace_extension(".npz");
                auto p5 = task.get_future().get();
                pool.enqueue(Yao::get_hc_features,
                             p5,
                             inputs,
                             std::ref(reformat_chr),
                             write_file,
                             std::ref(pos_hc_sites),
                             std::ref(neg_hc_sites),
                             kmer_size,
                             mapq_thresh_hold,
                             coverage_thresh_hold,
                             identity_thresh_hold,
                             std::ref(motifset),
                             loc_in_motif,
                             std::ref(thread_cnt),
                             num_sub_thread);
                inputs.clear();
                file_cnt++;
                spdlog::info("File {} enter to"\
                         " thread pool, progress [{}/{}], current work thread: {}",
                             p5.get_filename(), file_cnt, filename_to_path.size(), thread_cnt);
            }
            catch (...) {
                spdlog::error("Couldn't find file: {}", file_name_hold);
                inputs.clear();
            }
        }
    }
    auto ed = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(ed - st);
    spdlog::info("extract hc features finished, cost {} seconds", duration.count());
}
