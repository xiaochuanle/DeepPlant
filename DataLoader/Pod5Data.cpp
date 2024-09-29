//
// Created by dell on 2023/8/9.
//
#include <iostream>
#include <boost/uuid/uuid_io.hpp>
#include <torch/torch.h>
#include <spdlog/spdlog.h>
#include "Pod5Data.h"
#include "pod5_format/c_api.h"

Yao::Pod5Data::Pod5Data(fs::path data_path_) : data_path(std::move(data_path_)) {
    // Initialize the POD5 library
    pod5_init();
    // open the file ready for walking;
    Pod5FileReader_t *file = pod5_open_file(data_path.c_str());
    if (!file)
        std::cerr << "Failed to open file "
                  << data_path << ": " << pod5_get_error_string() << "\n";

    size_t batch_count = 0;
    if (pod5_get_read_batch_count(&batch_count, file) != POD5_OK) {
        std::cerr << "Failed to query batch count: " << pod5_get_error_string() << "\n";
    }


    size_t read_count_ = 0;
    for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {

        Pod5ReadRecordBatch_t *batch = nullptr;
        if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
            std::cerr << "Failed to get batch: " << pod5_get_error_string() << "\n";
        }

        std::size_t batch_row_count = 0;
        if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
            std::cerr << "Failed to get batch row count\n";
        }

        for (std::size_t row = 0; row < batch_row_count; ++row) {
            uint16_t read_table_version = 0;
            ReadBatchRowInfo_t read_data;
            if (pod5_get_read_batch_row_info_data(
                    batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data, &read_table_version)
                != POD5_OK) {
                std::cerr << "Failed to get read " << row << "\n";
            }

            read_count_ += 1;

            std::size_t sample_count = 0;
            pod5_get_read_complete_sample_count(file, batch, row, &sample_count);

            std::vector<std::int16_t> samples;
            samples.resize(sample_count);
            pod5_get_read_complete_signal(file, batch, row, samples.size(), samples.data());
//            pod5read *read = new pod5read;
            std::shared_ptr<pod5read> read = std::make_shared<pod5read>();
            std::vector<float> signal(samples.begin(), samples.end());
            read->sample_data = torch::from_blob(signal.data(), signal.size(), torch::kFloat32).clone();
            read->offset = read_data.calibration_offset;
            read->scale = read_data.calibration_scale;
            // get read_id str
            auto uuid_data = reinterpret_cast<boost::uuids::uuid const *>(read_data.read_id);
            std::string str_read_id = boost::uuids::to_string(*uuid_data);
            read->read_id = str_read_id;
            if (read->read_id.size() != 36) {
                std::cerr << "Unexpected length of UUID\n";
            }
            id_to_read[str_read_id] = read;
        }
        if (pod5_free_read_batch(batch) != POD5_OK) {
            std::cerr << "Failed to release batch\n";
        }
    }
    read_count = read_count_;
    // Close the reader
    if (pod5_close_and_free_reader(file) != POD5_OK) {
        std::cerr << "Failed to close reader: " << pod5_get_error_string() << "\n";
    }

    // Cleanup the library
    pod5_terminate();
}

at::Tensor Yao::Pod5Data::get_normalized_signal_by_read_id(std::string & id) {
//    if (id == "d420b6fb-3375-4476-8a50-9fceecfbefd9") {
//        spdlog::info("find it");
//    }

    if (id_to_read.find(id) == id_to_read.end()) {
        spdlog::error("cant find read id in current pod5 file - {}", this->get_filename());
        return at::Tensor();
    }

    at::Tensor signal = id_to_read[id]->sample_data;
    // scale back signal to orignal
    float offset = id_to_read[id]->offset;
    float scale = id_to_read[id]->scale;
    signal = scale * (signal + offset);
    // normalize signal, method = mad
    auto center = torch::median(signal);
    float c = 0.674489;
    auto mad = torch::median(torch::abs(signal - center)) / c;
    if (mad.item<float>() != 0)
        signal = (signal - torch::median(signal)) / mad;
    return signal.to(torch::kFloat32);
}

at::Tensor Yao::Pod5Data::get_raw_signal_by_read_id(std::string id) {
    return id_to_read[id]->sample_data.clone();
}

std::tuple<float, float> Yao::Pod5Data::get_offset_and_scale_by_read_id(std::string id) {
    return std::tuple<float, float>(id_to_read[id]->offset, id_to_read[id]->scale);
}

std::string Yao::Pod5Data::get_filename() {
    return data_path.filename();
}

bool Yao::Pod5Data::contain_read(std::string &id) {
    return id_to_read.find(id) != id_to_read.end();
}

void Yao::Pod5Data::release() {
//    for (auto &[key, ptr]: id_to_read) {
//        delete ptr;
//        ptr = nullptr;
//    }
//    id_to_read.clear();
}

Yao::Pod5Data::~Pod5Data() {

}




