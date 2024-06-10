#pragma once

#include <string>
#include <memory>
#include <dlfcn.h>
#include <iostream>
#include "deepspeed_aio_base.h"

class Trampoline;

class handle {
public:
    explicit handle(std::shared_ptr<Trampoline> trampoline) : trampoline_(trampoline) {}
    // void do_something_inner();
    int get_block_size() const;
    int get_queue_depth() const;
    bool get_single_submit() const;
    bool get_overlap_events() const;
    int get_thread_count() const;

    void read(torch::Tensor& buffer, const char* filename, const bool validate);
    void write(const torch::Tensor& buffer, const char* filename, const bool validate);
    void pread(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async);
    void pwrite(const torch::Tensor& buffer, const char* filename, const bool validate, const bool async);

    void sync_pread(torch::Tensor& buffer, const char* filename);
    void sync_pwrite(const torch::Tensor& buffer, const char* filename);
    void async_pread(torch::Tensor& buffer, const char* filename);
    void async_pwrite(const torch::Tensor& buffer, const char* filename);

    void new_cpu_locked_tensor(const size_t num_elem, const torch::Tensor& example_tensor);
    void free_cpu_locked_tensor(torch::Tensor& tensor);

    void wait();

private:
    std::shared_ptr<Trampoline> trampoline_;
};

class Trampoline : public std::enable_shared_from_this<Trampoline> {
public:
    explicit Trampoline(const std::string& device_type);
    void load_device(const std::string& device_type);
    // void do_something();
    std::shared_ptr<handle> get_handle();  // Method to get a handle instance
    void aio_read(torch::Tensor& buffer, const char* filename, const bool validate);
    void aio_write(const torch::Tensor& buffer, const char* filename, const bool validate);
    void deepspeed_memcpy(torch::Tensor& dest, const torch::Tensor& src);
    ~Trampoline();

private:
    BaseDevice* device;
    void* handle_;
    friend class handle;  // Allow handle to access private members
};