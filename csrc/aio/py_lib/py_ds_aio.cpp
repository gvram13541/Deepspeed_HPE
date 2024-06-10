#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include "trampoline.h"  // Include the header file for your Trampoline class

namespace py = pybind11;

PYBIND11_MODULE(py_ds_trampoline, m) {

    // Expose the load_device function directly
    m.def("load_device", [](const std::string& device_type) {
        auto aio = std::make_shared<Trampoline>(device_type);
        aio->load_device(device_type);
    }, "Load Device");

    m.def("aio_read", [](std::shared_ptr<Trampoline> aio, torch::Tensor& buffer, const char* filename, const bool validate) {
        aio->aio_read(buffer, filename, validate);
    }, "DeepSpeed Asynchronous I/O Read");

    m.def("aio_write", [](std::shared_ptr<Trampoline> aio, const torch::Tensor& buffer, const char* filename, const bool validate) {
        aio->aio_write(buffer, filename, validate);
    }, "DeepSpeed Asynchronous I/O Write");

    m.def("deepspeed_memcpy", [](std::shared_ptr<Trampoline> aio, torch::Tensor& dest, const torch::Tensor& src) {
        aio->deepspeed_memcpy(dest, src);
    }, "DeepSpeed Memory Copy");

    py::class_<handle, std::shared_ptr<handle>>(m, "aio_handle")
        .def("get_block_size", &Trampoline::get_block_size)
        .def("get_queue_depth", &Trampoline::get_queue_depth)
        .def("get_single_submit", &Trampoline::get_single_submit)
        .def("get_overlap_events", &Trampoline::get_overlap_events)
        .def("get_thread_count", &Trampoline::get_thread_count)
        .def("read", &Trampoline::read)
        .def("write", &Trampoline::write)
        .def("pread", &Trampoline::pread)
        .def("pwrite", &Trampoline::pwrite)
        .def("sync_pread", &Trampoline::sync_pread)
        .def("sync_pwrite", &Trampoline::sync_pwrite)
        .def("async_pread", &Trampoline::async_pread)
        .def("async_pwrite", &Trampoline::async_pwrite)
        .def("new_cpu_locked_tensor", &Trampoline::new_cpu_locked_tensor)
        .def("free_cpu_locked_tensor", &Trampoline::free_cpu_locked_tensor)
        .def("wait", &Trampoline::wait);

    py::class_<Trampoline, std::shared_ptr<Trampoline>>(m, "Trampoline")
        .def(py::init<const std::string&>())
        .def("load_device", &Trampoline::load_device)
        .def("get_handle", &Trampoline::get_handle)
        .def("aio_read", &Trampoline::aio_read)
        .def("aio_write", &Trampoline::aio_write)
        .def("deepspeed_memcpy", &Trampoline::deepspeed_memcpy);
}

