#pragma once

class DeepSpeedAIOBase {
public:
    virtual ~DeepSpeedAIOBase() = default;

    virtual void aio_read() = 0;
    virtual void aio_write() = 0;
    virtual void deepspeed_memcpy() = 0;

    virtual int get_block_size() const = 0;
    virtual int get_queue_depth() const = 0;
    virtual bool get_single_submit() const = 0;
    virtual bool get_overlap_events() const = 0;
    virtual int get_thread_count() const = 0;

    virtual void read() = 0;
    virtual void write() = 0;
    virtual void pread() = 0;
    virtual void pwrite() = 0;

    virtual void sync_pread() = 0;
    virtual void sync_pwrite() = 0;
    virtual void async_pread() = 0;
    virtual void async_pwrite() = 0;

    virtual void new_cpu_locked_tensor() = 0;
    virtual void free_cpu_locked_tensor() = 0;

    virtual void wait() = 0;
};


