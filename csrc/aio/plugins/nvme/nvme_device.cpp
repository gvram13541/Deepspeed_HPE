#include "nvme_device.h"

// This function creates an instance of NVMEDevice and returns a pointer to it.
// This is required to integrate the custom plugin into DeepSpeed's plugin system.

extern "C" DeepSpeedAIOBase* create_device() {
    return new NVMEDevice();
}