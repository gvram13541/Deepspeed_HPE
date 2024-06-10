import distutils.spawn
import subprocess
from .plugin_manager import PluginManager
from .builder import OpBuilder

class PluginsBuilder(OpBuilder):
    BUILD_VAR = "PLUGINS_BUILD_AIO"  # Environment variable for building plugins
    # NAME = "plugins"  # Name of the plugin builder

    def __init__(self, device_type):
        # Initialize the device type and call the superclass constructor
        self.device_type = device_type
        super().__init__(name=self.device_type)
        # Initialize the PluginManager
        self.plugin_manager = PluginManager()
        self.device_module = None
        self.trampoline = None
        # Initialize sources and include paths for the plugin
        self.initialize_sources_and_paths()

    def initialize_sources_and_paths(self):
        # Retrieve plugin information from the PluginManager
        plugin_info = self.plugin_manager.get_plugin_info(self.device_type)
        if plugin_info:
            # Set the source and include paths based on the plugin information
            self.sources = plugin_info['source_paths']
            self.include_paths = plugin_info['include_paths']

    def absolute_name(self):
        # Return the absolute name of the plugin module
        # print(f'deepspeed.ops.plugins.{self.NAME}_op')
        return f'deepspeed.ops.plugins.{self.device_type}_op'

    def sources(self):
        # Retrieve the sources for the plugin from the PluginManager
        return self.plugin_manager.get_plugin_sources(self.device_type)
    
    def include_paths(self):
        # Base include path for common headers
        base_include_path = ['csrc/aio/common']
        # Include paths specific to the device type
        device_include_paths = self.plugin_manager.get_plugin_include_paths(self.device_type)
        # Return the combined list of include paths
        return base_include_path + device_include_paths

    def cxx_args(self):
        # Define the C++ compilation arguments
        CPU_ARCH = self.cpu_arch()
        SIMD_WIDTH = self.simd_width()
        import torch  # Import torch to get its version
        TORCH_MAJOR, TORCH_MINOR = map(int, torch.__version__.split('.')[0:2])
        if TORCH_MAJOR >= 2 and TORCH_MINOR >= 1:
            CPP_STD = '-std=c++17'
        else:
            CPP_STD = '-std=c++14'
        return [
            '-g',  # Generate debugging information
            '-Wall',  # Enable all warnings
            '-O0',  # Disable optimizations for debugging
            CPP_STD,  # Set the C++ standard
            '-shared',  # Create a shared library
            '-fPIC',  # Generate position-independent code
            '-Wno-reorder',  # Suppress reordering warnings
            CPU_ARCH,  # CPU architecture flags
            '-fopenmp',  # Enable OpenMP support
            SIMD_WIDTH,  # SIMD width flags
            '-laio',  # Link with the libaio library
        ]

    def extra_ldflags(self):
        # Additional linker flags
        return ['-laio']

    def check_for_libaio_pkg(self):
        # Check for the presence of the libaio package using known package managers
        libs = dict(
            dpkg=["-l", "libaio-dev", "apt"],
            pacman=["-Q", "libaio", "pacman"],
            rpm=["-q", "libaio-devel", "yum"],
        )

        found = False
        for pkgmgr, data in libs.items():
            flag, lib, tool = data
            path = distutils.spawn.find_executable(pkgmgr)
            if path is not None:
                cmd = f"{pkgmgr} {flag} {lib}"
                result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                if result.wait() == 0:
                    found = True
                else:
                    self.warning(f"{self.NAME}: please install the {lib} package with {tool}")
                break
        return found

    def is_compatible(self, verbose=True): 
        # Check for the existence of libaio by using distutils to compile and link a test program
        # that calls io_submit, which is a function provided by libaio
        aio_compatible = self.has_function('io_pgetevents', ('aio', ))
        if verbose and not aio_compatible:
            self.warning(f"{self.NAME} requires the dev libaio .so object and headers but these were not found.")

            # Check for the libaio package via known package managers to print suggestions on which package to install
            self.check_for_libaio_pkg()

            self.warning(
                "If libaio is already installed (perhaps from source), try setting the CFLAGS and LDFLAGS environment variables to where it can be found."
            )
        # Call the superclass's is_compatible method and combine the result with the aio_compatible check
        return super().is_compatible(verbose) and aio_compatible

# PluginsBuilder().absolute_name()