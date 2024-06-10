import os

class PluginManager:
    def __init__(self):
        # Initialize the plugins dictionary
        self.plugins = {}
        # Detect available plugins
        self.detect_plugins()

    def available_plugins(self):
        # Return a list of available plugins
        return list(self.plugins.keys())
    
    def detect_plugins(self):
        # Get the absolute path to the plugins directory
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins'))
        # print(os.path.dirname(__file__))
        # print(base_path)
        
        # Iterate over each item in the plugins directory
        # print(os.listdir(base_path))
        for device_type in os.listdir(base_path):
            # Construct the full path to the plugin directory
            plugin_dir = os.path.join(base_path, device_type)
            # Check if the path is a directory
            if os.path.isdir(plugin_dir):
                # Add the plugin information to the plugins dictionary
                self.plugins[device_type] = {
                    'include_paths': self.get_include_paths(device_type),
                    'source_paths': self.get_sources(device_type)
                }

    def get_sources(self, device_type):
        # Get the absolute path to the specific device type directory within plugins
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins', device_type))
        # List all files in the directory and filter for .cpp and .h files
        sources = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.endswith(('.cpp', '.h'))]
        # Return the list of source file paths
        return sources

    def get_include_paths(self, device_type):
        # Get the absolute path to the specific device type directory within plugins
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../csrc/aio/plugins', device_type))
        # Return the base path as the include path
        include_paths = [base_path]
        return include_paths

    def get_plugin_info(self, device_type):
        # Retrieve plugin information for a given device type if it exists
        if device_type in self.plugins:
            return self.plugins[device_type]
        else:
            return None
        
# print(PluginManager().get_plugin_info("nvme"))
