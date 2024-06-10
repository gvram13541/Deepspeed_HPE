# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import pkgutil
import importlib
try:
    # during installation time accelerator is visible, otherwise return deepspeed.accelerator
    from accelerator import get_accelerator
except ImportError:
    from deepspeed.accelerator import get_accelerator 

# List of all available ops

# reflect all builder names into __op_builders__
op_builder_dir = get_accelerator().op_builder_dir()
op_builder_module = importlib.import_module(op_builder_dir)
__op_builders__ = []


# Import the PluginManager from the current module
from .plugin_manager import PluginManager
plugin_manager = PluginManager()

# Iterate over all available plugins detected by the PluginManager
for plugin_type in plugin_manager.available_plugins():
    # Define the module name for plugin operations
    module_name = f"plugins_ops"
    try:
        # Import the module dynamically using the op builder directory and module name
        module = importlib.import_module(f"{op_builder_dir}.{module_name}")
        # Check if 'PluginsBuilder' is defined in the module
        if 'PluginsBuilder' in module.__dict__:
            # Create an instance of PluginsBuilder with the plugin type
            builder = module.PluginsBuilder(plugin_type)
            # Append the builder to the list of op builders
            __op_builders__.append(builder)
    except ModuleNotFoundError:
        # If the module is not found, continue without doing anything
        pass

for _, module_name, _ in pkgutil.iter_modules([os.path.dirname(op_builder_module.__file__)]):
    # avoid self references
    if module_name != 'all_ops' and module_name != 'builder' and module_name!='plugins_ops':
        module = importlib.import_module("{}.{}".format(op_builder_dir, module_name))
        for member_name in module.__dir__():
            if member_name.endswith('Builder'):
                # append builder to __op_builders__ list
                builder = get_accelerator().create_op_builder(member_name)
                __op_builders__.append(builder)

ALL_OPS = {op.name: op for op in __op_builders__ if op is not None}
accelerator_name = get_accelerator()._name
