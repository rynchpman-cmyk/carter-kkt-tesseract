#!/usr/bin/env python3
"""Compile-load the complete Carter/KKT Slang migration on Vulkan."""

from pathlib import Path

import slangpy as spy


HERE = Path(__file__).resolve().parent
device = spy.create_device(type=spy.DeviceType.vulkan, include_paths=[HERE])
module = spy.Module.load_from_file(
    device, str(HERE / "carter_tesseract_full.slang")
)
print("complete Carter/KKT Slang module loaded on Vulkan")
print(module)
