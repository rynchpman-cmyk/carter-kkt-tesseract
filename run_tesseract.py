#!/usr/bin/env python3
"""Minimal headless Vulkan runner for carter_tesseract_kkt.comp."""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

from vulkan import *  # type: ignore[import-untyped]  # noqa: F403
from vulkan import ffi  # type: ignore[import-untyped]


HERE = Path(__file__).resolve().parent
SPIRV_PATH = HERE / "carter_tesseract_kkt.spv"
DEBUG_RECORD_SIZE = 80
MASTER_WEIGHT_COUNT = 1287


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "values",
        metavar="Z",
        type=float,
        nargs="*",
        default=[-1.0, -0.1, 0.0, 0.1, 1.0],
        help="input z values (defaults: -1 -0.1 0 0.1 1)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="portable hybrid_model.json exported by export_hybrid.py",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="also write machine-readable results to this path",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="number of recurrent Vulkan dispatches (default: 1)",
    )
    return parser.parse_args()


def find_compute_queue(physical_device) -> int:
    families = vkGetPhysicalDeviceQueueFamilyProperties(physical_device)  # noqa: F405
    for index, family in enumerate(families):
        if family.queueFlags & VK_QUEUE_COMPUTE_BIT:  # noqa: F405
            return index
    raise RuntimeError("The selected Vulkan device has no compute queue.")


def find_memory_type(physical_device, type_bits: int, required_flags: int) -> int:
    properties = vkGetPhysicalDeviceMemoryProperties(physical_device)  # noqa: F405
    for index in range(properties.memoryTypeCount):
        flags = properties.memoryTypes[index].propertyFlags
        if type_bits & (1 << index) and (flags & required_flags) == required_flags:
            return index
    raise RuntimeError(f"No compatible Vulkan memory type for flags 0x{required_flags:x}.")


def create_buffer(device, physical_device, size: int, usage: int):
    info = VkBufferCreateInfo(  # noqa: F405
        sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,  # noqa: F405
        size=size,
        usage=usage,
        sharingMode=VK_SHARING_MODE_EXCLUSIVE,  # noqa: F405
    )
    buffer = vkCreateBuffer(device, info, None)  # noqa: F405
    requirements = vkGetBufferMemoryRequirements(device, buffer)  # noqa: F405
    flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  # noqa: F405,E501
    memory_info = VkMemoryAllocateInfo(  # noqa: F405
        sType=VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,  # noqa: F405
        allocationSize=requirements.size,
        memoryTypeIndex=find_memory_type(
            physical_device, requirements.memoryTypeBits, flags
        ),
    )
    memory = vkAllocateMemory(device, memory_info, None)  # noqa: F405
    vkBindBufferMemory(device, buffer, memory, 0)  # noqa: F405
    return buffer, memory


def write_memory(device, memory, payload: bytes) -> None:
    pointer = vkMapMemory(device, memory, 0, len(payload), 0)  # noqa: F405
    try:
        ffi.memmove(pointer, payload, len(payload))
    finally:
        vkUnmapMemory(device, memory)  # noqa: F405


def read_memory(device, memory, size: int) -> bytes:
    pointer = vkMapMemory(device, memory, 0, size, 0)  # noqa: F405
    try:
        return bytes(pointer)[:size]
    finally:
        vkUnmapMemory(device, memory)  # noqa: F405


def load_hybrid_model(path: Path | None) -> dict | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "carter-tesseract-hybrid":
        raise ValueError(f"{path} is not a Carter Tesseract hybrid model.")
    if document.get("version") != 1 or document.get("master_width") != 32:
        raise ValueError("Unsupported hybrid model version or master width.")
    weights = document.get("master_weights", [])
    if len(weights) != MASTER_WEIGHT_COUNT:
        raise ValueError(
            f"Expected {MASTER_WEIGHT_COUNT} master weights, got {len(weights)}."
        )
    if not all(math.isfinite(float(value)) for value in weights):
        raise ValueError("The hybrid model contains non-finite weights.")
    return document


def build_params(element_count: int, model: dict | None) -> bytes:
    # std140 CarterParams: 24 floats, a column-major mat4, then uint/uint/int/uint.
    coefficients = [
        0.25, 0.10, 0.05, 0.02,       # c, kappa, alpha, chi
        0.01, 0.50, 1.0e-6, 1.0e-5,  # zeta, omegaDelta, epsilonN, tolerance
        1.00, 0.10, 0.10, 0.10,      # free-energy weights
        0.50, 0.20, 0.20, 0.10,      # lambda0, lambdaP, lambdaM, lambdaC
        0.05, 0.05, 0.02, 0.01,      # quadratic lambda terms
        0.10, 0.10, 0.02, 0.05,      # gamma/q terms, branchSplit
    ]
    target_t = [[0.0] * 4 for _ in range(4)]
    if model is not None:
        dynamics = model["dynamics"]
        coefficients[0] = float(dynamics["c"])
        coefficients[1] = float(dynamics["kappa"])
        coefficients[2] = float(dynamics["alpha"])
        coefficients[4] = float(dynamics["zeta"])
        coefficients[5] = float(dynamics["omega_delta"])
        target_t = model["target_t"]
    # GLSL mat4 storage is column-major; targetT[B][A] represents T^A_B.
    target_column_major = [
        float(target_t[row][column])
        for column in range(4)
        for row in range(4)
    ]
    return (
        struct.pack("<24f", *coefficients)
        + struct.pack("<16f", *target_column_major)
        + struct.pack("<IIiI", element_count, 0, 0, int(model is not None))
    )


def decode_debug(payload: bytes, count: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        offset = index * DEBUG_RECORD_SIZE
        floats = struct.unpack_from("<16f", payload, offset)
        masks = struct.unpack_from("<4I", payload, offset + 64)
        records.append(
            {
                "map": floats[0:4],
                "correction": floats[4:8],
                "branches": floats[8:12],
                "carter": floats[12:16],
                "masks": masks,
            }
        )
    return records


def main() -> int:
    args = parse_args()
    values = args.values
    model = load_hybrid_model(args.model)
    if not values:
        raise SystemExit("Provide at least one z value.")
    if args.steps < 1:
        raise SystemExit("--steps must be positive.")
    if not SPIRV_PATH.exists():
        raise SystemExit(f"Missing {SPIRV_PATH.name}; run .\\run.ps1 to compile it.")

    application_info = VkApplicationInfo(  # noqa: F405
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,  # noqa: F405
        pApplicationName="Carter Tesseract KKT",
        applicationVersion=1,
        pEngineName="none",
        engineVersion=1,
        apiVersion=(1 << 22) | (2 << 12),  # Vulkan 1.2
    )
    instance_info = VkInstanceCreateInfo(  # noqa: F405
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,  # noqa: F405
        pApplicationInfo=application_info,
    )
    instance = vkCreateInstance(instance_info, None)  # noqa: F405

    device = None
    descriptor_pool = None
    command_pool = None
    pipeline = None
    pipeline_layout = None
    descriptor_layout = None
    shader_module = None
    allocations = []

    try:
        physical_devices = vkEnumeratePhysicalDevices(instance)  # noqa: F405
        if not physical_devices:
            raise RuntimeError("No Vulkan physical devices found.")
        physical_device = physical_devices[0]
        device_properties = vkGetPhysicalDeviceProperties(physical_device)  # noqa: F405
        queue_family = find_compute_queue(physical_device)

        priority = [1.0]
        queue_info = VkDeviceQueueCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,  # noqa: F405
            queueFamilyIndex=queue_family,
            queueCount=1,
            pQueuePriorities=priority,
        )
        device_info = VkDeviceCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,  # noqa: F405
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
        )
        device = vkCreateDevice(physical_device, device_info, None)  # noqa: F405
        queue = vkGetDeviceQueue(device, queue_family, 0)  # noqa: F405

        descriptor_bindings = [
            VkDescriptorSetLayoutBinding(  # noqa: F405
                binding=0,
                descriptorType=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  # noqa: F405
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,  # noqa: F405
            )
        ]
        descriptor_bindings.extend(
            VkDescriptorSetLayoutBinding(  # noqa: F405
                binding=binding,
                descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  # noqa: F405
                descriptorCount=1,
                stageFlags=VK_SHADER_STAGE_COMPUTE_BIT,  # noqa: F405
            )
            for binding in range(1, 6)
        )
        layout_info = VkDescriptorSetLayoutCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,  # noqa: F405
            bindingCount=len(descriptor_bindings),
            pBindings=descriptor_bindings,
        )
        descriptor_layout = vkCreateDescriptorSetLayout(device, layout_info, None)  # noqa: F405,E501

        pipeline_layout_info = VkPipelineLayoutCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,  # noqa: F405
            setLayoutCount=1,
            pSetLayouts=[descriptor_layout],
        )
        pipeline_layout = vkCreatePipelineLayout(device, pipeline_layout_info, None)  # noqa: F405,E501

        spirv = SPIRV_PATH.read_bytes()
        shader_info = VkShaderModuleCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,  # noqa: F405
            codeSize=len(spirv),
            pCode=spirv,
        )
        shader_module = vkCreateShaderModule(device, shader_info, None)  # noqa: F405
        stage = VkPipelineShaderStageCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,  # noqa: F405
            stage=VK_SHADER_STAGE_COMPUTE_BIT,  # noqa: F405
            module=shader_module,
            pName="main",
        )
        compute_info = VkComputePipelineCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,  # noqa: F405
            stage=stage,
            layout=pipeline_layout,
        )
        pipeline = vkCreateComputePipelines(  # noqa: F405
            device, VK_NULL_HANDLE, 1, [compute_info], None  # noqa: F405
        )[0]

        input_payload = struct.pack(f"<{len(values)}f", *values)
        params_payload = build_params(len(values), model)
        master_weights = (
            model["master_weights"] if model is not None else [0.0] * MASTER_WEIGHT_COUNT
        )
        master_payload = struct.pack(
            f"<{MASTER_WEIGHT_COUNT}f", *master_weights
        )
        sizes = [
            len(params_payload),
            len(input_payload),
            len(input_payload),
            256 * 4,
            len(values) * DEBUG_RECORD_SIZE,
            len(master_payload),
        ]
        usages = [
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,  # noqa: F405
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # noqa: F405
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # noqa: F405
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # noqa: F405
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # noqa: F405
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,  # noqa: F405
        ]
        for size, usage in zip(sizes, usages):
            allocations.append(
                create_buffer(device, physical_device, size, usage)
            )

        write_memory(device, allocations[0][1], params_payload)
        write_memory(device, allocations[1][1], input_payload)
        write_memory(device, allocations[2][1], bytes(sizes[2]))
        write_memory(device, allocations[3][1], bytes(sizes[3]))
        write_memory(device, allocations[4][1], bytes(sizes[4]))
        write_memory(device, allocations[5][1], master_payload)

        pool_sizes = [
            VkDescriptorPoolSize(  # noqa: F405
                type=VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1  # noqa: F405
            ),
            VkDescriptorPoolSize(  # noqa: F405
                type=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=5  # noqa: F405
            ),
        ]
        pool_info = VkDescriptorPoolCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,  # noqa: F405
            maxSets=1,
            poolSizeCount=len(pool_sizes),
            pPoolSizes=pool_sizes,
        )
        descriptor_pool = vkCreateDescriptorPool(device, pool_info, None)  # noqa: F405
        allocation_info = VkDescriptorSetAllocateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,  # noqa: F405
            descriptorPool=descriptor_pool,
            descriptorSetCount=1,
            pSetLayouts=[descriptor_layout],
        )
        descriptor_set = vkAllocateDescriptorSets(device, allocation_info)[0]  # noqa: F405
        writes = []
        for binding, ((buffer, _), size) in enumerate(zip(allocations, sizes)):
            buffer_info = VkDescriptorBufferInfo(  # noqa: F405
                buffer=buffer, offset=0, range=size
            )
            writes.append(
                VkWriteDescriptorSet(  # noqa: F405
                    sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,  # noqa: F405
                    dstSet=descriptor_set,
                    dstBinding=binding,
                    descriptorCount=1,
                    descriptorType=(
                        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER  # noqa: F405
                        if binding == 0
                        else VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  # noqa: F405
                    ),
                    pBufferInfo=[buffer_info],
                )
            )
        vkUpdateDescriptorSets(device, len(writes), writes, 0, None)  # noqa: F405

        command_pool_info = VkCommandPoolCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,  # noqa: F405
            queueFamilyIndex=queue_family,
        )
        command_pool = vkCreateCommandPool(device, command_pool_info, None)  # noqa: F405
        command_buffer_info = VkCommandBufferAllocateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,  # noqa: F405
            commandPool=command_pool,
            level=VK_COMMAND_BUFFER_LEVEL_PRIMARY,  # noqa: F405
            commandBufferCount=1,
        )
        command_buffer = vkAllocateCommandBuffers(device, command_buffer_info)[0]  # noqa: F405,E501
        begin_info = VkCommandBufferBeginInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO  # noqa: F405
        )
        vkBeginCommandBuffer(command_buffer, begin_info)  # noqa: F405
        vkCmdBindPipeline(  # noqa: F405
            command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline  # noqa: F405
        )
        vkCmdBindDescriptorSets(  # noqa: F405
            command_buffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,  # noqa: F405
            pipeline_layout,
            0,
            1,
            [descriptor_set],
            0,
            None,
        )
        vkCmdDispatch(command_buffer, math.ceil(len(values) / 64), 1, 1)  # noqa: F405
        vkEndCommandBuffer(command_buffer)  # noqa: F405

        submit_info = VkSubmitInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_SUBMIT_INFO,  # noqa: F405
            commandBufferCount=1,
            pCommandBuffers=[command_buffer],
        )
        fence_info = VkFenceCreateInfo(  # noqa: F405
            sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO  # noqa: F405
        )
        for step_index in range(args.steps):
            fence = vkCreateFence(device, fence_info, None)  # noqa: F405
            try:
                vkQueueSubmit(queue, 1, [submit_info], fence)  # noqa: F405
                vkWaitForFences(  # noqa: F405
                    device, 1, [fence], VK_TRUE, 30_000_000_000  # noqa: F405
                )
            finally:
                vkDestroyFence(device, fence, None)  # noqa: F405
            if step_index + 1 < args.steps:
                recurrent_payload = read_memory(
                    device, allocations[2][1], sizes[2]
                )
                write_memory(device, allocations[1][1], recurrent_payload)

        output_payload = read_memory(device, allocations[2][1], sizes[2])
        debug_payload = read_memory(device, allocations[4][1], sizes[4])
        outputs = struct.unpack(f"<{len(values)}f", output_payload)
        debug_records = decode_debug(debug_payload, len(values))

        print(f"device: {device_properties.deviceName}")
        print("       z -> z_next       sigma iota kkt_valid active classifier")
        for z_value, output, debug in zip(values, outputs, debug_records):
            branches = debug["branches"]
            masks = debug["masks"]
            print(
                f"{z_value:8.4f} -> {output:12.6g}"
                f"  {branches[3]:+4.0f}  {branches[2]:4.0f}"
                f" {masks[3]:9d} {masks[0]:6d} 0x{masks[1]:02x}"
            )
        if not all(math.isfinite(value) for value in outputs):
            raise RuntimeError("The shader produced a non-finite output.")
        if not all(record["masks"][3] == 1 for record in debug_records):
            raise RuntimeError("At least one bounded KKT solve was invalid.")
        if args.json_output is not None:
            result_document = {
                "device": device_properties.deviceName,
                "model": str(args.model) if args.model is not None else None,
                "steps": args.steps,
                "inputs": values,
                "outputs": list(outputs),
                "debug": debug_records,
            }
            args.json_output.write_text(
                json.dumps(result_document, indent=2), encoding="utf-8"
            )
        return 0
    finally:
        if device is not None:
            vkDeviceWaitIdle(device)  # noqa: F405
            if command_pool is not None:
                vkDestroyCommandPool(device, command_pool, None)  # noqa: F405
            if descriptor_pool is not None:
                vkDestroyDescriptorPool(device, descriptor_pool, None)  # noqa: F405
            for buffer, memory in reversed(allocations):
                vkDestroyBuffer(device, buffer, None)  # noqa: F405
                vkFreeMemory(device, memory, None)  # noqa: F405
            if pipeline is not None:
                vkDestroyPipeline(device, pipeline, None)  # noqa: F405
            if shader_module is not None:
                vkDestroyShaderModule(device, shader_module, None)  # noqa: F405
            if pipeline_layout is not None:
                vkDestroyPipelineLayout(device, pipeline_layout, None)  # noqa: F405
            if descriptor_layout is not None:
                vkDestroyDescriptorSetLayout(device, descriptor_layout, None)  # noqa: F405
            vkDestroyDevice(device, None)  # noqa: F405
        vkDestroyInstance(instance, None)  # noqa: F405


if __name__ == "__main__":
    raise SystemExit(main())
