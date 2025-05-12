import torch
from transformers import AutoModelForCausalLM, AutoConfig
import os
import json
import gc

def pack_weights(intweights: torch.Tensor, bits: int) -> torch.Tensor:
    # Make a copy to ensure we don't modify the original
    intweights = intweights.clone()
    intweights += 1
    original_shape = intweights.shape
    values_per_item = 8 // bits
    row_dim = (original_shape[0] + values_per_item - 1) // values_per_item

    if len(original_shape) == 1:
        packed_tensor_shape = (row_dim,)
    else:
        packed_tensor_shape = (row_dim, *original_shape[1:])

    packed = torch.zeros(packed_tensor_shape, device=intweights.device, dtype=torch.uint8)
    unpacked = intweights.to(torch.uint8)
    
    # Free memory
    del intweights
    
    def lshift(t: torch.Tensor, bits: int):
        return t << bits

    it = min(values_per_item, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= lshift(unpacked[start:end], bits * i)
    
    # Free memory
    del unpacked
    
    return packed

def quantize_tensor(tensor: torch.Tensor, bits: int = 2):
    # Compute weight scale from the maximum absolute value
    scale = tensor.abs().max()
    if scale == 0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
    
    # Ensure scale has shape [1] instead of being a scalar
    scale = scale.view(1)
    
    # Quantize: map tensor values to the range of [0, 2**bits - 1]
    quant_max = 2 ** bits - 1
    quantized = (tensor / scale * (quant_max / 2)).round().clamp(0, quant_max)
    quantized = quantized.to(torch.int32)
    
    return quantized, scale

def process_layer(layer, layer_idx, bits=2, quant_log=None, verbose=False):
    """Process a single layer to reduce memory usage"""
    layer_stats = {}
    layer_original_size = 0
    layer_packed_size = 0
    
    # Process self-attention modules
    for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        module = getattr(layer.self_attn, attr)
        weight = module.weight
        layer_original_size += weight.numel() * weight.element_size()
        
        # Quantize and pack
        quantized, scale = quantize_tensor(weight, bits=bits)
        packed = pack_weights(quantized, bits=bits)
        layer_packed_size += packed.numel() * packed.element_size()
        
        # Free memory
        del quantized
        gc.collect()
        
        # Store the packed weights as a buffer instead of a parameter
        if hasattr(module, "weight"):
            del module._parameters["weight"]  # Remove the parameter
        
        # Register buffers for the packed weight and scale
        module.register_buffer("weight_packed", packed)
        module.register_buffer("weight_scale", scale.to(torch.bfloat16))
        
        # Log scale value
        layer_stats[f"self_attn.{attr}.weight_scale"] = float(scale.item())
    
    # Process MLP modules
    for proj in ['gate_proj', 'up_proj', 'down_proj']:
        module = getattr(layer.mlp, proj)
        weight = module.weight
        layer_original_size += weight.numel() * weight.element_size()
        
        # Quantize and pack
        quantized, scale = quantize_tensor(weight, bits=bits)
        packed = pack_weights(quantized, bits=bits)
        layer_packed_size += packed.numel() * packed.element_size()
        
        # Free memory
        del quantized
        gc.collect()
        
        # Store the packed weights as a buffer instead of a parameter
        if hasattr(module, "weight"):
            del module._parameters["weight"]  # Remove the parameter
        
        # Register buffers for the packed weight and scale
        module.register_buffer("weight_packed", packed)
        module.register_buffer("weight_scale", scale.to(torch.bfloat16))
        
        # Log scale value
        layer_stats[f"mlp.{proj}.weight_scale"] = float(scale.item())
    
    # Store stats
    if quant_log is not None:
        quant_log[f"layer_{layer_idx}"] = layer_stats
    
    return layer_original_size, layer_packed_size

def main():
    # Input and output model paths
    pretrained_model_name_or_path = "/home/andreif/Documents/nanotron/hf_checkpoints/Converted-Nanotron-Llama-3-8B-Bitnet"
    # pretrained_model_name_or_path = "/home/andreif/Documents/nanotron/hf_checkpoints/Converted-Nanotron-Llama-3-8B-Bitnet-remind"
    output_dir = "/home/andreif/Documents/nanotron/hf_checkpoints/Packed-Llama3-8B-Bitnet/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration only first
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
    config.is_bitnet_config = True
    config.quantization_config = {"quant_method": "bitnet", "modules_to_not_convert": None, "bits": 2}
    
    # Save the updated config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    
    # Dictionary to store quantization scales per layer
    quant_log = {}
    total_original_size = 0
    total_packed_size = 0
    
    # Process one layer at a time to reduce memory usage
    num_layers = config.num_hidden_layers
    print(f"Processing {num_layers} layers one by one...")
    
    for i in range(num_layers):
        # Load specific layer only
        layer_path = os.path.join(pretrained_model_name_or_path, f"model.layers.{i}")
        print(f"Processing layer {i}/{num_layers}...")
        
        # Load model with specific layer only
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            device_map="cpu",
            torch_dtype=None,
            low_cpu_mem_usage=True
        )
        
        # Process the layer
        layer_original, layer_packed = process_layer(
            model.model.layers[i], 
            i, 
            bits=2, 
            quant_log=quant_log,
            verbose=(i == 0)
        )
        
        total_original_size += layer_original
        total_packed_size += layer_packed
        
        # Save this layer
        model.model.layers[i].save_pretrained(os.path.join(output_dir, f"model.layers.{i}"))
        
        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    print(f"Original size: {total_original_size/1024/1024:.2f} MB, Packed size: {total_packed_size/1024/1024:.2f} MB")
    print(f"Overall compression ratio: {total_original_size/total_packed_size:.2f}x")
    
    # Save quantization log for debugging
    with open(os.path.join(output_dir, "quant_log.json"), "w") as f:
        json.dump(quant_log, f, indent=4)
    
    print(f"Packed model saved to {output_dir}")

if __name__ == "__main__":
    main()