import torch
from transformers import AutoModelForCausalLM
import os
import json

def pack_weights(intweights: torch.Tensor, bits: int) -> torch.Tensor:
    # intweights = intweights.clone()  # make sure not to modify original
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

    def lshift(t: torch.Tensor, bits: int):
        return t << bits

    it = min(values_per_item, (original_shape[0] // row_dim) + 1)
    for i in range(it):
        start = i * row_dim
        end = min(start + row_dim, original_shape[0])
        packed[: (end - start)] |= lshift(unpacked[start:end], bits * i)

    return packed

def quantize_tensor(tensor: torch.Tensor, bits: int = 2):
    # Compute a weight scale from the maximum absolute value.
    # Prevent division by zero.
    scale = tensor.abs().max()
    if scale == 0:
        scale = torch.tensor(1.0, device=tensor.device, dtype=tensor.dtype)
    # Quantize: map tensor values to the range of [0, 2**bits - 1]
    quant_max = 2 ** bits - 1
    quantized = (tensor / scale * (quant_max / 2)).round().clamp(0, quant_max)
    quantized = quantized.to(torch.int32)
    return quantized, scale

def main():
    # Input and output model paths.
    pretrained_model_name_or_path = "/home/andreif/Documents/nanotron/hf_checkpoints/Converted-Nanotron-Llama-3-8B-Bitnet"
    # pretrained_model_name_or_path = "/home/andreif/Documents/nanotron/hf_checkpoints/Converted-Nanotron-Llama-3-8B-Bitnet-remind"
    output_dir = "/home/andreif/Documents/nanotron/hf_checkpoints/Packed-Llama3-8B-Bitnet/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the HF model.
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, device_map="cpu", torch_dtype=torch.bfloat16)
    model.eval()
    
    # Dictionary to store quantization scales per layer.
    quant_log = {}
    
    # Process each layer. Adjust which attributes to quantize as needed.
    for i, layer in enumerate(model.model.layers):
        quant_log[f"layer_{i}"] = {}
        # Quantize self-attention projections
        for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            weight = getattr(layer.self_attn, f"{attr}").weight
            quantized, scale = quantize_tensor(weight, bits=2)
            packed = pack_weights(quantized, bits=2)
            # Replace weight with the packed version and store scale.
            setattr(layer.self_attn, f"{attr}.weight", packed)
            setattr(layer.self_attn, f"{attr}.weight_scale", scale.to(torch.bfloat16))
            quant_log[f"layer_{i}"][f"self_attn.{attr}.weight_scale"] = scale.item()
        # Quantize MLP projections
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            weight = getattr(layer.mlp, f"{proj}").weight
            quantized, scale = quantize_tensor(weight, bits=2)
            packed = pack_weights(quantized, bits=2)
            setattr(layer.mlp, f"{proj}.weight", packed)
            setattr(layer.mlp, f"{proj}.weight_scale", scale.to(torch.bfloat16))
            quant_log[f"layer_{i}"][f"mlp.{proj}.weight_scale"] = scale.item()
    
    # Save quantization log for debugging.
    with open(os.path.join(output_dir, "quant_log.json"), "w") as f:
        json.dump(quant_log, f, indent=4)
    
    # Save the packed (quantized) model.
    model.save_pretrained(output_dir)
    print(f"Packed model saved to {output_dir}")

if __name__ == "__main__":
    main()