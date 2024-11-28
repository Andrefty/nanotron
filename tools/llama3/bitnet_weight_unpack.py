import torch
from transformers import AutoModelForCausalLM
import json
import os

def unpack_weights(packed: torch.Tensor, bits: int = 2) -> torch.Tensor:
    values_per_item = 8 // bits
    packed_shape = packed.shape

    if len(packed_shape) == 1:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim,)
    else:
        original_row_dim = packed_shape[0] * values_per_item
        unpacked_shape = (original_row_dim, *packed_shape[1:])

    unpacked = torch.zeros(unpacked_shape, device=packed.device, dtype=torch.uint8)

    for i in range(values_per_item):
        start = i * packed_shape[0]
        end = start + packed_shape[0]
        mask = (3 << (2 * i))
        unpacked[start:end] = (packed & mask) >> (2 * i)

    unpacked = unpacked.to(torch.float) - 1
    return unpacked

def main():
    pretrained_model_name_or_path = "/home/andreif/Documents/nanotron/models/Llama3-8B-1.58-100B-tokens"
    output_dir = "/home/andreif/Documents/nanotron/models/Unpacked-Llama3-Bitnet/"

    # Load the HF model
    hf_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
    hf_model.eval()

    # Create a directory for logs if it doesn't exist
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Unpack weights
    layers = hf_model.model.layers
    for i, layer in enumerate(layers):
        print(f"Unpacking weights for layer {i}...")
        layer_log = {}

        # Show a diff between the original and modified weights (visual check)
        layer_log["Q_proj_diff"] = layer.self_attn.q_proj.weight[:10, :10].tolist()
        layer_log["K_proj_diff"] = layer.self_attn.k_proj.weight[:10, :10].tolist()
        layer_log["V_proj_diff"] = layer.self_attn.v_proj.weight[:10, :10].tolist()
        layer_log["O_proj_diff"] = layer.self_attn.o_proj.weight[:10, :10].tolist()
        layer_log["Gate_proj_diff"] = layer.mlp.gate_proj.weight[:10, :10].tolist()
        layer_log["Up_proj_diff"] = layer.mlp.up_proj.weight[:10, :10].tolist()
        layer_log["Down_proj_diff"] = layer.mlp.down_proj.weight[:10, :10].tolist()

        # Unpack QKV projections
        layer_log["Q_proj_shape_before"] = layer.self_attn.q_proj.weight.shape
        layer.self_attn.q_proj.weight = unpack_weights(layer.self_attn.q_proj.weight)
        layer_log["Q_proj_shape_after"] = layer.self_attn.q_proj.weight.shape

        layer_log["K_proj_shape_before"] = layer.self_attn.k_proj.weight.shape
        layer.self_attn.k_proj.weight = unpack_weights(layer.self_attn.k_proj.weight)
        layer_log["K_proj_shape_after"] = layer.self_attn.k_proj.weight.shape

        layer_log["V_proj_shape_before"] = layer.self_attn.v_proj.weight.shape
        layer.self_attn.v_proj.weight = unpack_weights(layer.self_attn.v_proj.weight)
        layer_log["V_proj_shape_after"] = layer.self_attn.v_proj.weight.shape

        layer_log["O_proj_shape_before"] = layer.self_attn.o_proj.weight.shape
        layer.self_attn.o_proj.weight = unpack_weights(layer.self_attn.o_proj.weight)
        layer_log["O_proj_shape_after"] = layer.self_attn.o_proj.weight.shape

        # Unpack MLP projections
        layer_log["Gate_proj_shape_before"] = layer.mlp.gate_proj.weight.shape
        layer.mlp.gate_proj.weight = unpack_weights(layer.mlp.gate_proj.weight)
        layer_log["Gate_proj_shape_after"] = layer.mlp.gate_proj.weight.shape

        layer_log["Up_proj_shape_before"] = layer.mlp.up_proj.weight.shape
        layer.mlp.up_proj.weight = unpack_weights(layer.mlp.up_proj.weight)
        layer_log["Up_proj_shape_after"] = layer.mlp.up_proj.weight.shape

        layer_log["Down_proj_shape_before"] = layer.mlp.down_proj.weight.shape
        layer.mlp.down_proj.weight = unpack_weights(layer.mlp.down_proj.weight)
        layer_log["Down_proj_shape_after"] = layer.mlp.down_proj.weight.shape
        
        layer_log["Q_proj_diff_after"] = layer.self_attn.q_proj.weight[:10, :10].tolist()
        layer_log["K_proj_diff_after"] = layer.self_attn.k_proj.weight[:10, :10].tolist()
        layer_log["V_proj_diff_after"] = layer.self_attn.v_proj.weight[:10, :10].tolist()
        layer_log["O_proj_diff_after"] = layer.self_attn.o_proj.weight[:10, :10].tolist()
        layer_log["Gate_proj_diff_after"] = layer.mlp.gate_proj.weight[:10, :10].tolist()
        layer_log["Up_proj_diff_after"] = layer.mlp.up_proj.weight[:10, :10].tolist()
        layer_log["Down_proj_diff_after"] = layer.mlp.down_proj.weight[:10, :10].tolist()
        # Save log for the current layer
        with open(os.path.join(log_dir, f"layer_{i}_log.json"), "w") as log_file:
            json.dump(layer_log, log_file, indent=4)

    # Save the unpacked model
    hf_model.save_pretrained(output_dir)
    print(f"Unpacked model saved to {output_dir}")

if __name__ == "__main__":
    main()