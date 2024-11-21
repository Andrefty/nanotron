import torch
from transformers import AutoModelForCausalLM

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

    # Unpack weights
    layers = hf_model.model.layers
    for i, layer in enumerate(layers):
        print(f"Unpacking weights for layer {i}...")

        # Unpack QKV projections
        print(f"Q_proj shape before: {layer.self_attn.q_proj.weight.shape}")
        layer.self_attn.q_proj.weight = unpack_weights(layer.self_attn.q_proj.weight)
        print(f"Q_proj shape after: {layer.self_attn.q_proj.weight.shape}")
        print(f"K_proj shape before: {layer.self_attn.k_proj.weight.shape}")
        layer.self_attn.k_proj.weight = unpack_weights(layer.self_attn.k_proj.weight)
        print(f"K_proj shape after: {layer.self_attn.k_proj.weight.shape}")
        print(f"V_proj shape before: {layer.self_attn.v_proj.weight.shape}")
        layer.self_attn.v_proj.weight = unpack_weights(layer.self_attn.v_proj.weight)
        print(f"V_proj shape after: {layer.self_attn.v_proj.weight.shape}")
        print(f"O_proj shape before: {layer.self_attn.o_proj.weight.shape}")
        layer.self_attn.o_proj.weight = unpack_weights(layer.self_attn.o_proj.weight)
        print(f"O_proj shape after: {layer.self_attn.o_proj.weight.shape}")

        # Unpack MLP projections
        print(f"Gate_proj shape before: {layer.mlp.gate_proj.weight.shape}")
        layer.mlp.gate_proj.weight = unpack_weights(layer.mlp.gate_proj.weight)
        print(f"Gate_proj shape after: {layer.mlp.gate_proj.weight.shape}")
        print(f"Up_proj shape before: {layer.mlp.up_proj.weight.shape}")
        layer.mlp.up_proj.weight = unpack_weights(layer.mlp.up_proj.weight)
        print(f"Up_proj shape after: {layer.mlp.up_proj.weight.shape}")
        print(f"Down_proj shape before: {layer.mlp.down_proj.weight.shape}")
        layer.mlp.down_proj.weight = unpack_weights(layer.mlp.down_proj.weight)
        print(f"Down_proj shape after: {layer.mlp.down_proj.weight.shape}") 

    # Save the unpacked model
    hf_model.save_pretrained(output_dir)
    print(f"Unpacked model saved to {output_dir}")

if __name__ == "__main__":
    main()