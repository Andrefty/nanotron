# %% [markdown]
# ## What is BitNet?
# 
# [BitNet](https://arxiv.org/abs/2402.17764) replaces traditional Linear layers in Multi-Head Attention and Feed-Forward Networks with specialized layers called BitLinear with ternary (or binary in the older version) precision. The BitLinear layers introduce in this notebook quantize the weights using ternary precision (with values of -1, 0, and 1) and quantize the activations to 8-bit precision.
# 
# 
# <figure style="text-align: center;">
#   <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/bitlinear.png" alt="Alt Text" />
#   <figcaption>The architecture of BitNet with BitLinear layers</figcaption>
# </figure>
# 
# It's worth mentioning that the behavior of BitLinear differs between training and inference. For example, during training, we start by quantizing the weights into ternary values, using symmetric per tensor quantization. First, we compute the average of the absolute values of the weight matrix and use this as a scale. We then divide the weights by the scale, round the values, constrain them between -1 and 1, and finally rescale them to continue in full precision.
# 
# $$
# scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|}
# $$
# 
# $$
# W_q = \text{clamp}_{[-1,1]}(\text{round}(W*scale))
# $$
# 
# $$
# W_{dequantized} = W_q*scale_w
# $$
# 
# Activations are then quantized to a specified bit-width (e.g., 8-bit) using [absmax](https://arxiv.org/pdf/2208.07339) quantization (symmetric per channel quantization). This involves scaling the activations into a range [−128,127[. The quantization formula is:
# 
# $$
# scale_x = \frac{127}{|X|_{\text{max}, \, \text{dim}=-1}}
# $$
# 
# $$
# X_q = \text{clamp}_{[-128,127]}(\text{round}(X*scale))
# $$
# 
# $$
# X_{dequantized} = X_q * scale_x
# $$
# 
# The main obstacle to training in ternary precision is that the weight values are discretized (via the `round()` function) and thus non-differentiable. BitLinear solves this with a nice trick: [STE (Straight Through Estimator)](https://arxiv.org/abs/1903.05662). The STE allows gradients to flow through the non-differentiable rounding operation by approximating its gradient as 1 (treating `round()` as equivalent to the identity function). Another way to view it is that, instead of stopping the gradient at the rounding step, the STE lets the gradient pass through as if the rounding never occurred, enabling weight updates using standard gradient-based optimization techniques.
# 
# To learn more about how we trained, and fine-tuned bitnet models checkout the blogpost [here](https://)

# %% [markdown]
# ## How to load bitnet models from the hub ?
# 
# Models in ternary precision are packed with 2 bits per weight. You can load them directly using from_pretrained, provided that the quantization method is specified as bitnet in the config.json.
# 
# Start by changing the runtime to use GPUs, and follow the next steps :

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch

# %%
# model = AutoModelForCausalLM.from_pretrained("/home/andreif/Documents/nanotron/models/Llama3-8B-1.58-100B-tokens", device_map="auto", torch_dtype=torch.bfloat16)
# model = AutoModelForCausalLM.from_pretrained("/home/andreif/Documents/nanotron/models/Unpacked-Llama3-Bitnet", device_map="auto", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("/home/andreif/Documents/nanotron/hf_checkpoints/Converted-Nanotron-Llama-3-8B-Bitnet", device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("/home/andreif/Documents/nanotron/models/Llama3-8B-1.58-100B-tokens")
tokenizer.pad_token = tokenizer.eos_token
 
for name, module in model.named_modules():
    print(f"Layer Name: {name}")
    print(f"Layer Type: {type(module)}")
    # print(f"Layer Parameters: {module}")
    # print(f"Layer attributes: {module.__dict__}")
    # You can potentially inspect specific attributes or methods of the module here
    # For example, if it's a linear layer:
    if isinstance(module, torch.nn.Linear):  # or tf.keras.layers.Dense for TensorFlow
        print(f"  Weight shape: {module.weight.shape}")
        print(f"  weight_scale shape: {module.weight_scale}")
    elif isinstance(module, transformers.integrations.bitnet.BitLinear):
        print(f"  Weight shape: {module.weight.shape}")
        print(f"  weight_scale shape: {module.weight_scale}")
    elif isinstance(module, torch.nn.Embedding):
        print(f"  Embedding size: {module.weight.shape}")
    elif 'attention' in name.lower(): # Example for attention layers
        print("  This is likely an attention layer.")
    print("-" * 30)
# input_text = """
# What is the capital of France?
# """

# input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
# output = model.generate(input_ids, max_new_tokens=6)
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# generated_text


