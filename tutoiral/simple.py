import torch
import flashinfer

# Single decode attention
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
print(output)
