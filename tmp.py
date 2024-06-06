import torch
import argparse
from xformers.components.attention import ScaledDotProduct, SparseCS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, required=True)
    args = parser.parse_args()

    attention = ScaledDotProduct().cuda()

    # FW a random bunch of data
    inputs = torch.rand((16, 1024, 1024), device=torch.device("cuda"))

    # Not a very sparse mask to begin with
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    mask = (torch.rand((1024, 1024)) < args.fraction).cuda()
    att = attention(q=inputs, k=inputs, v=inputs, att_mask=mask)

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
    print(f"Peak memory use: {max_memory}MB")