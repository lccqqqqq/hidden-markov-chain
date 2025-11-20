#!/usr/bin/env python3
"""Quick test to verify MPS device is available and working"""

import torch

print("PyTorch version:", torch.__version__)
print()

# Check MPS availability
if torch.backends.mps.is_available():
    print("✓ MPS is available!")
    device = torch.device("mps")

    # Test basic operations
    try:
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = x @ y
        print(f"✓ Basic matrix multiplication works on MPS")
        print(f"  Result shape: {z.shape}, device: {z.device}")

        # Test gradient computation
        x.requires_grad = True
        y.requires_grad = True
        z = (x @ y).sum()
        z.backward()
        print(f"✓ Gradient computation works on MPS")

        print()
        print("Your MPS device is ready for training!")
        print("Update your config file with: device: 'mps'")

    except Exception as e:
        print(f"✗ MPS test failed: {e}")

else:
    print("✗ MPS is not available")
    print("  This might be because:")
    print("  - You're not on Apple Silicon (M1/M2/M3)")
    print("  - Your macOS version is too old (need macOS 12.3+)")
    print("  - Your PyTorch version doesn't support MPS")
    print()
    print("Fallback to CPU will be used.")
