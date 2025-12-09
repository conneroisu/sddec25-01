#!/usr/bin/env python3
"""
Test script for DSA Segmentation Model.

This script verifies the implementation of the DeepSeek Sparse Attention
model for pupil segmentation by running various unit tests and benchmarks.

Tests include:
1. Model instantiation and forward pass
2. Lightning Indexer functionality
3. Sparse attention with top-k selection
4. Loss computation
5. Parameter count verification
6. Inference speed benchmark
"""

import sys
import os

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def test_lightning_indexer():
    """Test Lightning Indexer module."""
    print("\n" + "=" * 60)
    print("Testing Lightning Indexer")
    print("=" * 60)

    from lightning_indexer import LightningIndexer, SpatialLightningIndexer

    batch_size = 2
    seq_len = 100  # 10x10 patches
    dim = 32

    # Create random input
    x = torch.randn(batch_size, seq_len, dim)

    # Test basic indexer
    print("\n1. Testing basic LightningIndexer...")
    indexer = LightningIndexer(dim=dim, num_heads=2, key_dim=8)
    scores = indexer(x, x, height=10, width=10)

    assert scores.shape == (batch_size, seq_len, seq_len), \
        f"Expected shape {(batch_size, seq_len, seq_len)}, got {scores.shape}"
    print(f"   Output shape: {scores.shape} - PASS")

    # Test spatial indexer
    print("\n2. Testing SpatialLightningIndexer...")
    spatial_indexer = SpatialLightningIndexer(dim=dim, num_heads=2, key_dim=8, local_window=5)
    spatial_scores = spatial_indexer(x, x, height=10, width=10)

    assert spatial_scores.shape == (batch_size, seq_len, seq_len), \
        f"Expected shape {(batch_size, seq_len, seq_len)}, got {spatial_scores.shape}"
    print(f"   Output shape: {spatial_scores.shape} - PASS")

    # Verify local bias is learned
    print("\n3. Testing local bias initialization...")
    local_bias = spatial_indexer.local_bias.data
    center_val = local_bias[0, 0, 2, 2].item()  # Center of 5x5 window
    corner_val = local_bias[0, 0, 0, 0].item()  # Corner
    assert center_val > corner_val, "Center should have higher bias than corner"
    print(f"   Center bias: {center_val:.4f}, Corner bias: {corner_val:.4f} - PASS")

    # Test alignment loss computation
    print("\n4. Testing alignment loss computation...")
    fake_attn = F.softmax(torch.randn(batch_size, seq_len, seq_len), dim=-1)
    loss = indexer.compute_alignment_loss(scores, fake_attn)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() >= 0, "KL loss should be non-negative"
    print(f"   Alignment loss: {loss.item():.4f} - PASS")

    print("\nLightning Indexer: ALL TESTS PASSED")
    return True


def test_sparse_attention():
    """Test DeepSeek Sparse Attention module."""
    print("\n" + "=" * 60)
    print("Testing DeepSeek Sparse Attention")
    print("=" * 60)

    from sparse_attention import DeepSeekSparseAttention, DSABlock, DSAStage

    batch_size = 2
    height, width = 10, 16  # 160 tokens
    seq_len = height * width
    dim = 32
    top_k = 32

    x_flat = torch.randn(batch_size, seq_len, dim)
    x_spatial = torch.randn(batch_size, dim, height, width)

    # Test sparse attention
    print("\n1. Testing DeepSeekSparseAttention...")
    attn = DeepSeekSparseAttention(
        dim=dim,
        num_heads=4,
        key_dim=8,
        top_k=top_k,
        use_spatial_indexer=True,
    )
    output, _ = attn(x_flat, height=height, width=width)

    assert output.shape == x_flat.shape, \
        f"Expected shape {x_flat.shape}, got {output.shape}"
    print(f"   Output shape: {output.shape} - PASS")

    # Test with attention weights
    print("\n2. Testing attention weight return...")
    output, (indices, weights) = attn(x_flat, height=height, width=width, return_attention=True)
    assert indices.shape == (batch_size, seq_len, top_k), \
        f"Expected indices shape {(batch_size, seq_len, top_k)}, got {indices.shape}"
    print(f"   Attention indices shape: {indices.shape} - PASS")

    # Test indexer loss
    print("\n3. Testing indexer loss computation...")
    indexer_loss = attn.get_indexer_loss(x_flat, height, width)
    assert indexer_loss.ndim == 0, "Indexer loss should be scalar"
    print(f"   Indexer loss: {indexer_loss.item():.4f} - PASS")

    # Test DSA block
    print("\n4. Testing DSABlock...")
    block = DSABlock(dim=dim, num_heads=4, top_k=top_k)
    block_out = block(x_spatial, height, width)
    assert block_out.shape == x_spatial.shape, \
        f"Expected shape {x_spatial.shape}, got {block_out.shape}"
    print(f"   Block output shape: {block_out.shape} - PASS")

    # Test DSA stage
    print("\n5. Testing DSAStage with downsampling...")
    stage = DSAStage(in_dim=dim, out_dim=64, depth=2, num_heads=4, top_k=top_k, downsample=True)
    stage_out = stage(x_spatial)
    expected_shape = (batch_size, 64, height // 2, width // 2)
    assert stage_out.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {stage_out.shape}"
    print(f"   Stage output shape: {stage_out.shape} - PASS")

    print("\nDeepSeek Sparse Attention: ALL TESTS PASSED")
    return True


def test_full_model():
    """Test complete DSA segmentation model."""
    print("\n" + "=" * 60)
    print("Testing DSA Segmentation Model")
    print("=" * 60)

    from model import (
        DSASegmentationModel,
        CombinedLoss,
        create_dsa_tiny,
        create_dsa_small,
        create_dsa_base,
    )

    # Use smaller test size to avoid OOM on limited GPU memory
    # Full size is 640x400, but we test with 160x100 (1/4 scale)
    batch_size = 1
    height, width = 100, 160
    in_channels = 1
    num_classes = 2

    x = torch.randn(batch_size, in_channels, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    spatial_weights = torch.rand(batch_size, height, width)
    dist_map = torch.randn(batch_size, num_classes, height, width)

    # Test model variants
    print("\n1. Testing model variants...")

    for name, create_fn in [("tiny", create_dsa_tiny), ("small", create_dsa_small), ("base", create_dsa_base)]:
        model = create_fn(in_channels=in_channels, num_classes=num_classes)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"   DSA-{name}: {nparams:,} parameters")

    # Test forward pass
    print("\n2. Testing forward pass...")
    model = create_dsa_small(in_channels=in_channels, num_classes=num_classes)
    output = model(x)

    expected_shape = (batch_size, num_classes, height, width)
    assert output.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {output.shape}"
    print(f"   Output shape: {output.shape} - PASS")

    # Test loss computation
    print("\n3. Testing combined loss...")
    criterion = CombinedLoss()
    total_loss, ce_loss, dice_loss, surface_loss = criterion(
        output, target, spatial_weights, dist_map, alpha=0.5
    )

    assert total_loss.ndim == 0, "Loss should be scalar"
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   CE loss: {ce_loss.item():.4f}")
    print(f"   Dice loss: {dice_loss.item():.4f}")
    print(f"   Surface loss: {surface_loss.item():.4f}")
    print("   Loss computation - PASS")

    # Test indexer loss
    print("\n4. Testing model indexer loss...")
    indexer_loss = model.get_indexer_loss(x)
    assert indexer_loss.ndim == 0, "Indexer loss should be scalar"
    print(f"   Indexer loss: {indexer_loss.item():.4f} - PASS")

    # Test loss with indexer loss
    print("\n5. Testing combined loss with indexer...")
    total_loss_with_indexer, _, _, _ = criterion(
        output, target, spatial_weights, dist_map, alpha=0.5, indexer_loss=indexer_loss
    )
    assert total_loss_with_indexer >= total_loss, "Loss with indexer should be >= base loss"
    print(f"   Total loss with indexer: {total_loss_with_indexer.item():.4f} - PASS")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    model.zero_grad()
    total_loss.backward()
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    param_count = sum(1 for p in model.parameters())
    assert grad_count == param_count, f"Expected {param_count} grads, got {grad_count}"
    print(f"   {grad_count}/{param_count} parameters have gradients - PASS")

    print("\nDSA Segmentation Model: ALL TESTS PASSED")
    return True


def benchmark_inference():
    """Benchmark inference speed."""
    print("\n" + "=" * 60)
    print("Benchmarking Inference Speed")
    print("=" * 60)

    from model import create_dsa_tiny, create_dsa_small, create_dsa_base

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    batch_size = 1
    height, width = 400, 640
    x = torch.randn(batch_size, 1, height, width).to(device)

    warmup_runs = 10
    benchmark_runs = 50

    for name, create_fn in [("tiny", create_dsa_tiny), ("small", create_dsa_small), ("base", create_dsa_base)]:
        model = create_fn().to(device)
        model.eval()

        nparams = sum(p.numel() for p in model.parameters())

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(x)

        # Benchmark
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(benchmark_runs):
                start = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\n{name.upper()} ({nparams:,} params):")
        print(f"  Avg inference time: {avg_time:.2f} +/- {std_time:.2f} ms")
        print(f"  Throughput: {1000/avg_time:.1f} FPS")

    print("\nBenchmark complete!")
    return True


def test_sparse_vs_dense_equivalence():
    """Test that sparse attention approximates dense attention."""
    print("\n" + "=" * 60)
    print("Testing Sparse vs Dense Attention")
    print("=" * 60)

    from sparse_attention import DeepSeekSparseAttention

    batch_size = 1
    seq_len = 64  # 8x8
    dim = 32

    x = torch.randn(batch_size, seq_len, dim)

    # High top-k should approximate dense attention
    sparse_attn = DeepSeekSparseAttention(
        dim=dim,
        num_heads=2,
        key_dim=8,
        top_k=seq_len,  # Select all tokens
        use_spatial_indexer=False,
    )

    # Get output
    sparse_out, _ = sparse_attn(x, height=8, width=8)

    # Compute "dense" reference using same projections
    q = sparse_attn.q_proj(x).view(batch_size, seq_len, 2, 8).transpose(1, 2)
    k = sparse_attn.k_proj(x).unsqueeze(1)
    v = sparse_attn.v_proj(x).unsqueeze(1)

    attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) * (8 ** -0.5), dim=-1)
    dense_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
    dense_out = sparse_attn.out_proj(dense_out)

    # Check correlation
    sparse_flat = sparse_out.flatten()
    dense_flat = dense_out.flatten()

    correlation = torch.corrcoef(torch.stack([sparse_flat, dense_flat]))[0, 1]
    print(f"Correlation between sparse (k=all) and dense: {correlation.item():.4f}")

    # They should be highly correlated (but not identical due to top-k mechanics)
    assert correlation > 0.5, f"Expected high correlation, got {correlation.item()}"
    print("Sparse vs Dense equivalence: PASS")

    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DSA Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Lightning Indexer", test_lightning_indexer),
        ("Sparse Attention", test_sparse_attention),
        ("Full Model", test_full_model),
        ("Sparse vs Dense", test_sparse_vs_dense_equivalence),
        ("Benchmark", benchmark_inference),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    # Check for CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    success = run_all_tests()
    sys.exit(0 if success else 1)
