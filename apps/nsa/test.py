#!/usr/bin/env python3
"""
Test script for NSA Pupil Segmentation model.

This script validates the NSAPupilSeg model architecture and components,
ensuring all parts work correctly together.
"""

import torch
import torch.nn as nn
import time
import sys

from nsa import (
    NSAPupilSeg,
    NSABlock,
    SpatialNSA,
    TokenCompression,
    TokenSelection,
    SlidingWindowAttention,
    CombinedLoss,
    create_nsa_pupil_seg,
)


def test_token_compression():
    """Test TokenCompression module."""
    print(
        "\n[1/7] Testing TokenCompression..."
    )

    batch_size = 2
    H, W = 8, 8  # spatial dimensions
    seq_len = H * W  # 64
    dim = 32
    block_size = 4
    stride = 2

    compression = TokenCompression(
        dim=dim,
        block_size=block_size,
        stride=stride,
    )

    k = torch.randn(
        batch_size, seq_len, dim
    )
    v = torch.randn(
        batch_size, seq_len, dim
    )

    k_cmp, v_cmp = compression(
        k, v, (H, W)
    )

    print(f"  Input K shape: {k.shape}")
    print(
        f"  Compressed K shape: {k_cmp.shape}"
    )
    print(f"  Input V shape: {v.shape}")
    print(
        f"  Compressed V shape: {v_cmp.shape}"
    )

    # Verify shapes
    assert k_cmp.shape[0] == batch_size
    assert k_cmp.shape[2] == dim
    assert v_cmp.shape[0] == batch_size
    assert v_cmp.shape[2] == dim

    print(
        "  ✓ TokenCompression test passed!"
    )
    return True


def test_sliding_window_attention():
    """Test SlidingWindowAttention module."""
    print(
        "\n[2/7] Testing SlidingWindowAttention..."
    )

    batch_size = 2
    dim = 32
    H, W = 32, 32
    window_size = 7
    num_heads = 2

    swa = SlidingWindowAttention(
        dim=dim,
        num_heads=num_heads,
        window_size=window_size,
    )

    x = torch.randn(
        batch_size, dim, H, W
    )
    out = swa(x)

    print(f"  Input shape: {x.shape}")
    print(
        f"  Output shape: {out.shape}"
    )

    assert out.shape == x.shape
    print(
        "  ✓ SlidingWindowAttention test passed!"
    )
    return True


def test_spatial_nsa():
    """Test SpatialNSA (Native Sparse Attention) module."""
    print(
        "\n[3/7] Testing SpatialNSA..."
    )

    batch_size = 2
    dim = 32
    H, W = 32, 32
    num_heads = 2

    nsa = SpatialNSA(
        dim=dim,
        num_heads=num_heads,
        compress_block_size=4,
        compress_stride=2,
        select_block_size=4,
        num_select=4,
        window_size=7,
    )

    x = torch.randn(
        batch_size, dim, H, W
    )
    out = nsa(x)

    print(f"  Input shape: {x.shape}")
    print(
        f"  Output shape: {out.shape}"
    )

    assert out.shape == x.shape
    print("  ✓ SpatialNSA test passed!")
    return True


def test_nsa_block():
    """Test NSABlock module."""
    print("\n[4/7] Testing NSABlock...")

    batch_size = 2
    dim = 32
    H, W = 32, 32

    block = NSABlock(
        dim=dim,
        num_heads=2,
        mlp_ratio=2.0,
        compress_block_size=4,
        compress_stride=2,
        select_block_size=4,
        num_select=4,
        window_size=7,
    )

    x = torch.randn(
        batch_size, dim, H, W
    )
    out = block(x)

    print(f"  Input shape: {x.shape}")
    print(
        f"  Output shape: {out.shape}"
    )

    assert out.shape == x.shape
    print("  ✓ NSABlock test passed!")
    return True


def test_nsa_pupil_seg():
    """Test complete NSAPupilSeg model."""
    print(
        "\n[5/7] Testing NSAPupilSeg model..."
    )

    batch_size = 2
    in_channels = 1
    num_classes = 2
    H, W = (
        400,
        640,
    )  # OpenEDS image size

    model = create_nsa_pupil_seg(
        size="small"
    )
    n_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"  Model parameters: {n_params:,}"
    )

    x = torch.randn(
        batch_size, in_channels, H, W
    )
    out = model(x)

    print(f"  Input shape: {x.shape}")
    print(
        f"  Output shape: {out.shape}"
    )

    expected_shape = (
        batch_size,
        num_classes,
        H,
        W,
    )
    assert (
        out.shape == expected_shape
    ), f"Expected {expected_shape}, got {out.shape}"

    print(
        "  ✓ NSAPupilSeg test passed!"
    )
    return True


def test_combined_loss():
    """Test CombinedLoss function."""
    print(
        "\n[6/7] Testing CombinedLoss..."
    )

    batch_size = 2
    num_classes = 2
    H, W = 100, 160

    criterion = CombinedLoss()

    logits = torch.randn(
        batch_size, num_classes, H, W
    )
    target = torch.randint(
        0,
        num_classes,
        (batch_size, H, W),
    )
    spatial_weights = torch.rand(
        batch_size, H, W
    )
    dist_map = torch.rand(
        batch_size, num_classes, H, W
    )
    alpha = 0.5

    (
        total_loss,
        ce_loss,
        dice_loss,
        surface_loss,
    ) = criterion(
        logits,
        target,
        spatial_weights,
        dist_map,
        alpha,
    )

    print(
        f"  Total loss: {total_loss.item():.4f}"
    )
    print(
        f"  CE loss: {ce_loss.item():.4f}"
    )
    print(
        f"  Dice loss: {dice_loss.item():.4f}"
    )
    print(
        f"  Surface loss: {surface_loss.item():.4f}"
    )

    assert not torch.isnan(total_loss)
    assert not torch.isinf(total_loss)

    print(
        "  ✓ CombinedLoss test passed!"
    )
    return True


def test_model_sizes():
    """Test all model size configurations."""
    print(
        "\n[7/7] Testing all model sizes..."
    )

    H, W = (
        200,
        320,
    )  # Smaller for faster testing
    x = torch.randn(1, 1, H, W)

    for size in [
        "tiny",
        "small",
        "medium",
    ]:
        print(
            f"\n  Testing {size} model..."
        )
        model = create_nsa_pupil_seg(
            size=size
        )
        n_params = sum(
            p.numel()
            for p in model.parameters()
        )

        model.eval()
        with torch.no_grad():
            out = model(x)

        print(
            f"    Parameters: {n_params:,}"
        )
        print(
            f"    Output shape: {out.shape}"
        )

        assert out.shape == (1, 2, H, W)

    print(
        "\n  ✓ All model sizes test passed!"
    )
    return True


def benchmark_inference():
    """Benchmark inference speed."""
    print("\n" + "=" * 60)
    print("Inference Benchmark")
    print("=" * 60)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    H, W = 400, 640
    x = torch.randn(1, 1, H, W).to(
        device
    )

    for size in [
        "tiny",
        "small",
        "medium",
    ]:
        model = create_nsa_pupil_seg(
            size=size
        ).to(device)
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        n_runs = 10
        start = time.time()
        with torch.no_grad():
            for _ in range(n_runs):
                _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.time() - start

        avg_time = (
            elapsed / n_runs * 1000
        )  # ms
        fps = n_runs / elapsed

        n_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print(
            f"\n{size.upper()} model:"
        )
        print(
            f"  Parameters: {n_params:,}"
        )
        print(
            f"  Avg inference time: {avg_time:.2f} ms"
        )
        print(f"  FPS: {fps:.1f}")


def main():
    """Run all tests."""
    print("=" * 60)
    print(
        "NSA Pupil Segmentation Model Tests"
    )
    print("=" * 60)

    tests = [
        (
            "Token Compression",
            test_token_compression,
        ),
        (
            "Sliding Window Attention",
            test_sliding_window_attention,
        ),
        (
            "Spatial NSA",
            test_spatial_nsa,
        ),
        ("NSA Block", test_nsa_block),
        (
            "NSA Pupil Seg Model",
            test_nsa_pupil_seg,
        ),
        (
            "Combined Loss",
            test_combined_loss,
        ),
        (
            "Model Sizes",
            test_model_sizes,
        ),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append(
                (name, passed)
            )
        except Exception as e:
            print(
                f"  ✗ {name} FAILED: {e}"
            )
            results.append(
                (name, False)
            )

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = (
            "✓ PASSED"
            if passed
            else "✗ FAILED"
        )
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n✓ All tests passed!")

        # Run benchmark if all tests pass
        benchmark_inference()

        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
