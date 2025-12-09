"""
Render TinyEfficientViTSeg with NSA model architecture using torchview.
"""

import torch
from torchview import draw_graph

from model import TinyEfficientViTSeg


def render_model(
    output_path: str = "model_architecture_nsa",
    input_size: tuple = (1, 1, 256, 256),
    depth: int = 3,
    expand_nested: bool = True,
    graph_dir: str = "TB",
    save_graph: bool = True,
    device: str = "cpu",
) -> None:
    """
    Render the TinyEfficientViTSeg with NSA model architecture to an image.

    Args:
        output_path: Output file path (without extension).
        input_size: Input tensor size (batch, channels, height, width).
        depth: Depth of nested modules to display.
        expand_nested: Whether to expand nested modules.
        graph_dir: Graph direction ('TB' for top-bottom, 'LR' for left-right).
        save_graph: Whether to save the graph to file.
        device: Device to use for model ('cpu' or 'cuda').
    """
    model = TinyEfficientViTSeg(
        in_channels=input_size[1],
        num_classes=2,
    )
    model = model.to(device)
    model.eval()

    model_graph = draw_graph(
        model,
        input_size=input_size,
        device=device,
        depth=depth,
        expand_nested=expand_nested,
        graph_dir=graph_dir,
        save_graph=save_graph,
        filename=output_path,
    )

    print(f"Model architecture (NSA) saved to {output_path}.png")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model_graph


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Render TinyEfficientViTSeg with NSA model architecture"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model_architecture_nsa",
        help="Output file path (without extension)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Input image height",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Input image width",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth of nested modules to display",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="TB",
        choices=["TB", "LR"],
        help="Graph direction (TB=top-bottom, LR=left-right)",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Do not expand nested modules",
    )

    args = parser.parse_args()

    render_model(
        output_path=args.output,
        input_size=(1, 1, args.height, args.width),
        depth=args.depth,
        expand_nested=not args.no_expand,
        graph_dir=args.direction,
    )
