"""Compose readable 20k/40k Lantern endpoint previews from verified outputs."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OLD = (
    ROOT
    / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted/outputs/remote"
    / "c4-render-ablation-20k-v1/37477/Lantern"
)
NEW = (
    ROOT
    / "outputs/analysis/c4-render-ablation-lantern-40k-job-37581/extracted/outputs/remote"
    / "c4-render-ablation-lantern-40k-v1/37581/Lantern"
)
OUT = ROOT / "outputs/analysis/c4-render-ablation-lantern-40k-job-37581/report/figures"
SUMMARY_20K = (
    ROOT
    / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted/outputs/remote"
    / "c4-render-ablation-20k-v1/37477/Lantern-summary/endpoint_technical.png"
)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    font_root = Path(windows_dir) / "Fonts"
    candidates = [font_root / "segoeui.ttf", font_root / "arial.ttf"]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def fixed(arm: str, step: int) -> Image.Image:
    base = OLD if step == 20_000 else NEW
    folder = f"step_{step:05d}"
    return Image.open(base / arm / "observations" / folder / "fixed_views.png").convert("RGB")


def endpoint_comparison() -> Path:
    tile_w, tile_h = 1024, 280
    margin, gap, header, label_h, footer = 34, 28, 105, 34, 74
    width = margin * 2 + tile_w * 2 + gap
    height = header + (label_h + tile_h) * 3 + footer + margin
    canvas = Image.new("RGB", (width, height), "#f4f4f1")
    draw = ImageDraw.Draw(canvas)
    title_font, heading_font, body_font = font(30), font(22), font(17)
    draw.text((margin, 20), "Lantern endpoint preview — fixed views", fill="#171717", font=title_font)

    arms = [
        ("material_only", "Material only", "HDR MAE 4.674e-3 → 4.272e-3  (-8.61%)"),
        ("material_render", "Material + render", "HDR MAE 5.060e-3 → 4.171e-3  (-17.56%)"),
    ]
    for column, (arm, heading, metric) in enumerate(arms):
        x = margin + column * (tile_w + gap)
        draw.text((x, 65), heading, fill="#171717", font=heading_font)
        image_20 = fixed(arm, 20_000)
        image_40 = fixed(arm, 40_000)
        difference = ImageChops.difference(image_20, image_40)
        difference = ImageEnhance.Contrast(difference).enhance(8.0)
        for row, (label, image) in enumerate(
            [("20k", image_20), ("40k", image_40), ("|20k − 40k| ×8", difference)]
        ):
            y = header + label_h + row * (label_h + tile_h)
            draw.text((x, y - label_h + 7), label, fill="#545454", font=body_font)
            canvas.paste(image, (x, y))
        draw.text((x, height - footer + 15), metric, fill="#333333", font=body_font)

    draw.text(
        (margin, height - 28),
        "Same four fixed cameras. Difference row is display-space absolute difference amplified 8×; it is diagnostic, not an HDR metric.",
        fill="#666666",
        font=font(15),
    )
    destination = OUT / "lantern_20k_40k_endpoint_preview.png"
    canvas.save(destination, optimize=True)
    return destination


def endpoint_40k() -> Path:
    tile_w, tile_h = 1024, 280
    margin, gap, header, label_h, footer = 34, 24, 70, 34, 68
    width = margin * 2 + tile_w
    height = header + (label_h + tile_h) * 3 + footer
    canvas = Image.new("RGB", (width, height), "#f4f4f1")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Lantern 40k — arm-to-arm preview", fill="#171717", font=font(30))
    only = fixed("material_only", 40_000)
    render = fixed("material_render", 40_000)
    difference = ImageEnhance.Contrast(ImageChops.difference(only, render)).enhance(12.0)
    for row, (label, image) in enumerate(
        [("Material only", only), ("Material + render", render), ("|arm difference| ×12", difference)]
    ):
        y = header + label_h + row * (label_h + tile_h)
        draw.text((margin, y - label_h + 7), label, fill="#454545", font=font(18))
        canvas.paste(image, (margin, y))
    draw.text(
        (margin, height - footer + 15),
        "40k audit: render arm HDR MAE is 2.36% lower; BaseColor MAE is 1.34% higher. Visual difference is subtle.",
        fill="#333333",
        font=font(16),
    )
    destination = OUT / "lantern_40k_arm_preview.png"
    canvas.save(destination, optimize=True)
    return destination


def source_pca_40k(arm: str) -> Path:
    if arm not in {"material_only", "material_render"}:
        raise ValueError(f"unknown arm: {arm}")
    technical = Image.open(SUMMARY_20K).convert("RGB")
    endpoint = fixed(arm, 40_000)
    views = ("front", "rear", "upper_side", "top")
    tile = 256
    margin, gap, header, row_label, footer = 30, 18, 94, 27, 58
    width = margin * 2 + tile * 3 + gap * 2
    height = header + len(views) * (row_label + tile) + footer
    canvas = Image.new("RGB", (width, height), "#141417")
    draw = ImageDraw.Draw(canvas)
    title = "Material only @ 40k" if arm == "material_only" else "Material + render @ 40k"
    draw.text((margin, 16), f"Lantern — Source / Raw PCA / {title}", fill="#f2f2f2", font=font(27))
    columns = ("Source", "Raw PCA (raw_q4)", title)
    for column, label in enumerate(columns):
        x = margin + column * (tile + gap)
        draw.text((x, 59), label, fill="#d7d7d7", font=font(16))
    for row, view in enumerate(views):
        y = header + row * (row_label + tile)
        draw.text((margin, y + 4), view, fill="#a9a9ad", font=font(15))
        image_y = y + row_label
        source = technical.crop((0, row * 310 + 54, tile, row * 310 + 54 + tile))
        raw = technical.crop((tile, row * 310 + 54, tile * 2, row * 310 + 54 + tile))
        trained = endpoint.crop((row * tile, 24, (row + 1) * tile, 24 + tile))
        for column, image in enumerate((source, raw, trained)):
            x = margin + column * (tile + gap)
            canvas.paste(image, (x, image_y))
    metric = (
        "40k audit HDR MAE 0.004272 · display SSIM 0.970615"
        if arm == "material_only"
        else "40k audit HDR MAE 0.004171 · display SSIM 0.971311"
    )
    draw.text((margin, height - footer + 17), metric, fill="#d7d7d7", font=font(16))
    destination = OUT / f"lantern_source_rawpca_{arm}_40k.png"
    canvas.save(destination, optimize=True)
    return destination


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (
        endpoint_comparison(),
        endpoint_40k(),
        source_pca_40k("material_only"),
        source_pca_40k("material_render"),
    ):
        print(path)


if __name__ == "__main__":
    main()
