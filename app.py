import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam_vit_b_01ec64.pth",
        help="Path to SAM checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model type"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device: cpu or cuda"
    )
    return parser.parse_args()


args = parse_args()

print("Loading SAM model...")
sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
sam.to(device=args.device)
print("Model loaded.")


def create_mask_generator(
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    min_mask_region_area
):
    return SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=int(points_per_side),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        min_mask_region_area=int(min_mask_region_area),
    )


def overlay_masks(image_np, masks):
    """
    Create a colorful overlay image.
    """
    if len(masks) == 0:
        return image_np

    overlay = image_np.copy().astype(np.float32)
    rng = np.random.default_rng(42)

    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    for ann in masks_sorted:
        mask = ann["segmentation"]
        color = rng.integers(0, 255, size=3)
        overlay[mask] = overlay[mask] * 0.55 + color * 0.45

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_boxes_and_ids(image_np, masks, top_k):
    """
    Draw bounding boxes and mask IDs for easier visual inspection.
    """
    result = image_np.copy()
    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    for idx, ann in enumerate(masks_sorted[:top_k]):
        x, y, w, h = ann["bbox"]
        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(
            result,
            str(idx),
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return result


def generate_mask_gallery(image_np, masks, top_k):
    """
    Generate individual mask images for gallery display.
    """
    gallery = []
    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    for idx, ann in enumerate(masks_sorted[:top_k]):
        mask = ann["segmentation"]

        masked_image = image_np.copy()
        background = np.ones_like(masked_image) * 255
        result = np.where(mask[:, :, None], masked_image, background)

        pil_img = Image.fromarray(result.astype(np.uint8))
        gallery.append((pil_img, f"Mask {idx}, Area = {ann['area']}"))

    return gallery


def run_sam_ui(
    input_image,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    min_mask_region_area,
    top_k
):
    if input_image is None:
        return None, None, [], "Please upload an image first."

    image_np = np.array(input_image.convert("RGB"))

    mask_generator = create_mask_generator(
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )

    masks = mask_generator.generate(image_np)
    masks_sorted = sorted(masks, key=lambda x: x["area"], reverse=True)

    overlay = overlay_masks(image_np, masks_sorted)
    boxed = draw_boxes_and_ids(overlay, masks_sorted, int(top_k))
    gallery = generate_mask_gallery(image_np, masks_sorted, int(top_k))

    output_dir = Path("outputs/ui_result")
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = output_dir / "sam_overlay_result.png"
    boxed_path = output_dir / "sam_boxed_result.png"

    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(boxed).save(boxed_path)

    if len(masks_sorted) > 0:
        max_area = masks_sorted[0]["area"]
        total_mask_area = sum([m["area"] for m in masks_sorted])
        image_area = image_np.shape[0] * image_np.shape[1]
        area_ratio = total_mask_area / image_area
    else:
        max_area = 0
        area_ratio = 0

    info = (
        f"Detected masks: {len(masks_sorted)}\n"
        f"Largest mask area: {max_area}\n"
        f"Total mask area / image area: {area_ratio:.2f}\n"
        f"Saved overlay to: {overlay_path}\n"
        f"Saved boxed result to: {boxed_path}"
    )

    return Image.fromarray(overlay), Image.fromarray(boxed), gallery, info


custom_css = """
/* Global background */
.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 32%),
        radial-gradient(circle at top right, rgba(168, 85, 247, 0.14), transparent 30%),
        linear-gradient(135deg, #020617 0%, #0f172a 45%, #111827 100%) !important;
    color: #e5e7eb !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

/* Main layout */
#main-container {
    max-width: 1280px;
    margin: 0 auto;
}

/* Header block */
#hero-section {
    padding: 28px 34px;
    margin-bottom: 22px;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.78));
    border: 1px solid rgba(148, 163, 184, 0.22);
    box-shadow: 0 24px 80px rgba(0, 0, 0, 0.38);
    backdrop-filter: blur(18px);
}

#hero-section h1 {
    margin-bottom: 8px;
    font-size: 42px;
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: -0.04em;
    background: linear-gradient(90deg, #e0f2fe 0%, #93c5fd 38%, #c4b5fd 72%, #f9a8d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

#hero-section p {
    color: #cbd5e1;
    font-size: 16px;
    line-height: 1.7;
    max-width: 820px;
}

/* Cards */
.panel {
    padding: 22px !important;
    border-radius: 22px !important;
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.20) !important;
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.30) !important;
    backdrop-filter: blur(16px);
}

/* Component containers */
.block, .form, .gr-box, .gr-panel {
    border-radius: 18px !important;
    border-color: rgba(148, 163, 184, 0.22) !important;
    background: rgba(15, 23, 42, 0.56) !important;
}

/* Labels */
label, .label-wrap span {
    color: #dbeafe !important;
    font-weight: 650 !important;
    letter-spacing: 0.01em;
}

/* Textbox */
textarea, input {
    background: rgba(2, 6, 23, 0.72) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(148, 163, 184, 0.24) !important;
    border-radius: 14px !important;
}

/* Slider */
input[type="range"] {
    accent-color: #60a5fa !important;
}

/* Button */
#run-button {
    border: none !important;
    border-radius: 16px !important;
    padding: 14px 20px !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%) !important;
    box-shadow: 0 16px 42px rgba(37, 99, 235, 0.38) !important;
    transition: all 0.22s ease !important;
}

#run-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 56px rgba(124, 58, 237, 0.44) !important;
    filter: brightness(1.08);
}

/* Images */
.image-container, .thumbnail-item, .gallery {
    border-radius: 18px !important;
    overflow: hidden !important;
}

/* Markdown titles inside panels */
.section-title h2 {
    color: #f8fafc !important;
    font-size: 20px !important;
    margin-bottom: 12px !important;
}

/* Footer note */
#footer-note {
    margin-top: 18px;
    padding: 14px 18px;
    border-radius: 16px;
    color: #94a3b8;
    background: rgba(2, 6, 23, 0.38);
    border: 1px solid rgba(148, 163, 184, 0.14);
    font-size: 13px;
}
"""


with gr.Blocks(
    title="SAM Interactive Segmentation UI",
    css=custom_css
) as demo:
    with gr.Column(elem_id="main-container"):
        gr.Markdown(
            """
            <div id="hero-section">
                <h1>SAM Interactive Segmentation UI</h1>
                <p>
                    Upload an image, adjust automatic mask generation parameters, and inspect segmentation results through
                    overlay visualization, mask IDs, individual mask previews, and quantitative summary information.
                </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    """
                    <div class="section-title">
                        <h2>Input and Parameters</h2>
                    </div>
                    """
                )

                input_image = gr.Image(
                    label="Input Image",
                    type="pil"
                )

                points_per_side = gr.Slider(
                    minimum=8,
                    maximum=64,
                    value=32,
                    step=8,
                    label="Sampling Density: points_per_side"
                )

                pred_iou_thresh = gr.Slider(
                    minimum=0.50,
                    maximum=1.00,
                    value=0.88,
                    step=0.01,
                    label="Predicted IoU Threshold: pred_iou_thresh"
                )

                stability_score_thresh = gr.Slider(
                    minimum=0.50,
                    maximum=1.00,
                    value=0.95,
                    step=0.01,
                    label="Stability Score Threshold: stability_score_thresh"
                )

                min_mask_region_area = gr.Slider(
                    minimum=0,
                    maximum=5000,
                    value=100,
                    step=50,
                    label="Minimum Mask Region Area: min_mask_region_area"
                )

                top_k = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=10,
                    step=1,
                    label="Number of Masks to Display"
                )

                run_button = gr.Button(
                    "Run Segmentation",
                    elem_id="run-button"
                )

            with gr.Column(elem_classes=["panel"]):
                gr.Markdown(
                    """
                    <div class="section-title">
                        <h2>Segmentation Results</h2>
                    </div>
                    """
                )

                overlay_output = gr.Image(label="Colored Mask Overlay")
                boxed_output = gr.Image(label="Overlay with Mask IDs")
                info_output = gr.Textbox(
                    label="Segmentation Summary",
                    lines=6
                )

        with gr.Column(elem_classes=["panel"]):
            gr.Markdown(
                """
                <div class="section-title">
                    <h2>Individual Mask Gallery</h2>
                </div>
                """
            )

            mask_gallery = gr.Gallery(
                label="Individual Masks",
                columns=5,
                height="auto"
            )

        gr.Markdown(
            """
            <div id="footer-note">
                This interface is designed for interactive inspection of SAM automatic mask generation results.
                It improves usability, visualization, and failure case analysis without modifying the core SAM algorithm.
            </div>
            """
        )

    run_button.click(
        fn=run_sam_ui,
        inputs=[
            input_image,
            points_per_side,
            pred_iou_thresh,
            stability_score_thresh,
            min_mask_region_area,
            top_k,
        ],
        outputs=[
            overlay_output,
            boxed_output,
            mask_gallery,
            info_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()