import os
import sys
import ast
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.modeling.trainer import GMICTrainer
from src.data_loading import loading
from src.data_loading import dataset
from src.scripts import predict


def load_config():
    return {
        "device_type": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_number": 0,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "pretrained": False,
        "fine-tuning": False,
        "smote_rate": 0.0,
        "turn_on_visualization": True,
        "output_path": "output/single_image",
        "segmentation_path": "output/single_image/segmentation",
        "learning_rate": 1e-3,
        "regularization": 1e-4,
        "batch_size": 1,
        "undersampling_rate": 0.0,
        "augmentation_rate": 0.0,
        "binary": True,
        "augment": False,
        "epoch_smote": 256,
        "model_idx": 1,
    }


def parse_center(center_str):
    return ast.literal_eval(center_str)


def preprocess_image(image_path, view, center_str, parameters):
    img = loading.read_image(image_path, 'float32')
    center = parse_center(center_str) if center_str else (img.shape[0] // 2, img.shape[1] // 2)
    processed = loading.process_image(img, view, 'NO', center)
    return processed


def run_single_image_prediction(model, image_path, view, label, center_str, parameters):
    processed_img = preprocess_image(image_path, view, center_str, parameters)

    device = parameters['device_type']
    img_tensor = torch.from_numpy(processed_img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        predictions = output[0] if isinstance(output, tuple) else output
        print(f"Predictions: {predictions}")

    saliency_maps = model.gmic.saliency_map.data.cpu().numpy()
    patch_locations = model.gmic.patch_locations
    patch_imgs = model.gmic.patches
    patch_attentions = model.gmic.patch_attns[0, :].data.cpu().numpy()

    return processed_img, saliency_maps, patch_locations, patch_imgs, patch_attentions


def main():
    parser = argparse.ArgumentParser(description='Run GMIC on a single image')
    parser.add_argument('--image-index', type=int, default=0,
                        help='Index of the image in mapping.csv to process')
    parser.add_argument('--checkpoint-path', type=str,
                        default='test_models/epoch=63-step=6208.ckpt',
                        help='Path to the .ckpt model file')
    parser.add_argument('--data-dir', type=str,
                        default='~/sdsHD/sd24f004/FFDM/demd/extracted',
                        help='Directory containing the extracted images and mapping.csv')
    parser.add_argument('--output-dir', type=str,
                        default='output/single_image',
                        help='Directory to save visualization outputs')
    args = parser.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    mapping_path = os.path.join(data_dir, 'mapping.csv')

    if not os.path.exists(mapping_path):
        print(f"Error: mapping.csv not found at {mapping_path}")
        return

    parameters = load_config()
    parameters['output_path'] = args.output_dir
    parameters['segmentation_path'] = os.path.join(args.output_dir, 'segmentation')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(parameters['segmentation_path'], exist_ok=True)

    print(f"Loading mapping from {mapping_path}...")
    mapping = pd.read_csv(mapping_path)

    if args.image_index >= len(mapping):
        print(f"Error: image-index {args.image_index} out of range (max: {len(mapping) - 1})")
        return

    row = mapping.iloc[args.image_index]
    png_relative_path = row['png']
    image_path = os.path.join(data_dir, png_relative_path)
    view = row['view']
    label = row['label']
    center_str = row['center'] if pd.notna(row['center']) else None

    print(f"Processing image {args.image_index}:")
    print(f"  Path: {image_path}")
    print(f"  View: {view}")
    print(f"  Label: {label}")
    print(f"  Center: {center_str}")

    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return

    print(f"Loading model from {args.checkpoint_path}...")
    model = GMICTrainer(parameters)
    checkpoint = torch.load(args.checkpoint_path, map_location=parameters['device_type'])
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state, strict=False)
    model.to(parameters['device_type'])
    model.eval()
    print("Model loaded successfully")

    print("Preprocessing image and running prediction...")
    input_img, saliency_maps, patch_locations, patch_imgs, patch_attentions = \
        run_single_image_prediction(model, image_path, view, label, center_str, parameters)

    input_img_expanded = np.expand_dims(np.expand_dims(input_img, 0), 0)
    seg_masks = [None, None]

    save_path = os.path.join(args.output_dir, f"visualization_{args.image_index}.png")
    os.makedirs(os.path.join(args.output_dir, "visualization"), exist_ok=True)

    print(f"Creating visualization...")
    parameters['class_names'] = ['benign', 'malignant']

    figure = predict.visualize_example(
        input_img_expanded,
        [png_relative_path],
        saliency_maps,
        seg_masks,
        patch_locations,
        patch_imgs,
        patch_attentions,
        parameters,
        save_path=save_path
    )

    print(f"Visualization saved to {save_path}")

    predict.save_saliency_maps(
        input_img_expanded,
        saliency_maps,
        parameters['segmentation_path'],
        png_relative_path.replace('/', '_').replace('.png', ''),
        parameters
    )

    print(f"Saliency maps saved to {parameters['segmentation_path']}")

    plt.figure(figsize=(8, 6))
    plt.imshow(input_img, cmap='gray', aspect='equal')
    plt.title(f"Input Image\nView: {view}, Label: {label}")
    plt.axis('off')
    input_save_path = os.path.join(args.output_dir, f"input_{args.image_index}.png")
    plt.savefig(input_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Input image saved to {input_save_path}")


if __name__ == "__main__":
    main()
