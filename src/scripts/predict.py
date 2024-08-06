import os, cv2, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch.utils.data import random_split
import lightning.pytorch as pl

# import own files 
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)

from src.utilities import tools
from src.modeling import gmic
from src.data import dataset


def visualize_example(input_img, saliency_maps, seg_masks,
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    _, _, H, W = input_img.shape

    # set up colormaps for benign and malignant
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas

    # create visualization template
    total_num_subplots = 2 + parameters["num_classes"] + parameters["K"]
    figure = plt.figure(figsize=(40, 3))

    # input image
    subfigure = figure.add_subplot(1, total_num_subplots, 1)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    
    for idx, seg_mask in enumerate(seg_masks):
        if seg_mask is not None:
            if idx == 0:
                subfigure.imshow(seg_mask, alpha=0.85, cmap=alpha_green, clim=[0.9, 1])
            else:
                subfigure.imshow(seg_mask, alpha=0.85, cmap=alpha_red, clim=[0.9, 1])

    subfigure.set_title("input image")
    subfigure.axis('off')

    # patch map
    print(patch_locations)
    subfigure = figure.add_subplot(1, total_num_subplots, 2)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(tools.get_crop_mask(
        patch_locations[0, np.arange(parameters["K"]), :],
        parameters["crop_shape"], (H, W),
        "upper_left"), alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])

    for seg_mask in seg_masks:
        if seg_mask is not None:
            subfigure.imshow(seg_mask, alpha=0.85, cmap="Reds", clim=[0.9, 1])

    subfigure.set_title("patch map")
    subfigure.axis('off')

    # class activation maps
    for idx, class_name in enumerate(parameters["class_names"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 3 + idx)
        subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
        resized_cam = cv2.resize(saliency_maps[0, idx, :, :], (W, H))
        if idx == 0: # "No Finding"
            subfigure.imshow(resized_cam, cmap=alpha_green, clim=[0.0, 1.0])
        else:
            subfigure.imshow(resized_cam, cmap=alpha_red, clim=[0.0, 1.0])
        subfigure.set_title("SM: " + class_name)
        subfigure.axis('off')

    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 3 + parameters["num_classes"] + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    
    print(save_dir)
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()


def save_saliency_maps(input_img, saliency_maps, datum, save_dir, file_path, turn_on_visualization):
    """Store saliency maps for benign and malignant tissue as separate layers and polylines"""

    input_img = input_img[0, 0, :, :]
    H, W = input_img.shape
    view = file_path.split('_')[1].split('.')[0]
    window_location = datum["window_location"][0][view][0]

    saliency_maps_benign = (saliency_maps[0,0,:,:]*500).astype(np.uint8)
    saliency_maps_benign = cv2.resize(saliency_maps_benign, (W, H))
    saliency_maps_malignant = (saliency_maps[0,1,:,:]*500).astype(np.uint8)
    saliency_maps_malignant = cv2.resize(saliency_maps_malignant, (W, H))

    process_saliency_map(input_img, saliency_maps_benign, window_location, save_dir, file_path, "benign", turn_on_visualization)
    process_saliency_map(input_img, saliency_maps_malignant, window_location, save_dir, file_path, "malignant", turn_on_visualization)


def process_saliency_map(input_img, saliency_map, window_location, save_dir, file_path, label, turn_on_visualization):
    contours, _ = cv2.findContours(saliency_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    os.makedirs(save_dir, exist_ok=True)

    max_intensity = np.max(saliency_map)
    intensity_threshold = 0.1 * max_intensity  # 10% of the maximum intensity
    top_contours = []  # List to store the top three contours, their intensities, and centroids
    _, img_mask = cv2.threshold(input_img, 0, 255, cv2.THRESH_BINARY)

    for contour in contours:
        # Compute the average intensity of pixels within the contour
        mask = np.zeros_like(saliency_map)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        intensity = np.mean(saliency_map[mask == 255])

        # Check if the current contour has higher intensity than the threshold
        if intensity > intensity_threshold:
            # Calculate the centroid of the current contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Check if the centroid is within the non-black region of the input image
                if img_mask[cY, cX] != 0:  # Assuming black regions have pixel value 0
                    # Check if there are fewer than three contours in top_contours
                    # or if the current contour has a higher intensity than the lowest intensity in the top three
                    if len(top_contours) < 3 or intensity > top_contours[-1][1]:
                        # Add the current contour, intensity, and centroid to the list
                        top_contours.append((contour, intensity, (cX, cY)))
                        # Sort the top_contours list based on intensities (highest to lowest)
                        top_contours.sort(key=lambda x: x[1], reverse=True)
                        # Keep only the top three contours
                        top_contours = top_contours[:3]

    for i, (contour, intensity, centroid) in enumerate(top_contours):
        polyline = [point[0].tolist() for point in contour]
        for p in polyline:
            p[0] -= window_location[2]
            p[1] -= window_location[0]

        with open(os.path.join(save_dir, "{0}_polyline_{1}_{2}.txt".format(file_path, label, i)), 'w') as f:
            f.write(f"Saliency Map:\n")
            for point in polyline:
                f.write(f"{point[0]}, {point[1]}\n")
            f.write("---\n")
        
        if turn_on_visualization:
            image_with_contours = cv2.drawContours(saliency_map.copy(), [contour], -1, 255, 3)
            # plt.imshow(input_img, cmap='gray', aspect='equal')
            plt.imshow(image_with_contours, alpha=0.5, cmap="gray")
            print("Polyline saved to: {}".format(os.path.join(save_dir, "{}_seg_{}_{}.png".format(file_path, label, i))))
            plt.savefig(os.path.join(save_dir, "{0}_seg_{1}_{2}.png".format(file_path, label, i)))

    if not contours:
        print(file_path, "\n\tNo contours found in the saliency map.")


if __name__ == "__main__":
    # check if GPU is available
    if torch.cuda.is_available():
        print(f"{torch.cuda.device_count()} GPUs are available")
        device = "gpu"
    elif torch.backends.mps.is_available():
        print("Apple MPS is available")
        device = "mps"
    else: 
        device = "cpu"

    # set path variables
    model_path = 'models/sample_model_1.p'

    data_path = '/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = '/home/ubuntu/gmic/vindrmammo_data'
    segmentation_path = os.path.join(data_path, 'segmentation')
    output_path = '/home/ubuntu/gmic/predict_output'

    # set hyperparameters
    parameters = {
        # training related hyper-parameters
        "device_type": device,
        "gpu_number": 0,
        "batch_size": 1,
        "pretrained": True,

        "smote_rate": 0.0,

        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": image_dir,
        "segmentation_path": segmentation_path,
        "output_path": output_path,
        "turn_on_visualization": True,

        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6, # num patches
        "crop_shape": (512, 512), # patch size
        "percent_t": 0.03,
        "post_processing_dim": 256,
        "num_classes": 2, # output classes
        "use_v1_global": False,
    }

    data = dataset.ClassificationImages([image_dir], undersampling_rate=0.0, augmentation_rate=0.0, binary=True, augment=False)
    parameters["class_names"] = data.labels

    ds_train, ds_valid, ds_test = random_split(data, [0.8, 0.1, 0.1])

    # Training
    gmic_module = gmic.GMIC(parameters, data.class_weights(), dataset_predict=ds_train, model_path=model_path)

    trainer = pl.Trainer(fast_dev_run=True, accelerator=device, devices=[parameters["gpu_number"]])
    prediction = trainer.predict(gmic_module)

    print(f"Categories: {data.labels}")
    print(f"Prediction: {prediction}")
