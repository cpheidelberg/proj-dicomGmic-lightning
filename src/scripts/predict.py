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
from src.data_loading import dataset


def visualize_example(img, saliency_maps, patches, patch_img, patch_attentions, save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, H, W = img.shape

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
    subfigure.imshow(img[0, 0, :, :], aspect='equal', cmap='gray')

    subfigure.set_title("input image")
    subfigure.axis('off')

    # patch map
    print(patches)
    subfigure = figure.add_subplot(1, total_num_subplots, 2)
    subfigure.imshow(img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.imshow(
        tools.get_crop_mask(patches[0, np.arange(parameters["K"]), :], parameters["crop_shape"], (H, W), "upper_left"),
        alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1],
    )

    subfigure.set_title("patch map")
    subfigure.axis('off')

    # class activation maps
    for i, class_name in zip(range(parameters["num_classes"]), parameters["class_names"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 3 + i)
        subfigure.imshow(img[0, 0, :, :], aspect='equal', cmap='gray')
        resized_cam = cv2.resize(saliency_maps[0,i,:,:], (W, H))

        subfigure.imshow(resized_cam, cmap=alpha_green if i == 0 else alpha_red, clim=[0.0, 1.0])

        subfigure.set_title("SM: " + class_name)
        subfigure.axis('off')

    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(1, total_num_subplots, 3 + parameters["num_classes"] + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest', aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))

    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()


def save_saliency_maps(img, saliency_maps, folder, filename, parameters):
    """Store saliency maps for benign and malignant tissue as separate layers and polylines"""

    img = img[0, 0, :, :]
    H, W = img.shape
    window_location = (0, H, 0, W)

    for i in range(parameters["num_classes"]):
        maps = cv2.resize((saliency_maps[0,i,:,:] * 500).astype(np.uint8), (W, H))
        process_saliency_map(img, maps, window_location, folder, filename, parameters["class_names"][i], parameters["turn_on_visualization"])


def process_saliency_map(input_img, saliency_map, window_location, folder, filename, label, turn_on_visualization):
    contours, _ = cv2.findContours(saliency_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    os.makedirs(folder, exist_ok=True)

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

        with open(os.path.join(folder, f"{filename}_polyline_{label}_{i}.txt"), 'w') as f:
            f.write(f"Saliency Map:\n")
            for point in polyline:
                f.write(f"{point[0]}, {point[1]}\n")
            f.write("---\n")
        
        if turn_on_visualization:
            image_with_contours = cv2.drawContours(saliency_map.copy(), [contour], -1, 255, 3)
            # plt.imshow(input_img, cmap='gray', aspect='equal')
            plt.imshow(image_with_contours, alpha=0.5, cmap="gray")
            print("Polyline saved to: ", os.path.join(folder, f"{filename}_seg_{label}_{i}.png"))
            plt.savefig(os.path.join(folder, f"{filename}_seg_{label}_{i}.png"))

    if not contours:
        print(filename, "\n\tNo contours found in the saliency map.")


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
    model_path = 'tb_logs/awsTest/version_6/checkpoints/epoch=7-step=7920.ckpt'

    data_path = '/home/ubuntu/sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output'
    image_dir = '/home/ubuntu/gmic/vindrmammo_data'
    output_path = '/home/ubuntu/gmic/predict_output'
    segmentation_path = os.path.join(output_path, 'segmentation')

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
