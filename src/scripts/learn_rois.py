import os
import cv2
import json
import argparse
import pickle
import cProfile
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from src.modeling import gmic as gmic
from src.data_loading import loading
from src.constants import VIEWS, PERCENT_T_DICT


class ActiveLearningGMIC(gmic.GMIC):
    def __init__(self, parameters):
        super(ActiveLearningGMIC, self).__init__(parameters)
        self.device = parameters["device_type"]


    def compute_loss(self, predicted_saliency_map, true_saliency_map):
        # Beispiel für den Mean Squared Error (MSE) als Verlustfunktion
        loss = F.mse_loss(predicted_saliency_map, true_saliency_map)
        return loss
    

    def active_learning_step(self, x_original, true_saliency_map, optimizer):
        # Umwandeln von x_original und true_saliency_map in PyTorch-Tensoren
        x_original_tensor = torch.Tensor(x_original).to(self.device)

        # TODO: Umwandeln von self.saliency_map von einer heat map in eine Contour oder mask
        true_saliency_map_tensor = torch.Tensor(true_saliency_map).to(self.device)

        # Vorwärtsdurchlauf
        self.forward(x_original_tensor)

        # Verlust berechnen
        loss = self.compute_loss(self.saliency_map, true_saliency_map_tensor)

        # Rückwärtsdurchlauf
        self.zero_grad()
        loss.backward()
        optimizer.step()

        return loss
    

    def save_model_weights(self, save_path):
        """Save learned weights in new file at 'save_path'"""
        torch.save(self.state_dict(), save_path)


def load_annotation_layer(json_path, height, width):

    with open(json_path, 'r') as json_file:
        data = json.load(json_file)

    # Erstellen Sie ein leeres Bild, auf dem Sie die Ellipsen zeichnen können
    image = np.zeros((height, width), dtype=np.uint8)  # Ersetzen Sie 'height' und 'width' durch die Abmessungen Ihres Bildes

    # Iterieren Sie über die Datensätze in der JSON-Datei
    for entry in data:
        # Holen Sie sich die Punkte, die die Ellipse definieren
        points = entry['points']
        
        # Extrahieren Sie die Parameter der Ellipse (Zentrum, Halbachsen und Winkel)
        x1, y1 = points[0]['x'], points[0]['y']
        x2, y2 = points[1]['x'], points[1]['y']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        major_axis = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2
        minor_axis = entry['radius'] / 2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Zeichnen Sie die Ellipse auf das Bild
        cv2.ellipse(image, (int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), angle, 0, 360, 255, 2)

    return image


def load_exam_list(path):
    with open(path, "rb") as f:
        exam_list = pickle.load(f)

    return exam_list


def run_active_learning(exam_list_path, model_path, json_path, model_index, parameters):

    parameters["percent_t"] = PERCENT_T_DICT[model_index]
    pretrained_model_path = os.path.join(model_path, "sample_model_{0}.p".format(model_index))
    model = ActiveLearningGMIC(parameters)
    print(f"Model loaded on device: {model.device}")

    # load parameters
    if parameters["device_type"] == "gpu":
        model.load_state_dict(torch.load(pretrained_model_path), strict=False)
    else:
        model.load_state_dict(torch.load(pretrained_model_path, map_location="cpu"), strict=False)
    print(f"Weights loaded from {pretrained_model_path}")

    exam_list = load_exam_list(exam_list_path)
    print("Exam list loaded from {}".format(exam_list_path))


    # iterate through each exam
    for datum in tqdm(exam_list):
        for view in VIEWS.LIST:
            short_file_path = datum[view][0]
            # load image
            # the image is already flipped so no need to do it again
            loaded_image = loading.read_image(os.path.join(parameters["image_path"], short_file_path + ".png"))
            loaded_image = loading.process_image(loaded_image, view, datum["horizontal_flip"], datum["best_center"][view][0])

            # convert python 2D array into 4D torch tensor in N,C,H,W format
            x_original = np.expand_dims(np.expand_dims(loaded_image, 0), 0).copy()
            x_height, x_width = x_original.shape[2], x_original.shape[3]

            true_saliency_map = load_annotation_layer(json_path, x_height, x_width)
            new_saliency_map = cv2.resize(true_saliency_map, (30, 46))
            new_saliency_array = np.array([new_saliency_map, new_saliency_map])

            model = ActiveLearningGMIC(parameters)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
            loss = model.active_learning_step(x_original, new_saliency_array, optimizer)
            print(loss)
    learned_path = os.path.join(model_path, "active_learning_model_{0}.p".format(model_index))
    model.save_model_weights(learned_path)
    print("Learned weights stored at {}".format(learned_path))


def main():

    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Run GMIC on the sample data')
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--exam-path', required=True)
    parser.add_argument('--image-path', required=True)
    parser.add_argument('--segmentation-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
    parser.add_argument("--gpu-number", type=int, default=0)
    parser.add_argument("--model-index", type=str, default="1")
    parser.add_argument('--profile-path', default=None, help="Enable cProfile profiling and specify the output path")

    args = parser.parse_args()

    # parameters copied from run_model.py
    parameters = {
        "device_type": args.device_type,
        "gpu_number": args.gpu_number,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": args.image_path,
        "segmentation_path": args.segmentation_path,
        "output_path": args.output_path,
        # model related hyper-parameters
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim":256,
        "num_classes":2,
        "use_v1_global":False,
    }


    model_path = "models"
    json_path = '/media/ayk/4644D1AB10BDC110/Medken/proj-dicomGmic-lightning-master/proj-dicomGmic-lightning/medken_feedback.json'
    exam_list_path = args.exam_path
    model_index=args.model_index

    if args.profile_path:
        print(exam_list_path)
        print(args.profile_path)
        profile_args = {
            "exam_list_path": exam_list_path,
            "model_path": model_path,
            "json_path": json_path,
            "model_index": model_index,
            "parameters": parameters
        }
        cProfile.runctx("run_active_learning(**profile_args)", globals(), locals(), args.profile_path)
    else:
        run_active_learning(exam_list_path, model_path, json_path, model_index, parameters)


if __name__ == "__main__":
    main()
