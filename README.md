# Adapted version of the forked GMIC project classifying DICOM files and creating DICOM SR description files with ROIs

## Introduction
This is an implementation of the Globally-Aware Multiple Instance Classifier (GMIC) model as described in [our paper](https://arxiv.org/abs/2002.07613).  
The implementation allows users to obtain breast cancer predictions and visualization of saliency maps by applying one of our pretrained models. We provide weights for 5 GMIC-ResNet-18 models. The model is implemented in PyTorch. 

Code functionality:
* Required `PNG` images are extracted from input `DICOM` files from `$DICOM_FOLDER / $DICOM_FILE`
* Both views (`CC`, `MLO`) are required for left and right resulting in four `PNG` images (`L-CC`, `L-MLO`, `R-CC`, `R-MLO`) and DICOM files have to contain metadata accordingly. As a part of this repository, 4 sample exams are provided with 4 DICOM files each in `sample_data/dicom-exams` directory list. 
* Output: The GMIC model generates one prediction for each image: probability of benign and malignant findings. All predictions are saved into a csv file `$OUTPUT_PATH/predictions.csv` that contains the following columns: image_index, benign_pred, malignant_pred, benign_label, malignant_label. In addition, each input image is associated with a visualization file saved under `$OUTPUT_PATH/visualization`. An exemplar visualization file is illustrated below. The images (from left to right) represent:
  * input mammography with ground truth annotation (green=benign, red=malignant),
  * patch map that illustrates the locations of ROI proposal patches (blue squares),
  * saliency map for benign class,
  * saliency map for malignant class,
  * 6 ROI proposal patches with the associated attention score on top.
  
![alt text](https://github.com/nyukat/GMIC/blob/master/sample_data/sample_visualization.png)

* The saliency maps for benign and malignant regions of interest (ROIs) are exported as `TXT` files under `$OUTPUT_PATH/segmentation`. These are also converted into `DICOM SR` files next to the original `DICOM` files under `$DICOM_FOLDER` linking to the original files as well. 
* Active learning is implemented comparing predicted heat maps for each region of interest with annotated images where ellipses mark each true ROI (**not fully tested yet**)


## Prerequisites

* Python
* PyTorch
* torchvision
* NumPy
* SciPy
* H5py
* imageio
* pandas
* opencv-python
* tqdm
* matplotlib
* pydicom
* highdicom


## Execute a script

You need to first install conda in your environment. **Before running the code, please run `pip install -r requirements.txt` first.** Once you have installed all the dependencies, `run_prediction.sh` will automatically run the entire prediction pipeline for all exams and save the prediction results in csv. In addition, visualizations will be saved as `PNG` if the flag  Note that you need to first cd to the project directory and then execute `./run_prediction.sh`. When running the individual Python scripts, please include the path to this repository in your `PYTHONPATH`. 

We recommend running the code with a GPU. To run the code with CPU only, please change `DEVICE_TYPE` in run.sh to 'cpu'. 

The following variables defined in `run.sh` can be modified as needed:
* `MODEL_PATH`: The path where the model weights are saved.
* `DICOM_FOLDER`: The path to all studies which have to be analyzed
* `DICOM_FILE`: The file name of registered `DICOM` files
* `DATA_FOLDER`: The path of all converted `PNG` images loaded by the model.
* `CROPPED_IMAGE_PATH`: The directory where cropped mammograms are saved.
* `SEG_PATH`: The directory where ground truth segmenations are saved.
* `EXAM_LIST_PATH`: The path where the exam list is stored.
* `OUTPUT_PATH`: The path where visualization files and predictions will be saved.
* `DEVICE_TYPE`: Device type to use in heatmap generation and classifiers, either 'cpu' or 'gpu'.
* `GPU_NUMBER`: GPUs number multiple GPUs are available.
* `MODEL_INDEX`: Which one of the five models to use. Valid values include {'1', '2', '3', '4', '5','ensemble'}.
* `visualization-flag`: Whether to generate visualization.


### Data

`sample_data/dicom_exams` contains 4 exams each of which includes 4 the original mammography images (L-CC, L-MLO, R-CC, R-MLO). 
The file name and the concatenation relies on the `DICOM` metadata `ViewPosition` and `PatientOrientation`. Thus, it has to be ensured that they are correctly stored.

For supervised learning, an annotated `JSON` file is used with ellipses of regions of interest. Each `DICOM` file can have two corresponding annotation layers for benign and malignant regions. 
They are stored beside the original files under `$DICOM_FOLDER`.

### Prediction

In order to predict a mammography exam, values for `L-CC`, `R-CC`, `L-MLO`, and `R-MLO` are concatenated as a list of image filenames without extensions and directory name saved in a list of dictionaries in a `pickle` file.

Following steps are used: 
1. Convert DICOM exams
2. Crop Mammograms
3. Extract Centers
4. Run Classifier
5. Create DICOM SR exams

### Active learning routine

The active learning enables a continuous improvement of the model and the integration of unseen mammography prediction results. 
A pipeline was implemented and can be executed with `./run_learning.sh`. 
If not run before, the preprocessing and predicting routine have to be executed in order to get the necessary file links via the `EXAM_LIST` dictionary and predicted heat maps for the ROIs.

## Acknowledges

The adopted code heavily relies on the model developed by Geras et al.:

**An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization**\
Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu, Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho and Krzysztof J. Geras\
Medical Image Analysis
2020
    
    @article{shen2020interpretable, 
    title={An interpretable classifier for high-resolution breast cancer screening images utilizing weakly supervised localization},
    author={Shen, Yiqiu and Wu, Nan and Phang, Jason and Park, Jungkyu and Liu, Kangning and Tyagi, Sudarshini and Heacock, Laura and Kim, S Gene and Moy, Linda and Cho, Kyunghyun and others},
    journal={Medical Image Analysis},
    pages={101908},
    year={2020},
    publisher={Elsevier}
}