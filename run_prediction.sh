#!/bin/bash

NUM_PROCESSES=10
DEVICE_TYPE='cpu'
GPU_NUMBER=0
MODEL_INDEX='2'

MODEL_PATH='models/'
DICOM_FOLDER='../Data/BSC-DBT'
DICOM_FILE='1-1.dcm'
DATA_FOLDER='../Data/BSC-DBT/images'
INITIAL_EXAM_LIST_PATH='../Data/BSC-DBT/exam_list.pkl'
CROPPED_IMAGE_PATH='../Data/BSC-DBT/output/cropped_images'
CROPPED_EXAM_LIST_PATH='../Data/BSC-DBT/output/cropped_images/cropped_exam_list.pkl'
SEG_PATH='../Data/BSC-DBT/output/segmentation'
EXAM_LIST_PATH='../Data/BSC-DBT/output/data.pkl'
OUTPUT_PATH='../Data/BSC-DBT/output'

export PYTHONPATH=$(pwd):$PYTHONPATH

# echo 'Stage 1: Convert DICOM exams'
# python3 -m cProfile -o convert.prof src/dicom/convert_dicom.py \
#     --dicom-data-folder $DICOM_FOLDER \
#     --dicom-file $DICOM_FILE \
#     --image-data-folder $DATA_FOLDER \
#     --exam-list-path $INITIAL_EXAM_LIST_PATH

# echo 'Stage 2: Crop Mammograms'
# python3 src/cropping/crop_mammogram.py \
#     --input-data-folder $DATA_FOLDER \
#     --output-data-folder $CROPPED_IMAGE_PATH \
#     --exam-list-path $INITIAL_EXAM_LIST_PATH  \
#     --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
#     --num-processes $NUM_PROCESSES

# echo 'Stage 3: Extract Centers'
# python3 src/optimal_centers/get_optimal_centers.py \
#     --cropped-exam-list-path $CROPPED_EXAM_LIST_PATH \
#     --data-prefix $CROPPED_IMAGE_PATH \
#     --output-exam-list-path $EXAM_LIST_PATH \
#     --num-processes $NUM_PROCESSES

# echo 'Stage 4: Run Classifier'
# python3 -m cProfile -o model.prof src/scripts/run_model.py \
#     --model-path $MODEL_PATH \
#     --data-path $EXAM_LIST_PATH \
#     --image-path $CROPPED_IMAGE_PATH \
#     --segmentation-path $SEG_PATH \
#     --output-path $OUTPUT_PATH \
#     --device-type $DEVICE_TYPE \
#     --gpu-number $GPU_NUMBER \
#     --model-index $MODEL_INDEX \
#     --visualization-flag

echo 'Stage 5: Convert DICOM exams'
python3 src/dicom/create_dicomsr.py \
    --segmentation-path $SEG_PATH \
    --dicom-file $DICOM_FILE \
    --exam-list-path $INITIAL_EXAM_LIST_PATH \