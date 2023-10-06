import numpy as np
import matplotlib.pyplot as plt
import pydicom as dcm
from pydicom.data import get_testdata_file

import sys, pickle, os, cv2

# replace this path with path to GMIC
sys.path.append("./GMIC/")

from src.data_loading import loading
from src.modeling import gmic as gmic


def getMetadata(dataset:dcm.FileDataset, idx:int) -> tuple[str, str]:
    viewPosition = dataset.ViewPosition
    # breastOrientation = 'R' if 'R' in dataset.PatientOrientation[1] else 'L'
    if idx > 1:
        breastOrientation = 'R' 
    else:
        breastOrientation = 'L'
    horizontalFlip = "NO" if dataset.XRay3DAcquisitionSequence[0].FieldOfViewHorizontalFlip[0] == "N" else "YES"

    filename = f"{breastOrientation}-{viewPosition}"
    return filename, horizontalFlip


def addCancerLabel(metaData:dict) -> dict:
    metaData['cancer_label'] = {
        'benign': 0, 'right_benign': 0, 'malignant': 0, 'left_benign': 0, 'unknown': 0, 'right_malignant': 0, 'left_malignant': 0
    }
    return metaData


def savePickle(metaData:dict, outPath:str=None):
    with open(outPath, 'wb') as file:
        pickle.dump(metaData, file)


def savePng(dataset:dcm.FileDataset, idx:int, outPath:str):
    image = dataset.pixel_array[idx]
    if 'R' in dataset.PatientOrientation[1]:
        image = np.rot90(image, 2)

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    cv2.imwrite(outPath, image)

    print(f"\tPNG image from slice {idx} saved to {outPath}")


def convertExam(examDir:str, idx:int, pngPath:str, pklPath:str) -> dict:
    metaData = {}
    examDir = os.path.join(examDir, [dir for dir in os.listdir(examDir) if os.path.isdir(os.path.join(examDir, dir))][0])

    i = 0
    for root, dirs, files in os.walk(examDir):
        for file in files:
            if file == inPath:
                filePath = os.path.join(root, file)
                print(f"\tRead DICOM file: {filePath}")
    
                ds = dcm.dcmread(filePath, force=True) # dicom dataset
                metaStr, flip = getMetadata(ds, i)
                metaData["horizontal_flip"] = flip
                metaData[metaStr] = [f"{idx}_{metaStr}"]
                outPath = os.path.join(pngPath, f"{idx}_{metaStr}.png")
                sliceID = len(ds.pixel_array) // 2
                savePng(ds, sliceID, outPath)
                i += 1
        
    print("\tAll DICOM files processed")
    metaData = addCancerLabel(metaData)
    return metaData


def main(examDir:str, pngPath:str=None, pklPath:str=None):
    """Walk through all DICOM files of an exam to get all required images for model input"""

    metaData = []
    for idx, examFolder in enumerate(os.scandir(examDir)):
        if examFolder.name.startswith("DBT-P"):
            print(examFolder.path)
            metaData.append(convertExam(examFolder.path, idx, pngPath, pklPath))

    # TODO: save metadata in single pickle file
    savePickle(metaData, examDir+"/"+pklPath)
    print(f"\tPickle file saved: {examDir}/{pklPath}")


if __name__ == "__main__":

    examDir = "../Data/BSC-DBT"
    pngPath = os.path.join(examDir, "images")
    inPath = "1-1.dcm"
    pklPath = "exam_list.pkl"

    main(examDir, pngPath, pklPath)