import numpy as np
import pydicom as dcm
from pydicom.data import get_testdata_file

import argparse, pickle, os, cv2


def getMetadata(dataset:dcm.FileDataset, idx:int) -> tuple[str, str]:
    viewPosition = dataset.ViewPosition
    breastOrientation = 'R' if 'R' in dataset.PatientOrientation[1] else 'L'
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


def convertExam(dicomDir:os.DirEntry, dicomFile:str, idx:int, pngPath:str) -> dict:
    metaData = {}
    metaData["examID"] = dicomDir.name
    dicomDir = os.path.join(dicomDir.path, [dir for dir in os.listdir(dicomDir.path) if os.path.isdir(os.path.join(dicomDir.path, dir))][0])
    i = 0
    for root, dirs, files in os.walk(dicomDir):
        for file in files:
            if file == dicomFile:
                filePath = os.path.join(root, file)
                print(f"\tRead DICOM file: {filePath}")
    
                ds = dcm.dcmread(filePath, force=True) # dicom dataset
                metaStr, flip = getMetadata(ds, i)
                metaData["horizontal_flip"] = flip
                metaData[metaStr] = [f"{idx}_{metaStr}"]
                metaData[f"{metaStr}_path"] = "/".join(filePath.split('/')[:-1])
                outPath = os.path.join(pngPath, f"{idx}_{metaStr}.png")
                sliceID = len(ds.pixel_array) // 2
                savePng(ds, sliceID, outPath)
                i += 1
        
    print("\tAll DICOM files processed")
    metaData = addCancerLabel(metaData)
    return metaData

def convertList(dicomDir:str, dicomFile:str, pngPath:str=None, pklPath:str=None):
    """Walk through all DICOM files of an exam to get all required images for model input"""

    metaData = []
    for idx, dicomFolder in enumerate(os.scandir(dicomDir)):
        if dicomFolder.name.startswith("DBT-P"):
            metaData.append(convertExam(dicomFolder, dicomFile, idx, pngPath))

    # TODO: save metadata in single pickle file
    savePickle(metaData, pklPath)
    print(f"\tPickle file saved: {pklPath}")


def main():
    # retrieve command line arguments
    parser = argparse.ArgumentParser(description='Extract model input from DICOM data')
    parser.add_argument('--dicom-data-folder', required=True)
    parser.add_argument('--dicom-file', required=True)
    parser.add_argument('--exam-list-path', required=True)
    parser.add_argument('--image-data-folder', required=True)
    args = parser.parse_args()

    dicomDir = args.dicom_data_folder
    dicomFile = args.dicom_file
    pklPath = args.exam_list_path
    pngPath = args.image_data_folder

    convertList(dicomDir, dicomFile, pngPath, pklPath)


if __name__ == "__main__":
    main()