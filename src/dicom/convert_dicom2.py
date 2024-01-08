import numpy as np
import pydicom as dcm
from pydicom.data import get_testdata_file

import argparse, pickle, os, cv2
from tqdm import tqdm
from multiprocessing import Pool

def getMetadata(dataset:dcm.FileDataset, idx:int) -> tuple[str, str]:
    viewPosition = dataset.ViewPosition
    breastOrientation = dataset.ImageLaterality
    horizontalFlip = "NO"

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


def savePng(dataset:dcm.FileDataset, outPath:str):
    image = dataset.pixel_array

    os.makedirs(os.path.dirname(outPath), exist_ok=True)
    cv2.imwrite(outPath, image)

    # print(f"\tPNG image saved to {outPath}")


def convertExam(dicomDir:str, dicomFile:str, idx:int, pngPath:str) -> dict:
    metaData = {}
    metaData["examID"] = dicomDir.split("/")[-1]
    # print(f"Read examID: {dicomDir.split("/")[-1]}")
    i = 0
    for root, dirs, files in os.walk(dicomDir):
        for file in files:
            if file == dicomFile:
                filePath = os.path.join(root, file)
                # print(f"\tRead DICOM file: {filePath}")
    
                ds = dcm.dcmread(filePath, force=True) # dicom dataset
                metaStr, flip = getMetadata(ds, i)
                metaData["horizontal_flip"] = flip
                metaData[metaStr] = [f"{idx}_{metaStr}"]
                metaData[f"{metaStr}_path"] = "/".join(filePath.split('/')[:-1])
                outPath = os.path.join(pngPath, f"{idx}_{metaStr}.png")
                savePng(ds, outPath)
                i += 1
        
    # print("\tAll DICOM files processed")
    metaData = addCancerLabel(metaData)
    return metaData


def convertExamStar(args):
    return convertExam(*args)


def convertList(dicomDir:str, dicomFile:str, pngPath:str=None, pklPath:str=None):
    """Walk through all DICOM files of an exam to get all required images for model input"""
    with Pool() as pool:
        folders = [d.path for d in os.scandir(dicomDir) if "." not in d.name]
        # folders = [d.path for d in os.scandir(dicomDir) if d.name.startswith("ff")]
        tasks = [(f, dicomFile, folders.index(f), pngPath) for f in folders]
        results = list(tqdm(pool.imap(convertExamStar, tasks), total=len(folders)))

    # TODO: save metadata in single pickle file
    savePickle(results, pklPath)
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