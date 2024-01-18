import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle, h5py, time
from tqdm import tqdm
from PIL import Image
import multiprocessing

import torch
from torch.utils.data import Dataset
from torchvision import transforms
    

class ClassificationImages(Dataset):

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str):
        self.imageFolder = imageFolder
        self.imageFiles = os.listdir(imageFolder)
        self.imageDict = self.loadPickle(dictPath)
        self.flatDict = self.flattenDict(self.imageDict)

        self.labels = pd.read_csv(labelPath)
        self.labels = self.convertLabels(self.labels)
        self.unique_categories = list(set([label for labels in self.labels["finding_categories"] for label in labels]))

        self.transform = transforms.Compose([
            transforms.Resize((2944, 1920)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize((0.5), (0.5)),
        ])

        # TODO: why more labels than images? -> multiple findings


    def __len__(self):
        return len(self.imageFiles)


    def __getitem__(self, idx):
        
        imagePath = os.path.join(self.imageFolder, self.imageFiles[idx])
        image = Image.open(imagePath)

        image = self.transform(image)

        studyID = self.flatDict[idx]["examID"]
        imageID = self.flatDict[idx]["dicom"].split('/')[-1]

        labelList = self.labels.loc[self.labels["study_id"] == studyID].loc[self.labels["image_id"] == imageID]["finding_categories"].iloc[0]
        # metadata from preprocessing in self.imageDict[self.flatDict.iloc[idx]["examID"]]
        labelEnc = self.createHotEncoding(labelList)

        return image, labelEnc


    def loadPickle(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        return data

    def flattenDict(self, data):
        flatDict = []
        for d in data:
            dictLCC = self.extractImageDict(d, "L-CC")
            dictRCC = self.extractImageDict(d, "R-CC")
            dictLMLO = self.extractImageDict(d, "L-MLO")
            dictRMLO = self.extractImageDict(d, "R-MLO")
            flatDict.append(dictLCC)
            flatDict.append(dictRCC)
            flatDict.append(dictLMLO)
            flatDict.append(dictRMLO)

        return flatDict 

    
    def extractImageDict(self, data, view):
        imgDict = {}
        imgDict["examID"] = data["examID"]
        imgDict["view"] = view
        imgDict["image"] = data[view]
        imgDict["dicom"] = data[view+"_path"]

        return imgDict


    def createHotEncoding(self, labelList):
        encoding = np.zeros(len(self.unique_categories), dtype=np.float32)

        for label in labelList:
            label_index = self.unique_categories.index(label)
            encoding[label_index] = 1.0

        return encoding
            

    def convertLabels(self, df):
        column = "finding_categories"
        df[column] = df[column].apply(ast.literal_eval)
        # df_single = df.explode(column, ignore_index=True)

        return df


class HDF5Dataset(Dataset):
    def __init__(self, dataFolder, batchSize=32):
        self.dataFolder = dataFolder
        self.batchSize = batchSize
        self.dataFiles = [f for f in sorted(os.listdir(dataFolder))]


    def __len__(self):
        return 1000 #len(self.dataFiles) * self.batchSize


    def __getitem__(self, idx):
        batchId = idx // self.batchSize
        dataFile = os.path.join(self.dataFolder, self.dataFiles[batchId])
        with h5py.File(dataFile, "r") as hf:
            image = hf[f"image_{idx}"][:]
            label = hf[f"label_{idx}"][:]

        return image, label


def save_data_to_hdf5(h5Path, dataset):
    totalSize = len(dataset)
    batchSize = 32
    numBatches = totalSize // batchSize
    num_workers = multiprocessing.cpu_count()

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        args = [(h5Path, dataset, batchId, batchSize) for batchId in range(numBatches)]
        results = list(tqdm(pool.imap(save_batch_to_hdf5_star, args), total=len(args)))


def save_batch_to_hdf5_star(args):
    return save_batch_to_hdf5(*args)


def save_batch_to_hdf5(h5Path, dataset, batchId, batchSize):
    startId = batchId * batchSize
    endId = (batchId + 1) * batchSize
    
    start = time.time()
    h5File = os.path.join(h5Path, f"study_{batchId}.h5")
    with h5py.File(h5File, "w") as hf:
        for i in range(startId, endId):
            nameImage = f"image_{i}"
            nameLabel = f"label_{i}"
            setImage = hf.create_dataset(nameImage, data=dataset[startId+i][0])
            setLabel = hf.create_dataset(nameLabel, data=dataset[startId+i][1])

    stop = time.time()
    print("Writing file {}: {:.1g}s".format(batchId, stop - start))


def main(): 
    imageFolder = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images'
    imageDict = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/exam_list.pkl"
    labelFile = "sample_data/annotations/finding_annotations.csv"

    dataset = ClassificationImages(imageFolder, imageDict, labelFile)

    dataPath = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images_pickle"
    
    save_data_to_hdf5(dataPath, dataset)
    # with multiprocessing.Pool(12) as pool:
    #     args = [(dataPath, d, i) for i,d in enumerate(dataset)]
    #     results = list(tqdm(pool.imap(saveItemStar, args), total=len(args)))

    # data = HDF5Dataset(hdf5File)
    # print(data[0])


if __name__ == "__main__":
    main()