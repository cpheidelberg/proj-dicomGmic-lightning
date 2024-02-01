import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle, h5py, time, sys
from tqdm import tqdm
from PIL import Image
import multiprocessing

import torch
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)
from src.data_loading import loading


class ClassificationImages(Dataset):

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str):
        self.imageFolder = imageFolder
        self.imageFiles = os.listdir(imageFolder)
        self.imageDict = self.loadPickle(dictPath)
        self.flatDict = self.flattenDict(self.imageDict)

        self.labels = pd.read_csv(labelPath)
        self.filtered_categories = self.filterCategories(top_c=5)
        self.labels = self.convertLabels(self.labels)
        self.unique_categories = list(set([label for labels in self.labels["finding_categories"] for label in labels]))

        print("Number of classes: {}".format(len(self.unique_categories)))

        # TODO: why more labels than images? -> multiple findings


    def __len__(self):
        return len(self.imageFiles)


    def __getitem__(self, idx):
        
        # file path 1_L-CC
        # load image from imagePath
        imagePath = os.path.join(self.imageFolder, self.imageFiles[idx])
        data = self.flatDict[idx]
        view = data["view"]

        loaded_image = loading.load_image(
            image_path=imagePath,
            view=view,
            horizontal_flip=data["horizontal_flip"],
        )
        loaded_image = loading.process_image(loaded_image, view, data["best_center"][view][0])
        loaded_image = np.expand_dims(loaded_image, 0).copy()
        image = torch.Tensor(loaded_image)

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
        imgDict["horizontal_flip"] = data["horizontal_flip"]
        imgDict["best_center"] = data["best_center"]

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


    def filterCategories(self, top_c=5):
        """Filter labels dataframe for top_c most occuring finding_categories and remove corresponding images from imageList"""

        class_counts = self.labels["finding_categories"].value_counts()
        origLen = len(self.imageFiles)

        class_counts = class_counts.sort_values(ascending=False)
        top_labels = class_counts.head(top_c).index

        filtered_labels = self.labels[self.labels['finding_categories'].apply(lambda x: any(label in x for label in top_labels))]
        removed_labels = self.labels[~self.labels.index.isin(filtered_labels.index)]

        self.labels = filtered_labels

        removed_exams = list(set(removed_labels["study_id"].to_list())) # get unique study IDs from removed_labels

        print(len(self.flatDict))
        for study_id in removed_exams:
            view = [d["image"][0] for d in self.flatDict if d["examID"] == study_id]
            # TODO: remove element from flatDict
            self.flatDict = [d for d in self.flatDict if d["examID"] != study_id]
            # print(self.imageFiles.index[view[0]+".png"])
            for v in view:
                if v+".png" in self.imageFiles:
                    self.imageFiles.remove(v+".png")
            # break
            
            # if exam_id is not None:
            #     view = self.flatDict["view"].get(exam_id, None)
            #     if view is not None:
            #         # Identifizieren und Löschen der Einträge in self.imageFiles
            #         self.imageFiles = [file for file in self.imageFiles if not (file['examID'] == exam_id and file['view'] == view)]
        print(len(self.flatDict))

        newLen = len(self.imageFiles)
        print("{}/{} images for {} retained categories".format(newLen, origLen, top_c))


class HDF5Dataset(Dataset):
    def __init__(self, dataFolder, batchSize=32):
        self.dataFolder = dataFolder
        self.batchSize = batchSize
        self.dataFiles = [f for f in sorted(os.listdir(dataFolder))]


    def __len__(self):
        return len(self.dataFiles) * self.batchSize


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