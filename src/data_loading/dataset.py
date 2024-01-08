import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle
from PIL import Image

import torch
from torch.utils.data import Dataset
    

class ClassificationImages(Dataset):

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str):
        self.imageFolder = imageFolder
        self.imageDict = self.loadPickle(dictPath)
        self.flatDict = self.flattenDict(self.imageDict)

        self.labels = pd.read_csv(labelPath)
        self.labels = self.convertLabels(self.labels)

        print(len(self.flatDict))
        print(len(self.labels))

        # TODO: why more labels than images? -> multiple findings


    def __len__(self):
        return len(self.flatDict)


    def __getitem__(self, idx):
        
        imagePath = os.path.join(self.imageFolder, self.flatDict[idx]["image"][0]+".png")
        image = Image.open(imagePath)
        studyID = self.flatDict[idx]["examID"]
        imageID = self.flatDict[idx]["dicom"].split('/')[-1]

        labelList = self.labels[self.labels["study_id"] == studyID][self.labels["image_id"] == imageID]["finding_categories"].iloc[0]
        # metadata from preprocessing in self.imageDict[self.flatDict.iloc[idx]["examID"]]
        return image, labelList


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
        
            

    def convertLabels(self, df):
        column = "finding_categories"
        df[column] = df[column].apply(ast.literal_eval)
        # df_single = df.explode(column, ignore_index=True)

        return df


def main(): 
    imageFolder = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/images"
    imageDict = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/exam_list.pkl"
    labelFile = "sample_data/annotations/finding_annotations.csv"

    dataset = ClassificationImages(imageFolder, imageDict, labelFile)
    img, label = dataset[0]

    img = img/np.max(img)
    plt.title(label)
    plt.imshow(img, cmap="gray")


if __name__ == "__main__":
    main()

    plt.show()