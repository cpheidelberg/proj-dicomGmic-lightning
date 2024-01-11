import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle, h5py
from tqdm import tqdm
from PIL import Image
import multiprocessing

import torch
from torch.utils.data import Dataset
from torchvision import transforms
    

class ClassificationImages(Dataset):

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str):
        self.imageFolder = imageFolder
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
        return len(self.flatDict)


    def __getitem__(self, idx):
        
        imagePath = os.path.join(self.imageFolder, self.flatDict[idx]["image"][0]+".png")
        image = Image.open(imagePath)
        plt.imshow(np.array(image)/np.max(np.array(image)), cmap="gray")
        plt.savefig("tmp.png")

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
    def __init__(self, dataFile):
        self.dataFile = dataFile
        with h5py.File(self.dataFile, "r") as hf:
            self.num_samples = len(hf.keys())


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        with h5py.File(self.dataFile, "r") as hf:
            image = hf[f"image_{idx}"][:]
            label = hf[f"label_{idx}"][:]

        return image, label


def save_data_to_hdf5(dataset, hdf5_file):

    # Create a pool of worker processes
    num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_workers)
    chunk_size = len(dataset) // num_workers


    with tqdm(total=len(dataset)) as pbar:
        args_list = [(hdf5_file, dataset, i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
        results = pool.starmap(save_chunk_to_hdf5, args_list)
        pool.close()
        pool.join()


def save_chunk_to_hdf5(hdf5_file, dataset, start_idx, end_idx):
    with h5py.File(hdf5_file, "a") as hf:
        for idx in tqdm(range(start_idx, end_idx)):
            image_name = f"image_{idx}"
            label_name = f"label_{idx}"

            # Load and preprocess your image
            image = dataset[idx][0][:]
            label = dataset[idx][1][:]

            hf.create_dataset(image_name, data=image)
            hf.create_dataset(label_name, data=label)


# with h5py.File(hdf5_file, "w") as hf:
#     for idx, (image, labelEnc) in enumerate(tqdm(dataset)):
#         if idx == 10:
#             break
#         image_name = f"image_{idx}"
#         hf.create_dataset(image_name, data=image.numpy())
#         label_name = f"label_{idx}"
#         hf.create_dataset(label_name, data=labelEnc)


def main(): 
    imageFolder = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images'
    imageDict = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/exam_list.pkl"
    labelFile = "sample_data/annotations/finding_annotations.csv"

    dataset = ClassificationImages(imageFolder, imageDict, labelFile)

    dataPath = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/"
    hdf5_file = os.path.join(dataPath, "dataset.h5")
    
    save_data_to_hdf5(dataset, hdf5_file)

    # data = HDF5Dataset(hdf5_file)
    # print(data[0])


if __name__ == "__main__":
    main()

    plt.show()