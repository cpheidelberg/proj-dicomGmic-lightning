import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle, h5py, time, sys, shutil, random
from tqdm import tqdm
from PIL import Image
import multiprocessing
import resource

import torch
from torch.utils.data import Dataset, DataLoader, Subset

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)
from src.data_loading import loading


class ClassificationImages(Dataset):

    def __init__(self, imageFolder:str, dictPath: str, top_c=None, h5_file=None):
        self.imageFolder = imageFolder
        self.imageFiles = [folder_path+file for folder_path in imageFolder for file in os.listdir(folder_path)]
        random.shuffle(self.imageFiles)
        self.flatDictDF = pd.read_csv(dictPath, converters={"best_center": ast.literal_eval, "finding_categories": ast.literal_eval})

        if top_c:
            self.filterCategories(top_c)

        self.unique_categories = list(self.flatDictDF["finding_categories"].explode().unique())
        print("{} classes: {}".format(len(self.unique_categories), self.unique_categories))
        print(self.flatDictDF["finding_categories"].value_counts())

        # print(self.getLabelCount())


    def __len__(self):
        return len(self.imageFiles)


    def __getitem__(self, idx):
        
        imagePath = self.imageFiles[idx]

        if len(self.imageFiles[idx].split("/")[-1].split("_")) > 2:
            original_file_name = "_".join(self.imageFiles[idx].split("/")[-1].split("_")[:-1]).strip(".png")
        else:
            original_file_name = self.imageFiles[idx].split("/")[-1].strip(".png")

        # original_file_name = self.getOrigFilename(idx)
        # data = self.getDataentry(original_file_name)
        # loaded_image = self.getImage(imagePath, data)

        data = self.flatDictDF[self.flatDictDF["image"] == original_file_name]
        if data.empty:
            raise ValueError(f"Original file name '{original_file_name}' not found in flatDict")

        view = data["view"].iloc[0]
        loaded_image = loading.load_image(
            image_path=imagePath,
            view=view,
            horizontal_flip=data["horizontal_flip"].iloc[0],
        )
        loaded_image = loading.process_image(loaded_image, view, data["best_center"].iloc[0][view][0])
        loaded_image = np.expand_dims(loaded_image, 0).copy()
        # loaded_image = torch.Tensor(loaded_image)

        labelList = data["finding_categories"].iloc[0]
        labelEnc = self.createHotEncoding(labelList)

        return loaded_image, labelEnc


    def getOrigFilename(self, idx):
        if len(self.imageFiles[idx].split("/")[-1].split("_")) > 2:
            original_file_name = "_".join(self.imageFiles[idx].split("/")[-1].split("_")[:-1]).strip(".png")
        else:
            original_file_name = self.imageFiles[idx].split("/")[-1].strip(".png")

        return original_file_name


    def getDataentry(self, original_file_name: str) -> pd.DataFrame:
        """Find the entry of the original filename in self.flatDict["image"]"""
        data = self.flatDictDF[self.flatDictDF["image"] == original_file_name]
        if data.empty:
            raise ValueError(f"Original file name '{original_file_name}' not found in flatDict")
        return data


    def getImage(self, imagePath: str, data: pd.DataFrame) -> torch.Tensor:

        view = data["view"].iloc[0]
        loaded_image = loading.load_image(
            image_path=imagePath,
            view=view,
            horizontal_flip=data["horizontal_flip"].iloc[0],
        )
        loaded_image = loading.process_image(loaded_image, view, data["best_center"].iloc[0][view][0])
        loaded_image = np.expand_dims(loaded_image, 0).copy()
        # loaded_image = torch.Tensor(loaded_image)

        return loaded_image

            
    def getLabelCount(self):
        labelCount = {k: 0 for k in self.unique_categories}
        print(labelCount)
        for f in self.imageFiles:
            
            if len(f.split("/")[-1].split("_")) > 2:
                original_file_name = "_".join(f.split("/")[-1].split("_")[:-1]).strip(".png")
            else:
                original_file_name = f.split("/")[-1].strip(".png")
            data = self.flatDictDF[self.flatDictDF["image"] == original_file_name]
            labelList = data["finding_categories"].iloc[0]
            labelCount[labelList[0]] += 1
        return labelCount


    def addLabelstoDF(self):
        for i, f in enumerate(tqdm(self.imageFiles)):
            f = f.split("/")[-1]
            entry = self.flatDictDF[self.flatDictDF["image"] == f.strip(".png")]
            index = self.flatDictDF.index[self.flatDictDF["image"] == f.strip(".png")].tolist()[0]

            studyID = entry["examID"].iloc[0]
            imageID = entry["dicom"].iloc[0].split('/')[-1]

            labelList = self.labels.loc[self.labels["study_id"] == studyID].loc[self.labels["image_id"] == imageID]["finding_categories"].iloc[0]

            self.flatDictDF["finding_categories"].iloc[index] = labelList


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
        imgDict["image"] = data[view][0]
        imgDict["dicom"] = data[view+"_path"]
        imgDict["horizontal_flip"] = data["horizontal_flip"]
        imgDict["best_center"] = data["best_center"]

        return imgDict


    def createHotEncoding(self, labelList):
        encoding = np.zeros(len(self.unique_categories), dtype=np.float32)
        for label in labelList:
            label_index = self.unique_categories.index(label)
            encoding[label_index] = 1.0
        if np.max(encoding) == 0:
            print("No label found")

        return encoding
            

    def convertLabels(self, df):
        column = "finding_categories"
        df[column] = df[column].apply(ast.literal_eval)

        return df


    def filterCategories(self, top_c=5):
        """Filter labels dataframe for top_c most occuring finding_categories and remove corresponding images from imageList"""

        class_counts = self.flatDictDF["finding_categories"].value_counts()
        origLen = len(self.imageFiles)
        origLength = len(self.flatDictDF)

        class_counts = class_counts.sort_values(ascending=False)
        top_labels = class_counts.head(top_c).index

        removed_images = self.flatDictDF.loc[~self.flatDictDF['finding_categories'].isin(top_labels)]
        removed_images = removed_images['image'].tolist()
        self.flatDictDF = self.flatDictDF[self.flatDictDF["finding_categories"].isin(top_labels)]
        self.flatDictDF.to_csv("removedTop5.csv")
        self.imageFiles = [f for f in self.imageFiles if "_".join(f.split("/")[-1].split("_")[:2]).strip(".png") not in removed_images]

        print("{}/{} images for {} retained categories".format(len(self.imageFiles), origLen, top_c))
        print("{}/{} dictionary entries for {} retained categories".format(len(self.flatDictDF), origLength, top_c))


class H5Dataset(Dataset):
    def __init__(self, h5_filepath, relevant_labels=None):
        self.valid_indices = []
        self.h5_filepath = h5_filepath
        self.relevant_labels = relevant_labels

        self.h5_file = h5py.File(self.h5_filepath, "r")
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']

        if relevant_labels:
            self.encoding = {'No Finding': 0, 'Mass': 1, 'Asymmetry': 2, 'Focal Asymmetry': 3, 'Suspicious Calcification': 4, 'Architectural Distortion': 5}
            self.relevant_label_indices = [self.encoding[label] for label in relevant_labels]
            self.filter_data()

        print(self.getLabelCount())

    def getLabelCount(self):
        labelCount = {k: 0 for k in range(len(self.encoding))}
        for l in self.labels:
            label_index = l.argmax(axis=0)
            if label_index in self.relevant_label_indices:
                labelCount[label_index] += 1
        return labelCount

    def filter_data(self):
        for i, label in enumerate(self.labels):
            label_index = label.argmax(axis=0)
            if label_index in self.relevant_label_indices:
                self.valid_indices.append(i)

    def __len__(self):
        if self.relevant_labels:
            return len(self.valid_indices)
        else:
            return len(self.labels)

    def __getitem__(self, idx):
        if self.relevant_labels:
            idx = self.valid_indices[idx]
        image = torch.from_numpy(self.images[idx].astype('float32'))
        label = torch.from_numpy(self.labels[idx].astype('float32'))

        print(label)
        if self.relevant_labels:
            label = label[self.relevant_label_indices]
        print(label)
        return image, label

    def close(self):
        self.h5_file.close()


def create_chunked_h5(data):

    num_workers = multiprocessing.cpu_count() - 2
    dataloader = DataLoader(data, shuffle=True, num_workers=num_workers)

    num_images = len(data) // 4
    image_shape = (1, 2944, 1920) 
    label_shape = (6,)

    image_chunk_size = (1, 1, 2944, 1920)
    label_chunk_size = (1, 6)

    start_time = time.time()
    with h5py.File('balanced_top6/dataset.h5', 'w') as f:
        # Create datasets for images and labels with appropriate chunk sizes
        dset_images = f.create_dataset('images', shape=(num_images,) + image_shape, dtype='float32', chunks=image_chunk_size)
        dset_labels = f.create_dataset('labels', shape=(num_images,) + label_shape, dtype='float32', chunks=label_chunk_size)

        for i, (image, label) in enumerate(tqdm(dataloader, total=num_images)):
            if i == num_images:
                break
            image_np = image.numpy()
            label_np = label.numpy()

            # Store in HDF5 dataset
            dset_images[i] = image_np
            dset_labels[i] = label_np

    print("Data has been successfully saved to 'balanced_top6/dataset.h5'")

    print(f"Gesamtzeit: {time.time() - start_time}")


def main(): 

    image_path_train = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_cropped_top5/'
    image_path_test = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images/'
    label_file = "sample_data/annotations/finding_annotations.csv"
    data_path = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/data.pkl'

    h5Path = "../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/balanced_top6/dataset.h5"

    # data = ClassificationImages(imageFolder=[image_path_train, image_path_test], top_c=6)
    # data = ClassificationFromLabels(imageFolder=[image_path_train, image_path_test], dictPath=data_path, labelPath=label_file, top_c=3)
    data = H5Dataset(h5_filepath=h5Path, relevant_labels=["No Finding", "Mass", "Suspicious Calcification"])
    # create_chunked_h5(data)

    print(data[0])


if __name__ == "__main__":
    main()