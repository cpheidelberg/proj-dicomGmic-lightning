import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, ast, pickle, h5py, time, sys, shutil
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

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str, top_c=None, h5_file=None):
        self.imageFolder = imageFolder
        self.imageFiles = os.listdir(imageFolder)
        self.imageDict = self.loadPickle(dictPath)
        self.flatDict = self.flattenDict(self.imageDict)

        self.labels = pd.read_csv(labelPath)
        if top_c:
            self.filterCategories(top_c)
        self.labels = self.convertLabels(self.labels)
        self.unique_categories = list(set([label for labels in self.labels["finding_categories"] for label in labels]))
        print("Number of classes: {}".format(len(self.unique_categories)))
        self.category_dict = dict.fromkeys(self.unique_categories, 0)

        self.h5_file = h5_file
        if self.h5_file:
            self.create_h5_dataset()

        # save image_label pairs
        # self.label_counts = self.labels["finding_categories"].value_counts()
        # self.max_count = self.label_counts.max()
        # self.require_dict = {k[0]: (self.max_count - v) // v for k,v in zip(self.label_counts.keys(), self.label_counts.values)}


    def __len__(self):
        return len(self.imageFiles)


    def __getitem__(self, idx):
        
        imagePath = os.path.join(self.imageFolder, self.imageFiles[idx])

        if len(self.imageFiles[idx].split("_")) > 2:
            original_file_name = "_".join(self.imageFiles[idx].split("_")[:-1]).strip(".png")
        else:
            original_file_name = self.imageFiles[idx].strip(".png")
        # Find the entry of the original filename in self.flatDict["image"]
        data = None
        # index = None
        for i, entry in enumerate(self.flatDict):
            if entry["image"][0] == original_file_name:
                studyID = entry["examID"]
                imageID = entry["dicom"].split('/')[-1]
                data = entry
                # index = i
                break
        if data is None:
            raise ValueError(f"Original file name '{original_file_name}' not found in flatDict")
        view = data["view"]

        loaded_image = loading.load_image(
            image_path=imagePath,
            view=view,
            horizontal_flip=data["horizontal_flip"],
        )
        loaded_image = loading.process_image(loaded_image, view, data["best_center"][view][0])
        loaded_image = np.expand_dims(loaded_image, 0).copy()
        image = torch.Tensor(loaded_image)

        labelList = self.labels.loc[self.labels["study_id"] == studyID].loc[self.labels["image_id"] == imageID]["finding_categories"].iloc[0]
        labelEnc = self.createHotEncoding(labelList)

        if self.h5_file:
            self.save_to_h5(image, labelEnc, idx)

        self.category_dict[labelList[0]] += 1
        print(self.category_dict)

        # save image label pairs
        # req = self.require_dict[labelList[0]]
        # print(f"For label {labelList[0]} at image {original_file_name}, {req} copies are required")
        # for i in range(req):
        #     newPath = "/".join(self.imageFolder.split("/")[:-2]) + "/balanced_cropped_top5/" + os.path.splitext(self.imageFiles[idx])[0] + "_" + str(i) + ".png"
        #     shutil.copy(imagePath, newPath)
        # self.flatDict[i]["finding_categories"] = labelList

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

    
    def create_h5_dataset(self):
        with h5py.File(self.h5_file, 'w') as hf:
            hf.create_dataset('images', (len(self.imageFiles), 2944, 1920), dtype='i')
            hf.create_dataset('labels', (len(self.imageFiles), len(self.unique_categories)), dtype='i')


    def save_to_h5(self, image, label, idx):
        with h5py.File(self.h5_file, 'a') as hf:
            hf['images'][idx, ...] = image
            hf['labels'][idx, ...] = label


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
            self.flatDict = [d for d in self.flatDict if d["examID"] != study_id]
            for v in view:
                if v+".png" in self.imageFiles:
                    self.imageFiles.remove(v+".png")

        newLen = len(self.imageFiles)
        print("{}/{} images for {} retained categories".format(newLen, origLen, top_c))


class H5Dataset(Dataset):
    def __init__(self, h5_file, batch_size=1):
        self.h5_file = h5_file
        self.batch_size = batch_size

        self.h5f = h5py.File(self.h5_file, 'r')
        self.num_samples = len(self.h5f['images'])


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)

        # Load images and labels for the current batch
        images = self.h5f['images'][start_idx:end_idx]
        images = torch.tensor(images, dtype=torch.float32)
        labels = self.h5f['labels'][start_idx:end_idx]
        labels = np.squeeze(torch.tensor(labels, dtype=torch.float32))

        return images, labels


    def close(self):
        self.h5f.close()
        

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