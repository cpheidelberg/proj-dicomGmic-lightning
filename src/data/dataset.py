import numpy as np
import pandas as pd
import os, ast, pickle, h5py, time, sys, random
from tqdm import tqdm
import multiprocessing

import torch
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = "/".join(current_dir.split("/")[:-2])
sys.path.append(parent_dir)
from src.data import loading


class ClassificationImages(Dataset):
    def __init__(self, image_dir: str, dict_path: str, top_c: int):
        tab = pd.read_csv(dict_path)

        self.categories = list(tab['categories'].explode().unique())[:top_c]
        self.no_finding_idx = self.categories.index('No Finding')

        self.category_counts = [sum(tab['categories'] == c) for c in self.categories]
        self.category_starts = [sum(self.category_counts[:i]) for i in range(len(self.categories))]

        self.images = []
        for i in range(len(tab)):
            if tab.loc[i, 'categories'] in self.categories:
                self.images.append({
                    'path': os.path.join(image_dir, tab.loc[i, 'image']) + '.png',
                    'view': tab.loc[i, 'view'],
                    'category': tab.loc[i, 'categories'],
                    'center_x': tab.loc[i, 'center_x'],
                    'center_y': tab.loc[i, 'center_y']
                })

    def __len__(self):
        return len(self.categories) * self.category_counts[0]

    def __getitem__(self, index: int):
        return self._nth_image(self._convert_index(index))

    def _convert_index(self, index: int):
        category, scaled = divmod(index, self.category_counts[0])
        scaled = scaled * self.category_counts[category] // self.category_counts[0]
        return self.category_starts[category] + scaled

    def _nth_image(self, index: int):
        data = self.images[index]

        image = self._load_image(data['path'], data['view'], (data['center_x'], data['center_y']))
        enc = self._hot_encoding(data['category'])
        return image, enc

    def _load_image(self, path: str, view: str, best_center: tuple[int, int]):
        img = loading.load_image(path, view, horizontal_flip='NO')
        img = loading.process_image(img, view, best_center)
        img = np.expand_dims(img, 0).copy()
        return torch.Tensor(img)

    def _hot_encoding(self, category: str):
        index = self.categories.index(category)
        length = len(self.categories)
        encoding = np.zeros(length, dtype=np.float32)
        encoding[index] = 1.0
        return encoding


class ClassificationImagesFromPickle(ClassificationImages):

    def __init__(self, imageFolder:str, dictPath:str, labelPath:str, top_c=None):
        self.imageFolder = imageFolder
        self.imageFiles = [folder_path+file for folder_path in imageFolder for file in os.listdir(folder_path)]
        random.shuffle(self.imageFiles)
        self.imageDict = self.loadPickle(dictPath)
        print(self.imageDict[0])
        self.flatDict = self.flattenDict(self.imageDict)
        self.flatDictDF = pd.DataFrame(self.flatDict)

        print(self.flatDictDF.info())
        print(self.flatDictDF.iloc[0])

        self.labels = pd.read_csv(labelPath)

        # if top_c:
        #     self.filterCategories(top_c)

        self.labels = self.convertLabels(self.labels)
        self.unique_categories = list(set([label for labels in self.labels["finding_categories"] for label in labels]))
        print(self.unique_categories)

        self.flatDictDF["finding_categories"] = None
        self.addLabelstoDF()
        self.flatDictDF.to_csv(f"dictionaryTop{top_c}.csv")


    def __getitem__(self, idx):
        
        imagePath = self.imageFiles[idx]

        original_file_name = self.getOrigFilename(idx)
        data = self.getDataentry(original_file_name)
        loaded_image = self.getImage(imagePath, data)

        labelList = data["finding_categories"]
        labelEnc = self.createHotEncoding(labelList)

        return loaded_image, labelEnc


    def getImage(self, imagePath: str, data: pd.DataFrame) -> torch.Tensor:

        view = data["view"]
        loaded_image = loading.load_image(
            image_path=imagePath,
            view=view,
            horizontal_flip=data["horizontal_flip"],
        )
        loaded_image = loading.process_image(loaded_image, view, data["best_center"][view][0])
        loaded_image = np.expand_dims(loaded_image, 0).copy()
        loaded_image = torch.Tensor(loaded_image)

        return loaded_image


    def getDataentry(self, original_file_name: str) -> pd.DataFrame:
        """Find the entry of the original filename in self.flatDict["image"]"""
        data = None
        # index = None
        print(self.flatDict[0]["image"])
        for i, entry in enumerate(self.flatDict):
            if entry["image"] == original_file_name:
                studyID = entry["examID"]
                imageID = entry["dicom"].split('/')[-1]
                data = entry
                print(data)
                # index = i
                break
        if data is None:
            raise ValueError(f"Original file name '{original_file_name}' not found in flatDict")
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

    
    def extractImageDict(self, data, view):
        imgDict = {}
        imgDict["examID"] = data["examID"]
        imgDict["imageID"] = data[view+"_path"].split("/")[-1]
        imgDict["view"] = view
        imgDict["image"] = data[view][0]
        imgDict["dicom"] = data[view+"_path"]
        imgDict["horizontal_flip"] = data["horizontal_flip"]
        imgDict["best_center"] = data["best_center"]
        imgDict["window_location"] = data["window_location"]

        return imgDict


    def filterCategories(self, top_c=5):
        """Filter labels dataframe for top_c most occuring finding_categories and remove corresponding images from imageList"""

        class_counts = self.labels["finding_categories"].value_counts()
        origLen = len(self.imageFiles)
        origLength = len(self.labels)
        origLeng = len(self.flatDict)

        class_counts = class_counts.sort_values(ascending=False)
        top_labels = class_counts.head(top_c).index

        filtered_labels = self.labels[self.labels['finding_categories'].apply(lambda x: any(label in x for label in top_labels))]
        print(filtered_labels['finding_categories'].value_counts())
        removed_labels = self.labels[~self.labels.index.isin(filtered_labels.index)]
        assert len(self.labels) == len(filtered_labels) + len(removed_labels), "Filtering 'finding_annotations.csv' not successful"

        self.labels = filtered_labels
        filtered_exams_images = list(set((row["study_id"], row["image_id"]) for index, row in filtered_labels.iterrows()))

        print(f"Anzahl der einzigartigen Studien- und Bild-IDs: {len(filtered_exams_images)}")

        newImages = []
        newDict = []
        for exam_id, image_id in filtered_exams_images:
            for d in self.flatDict:
                if d["examID"] == exam_id and d["imageID"] == image_id:
                    newDict.append(d)
                    view = d["image"]
                    break
            for file in self.imageFiles:
                if file.endswith(f"/{view}.png"):
                    newImages.append(file)
                    break

        self.imageFiles = newImages
        self.flatDict = newDict

        print("{}/{} labels for {} retained categories".format(len(self.labels), origLength, top_c))
        print("{}/{} dicts for {} retained categories".format(len(self.flatDict), origLeng, top_c))
        print("{}/{} images for {} retained categories".format(len(self.imageFiles), origLen, top_c))


class PredictionClassificationImages(ClassificationImages):

    def __init__(self, imageFolder:str, dictPath: str, top_c=None, h5_file=None):
        self.imageFolder = imageFolder
        self.imageFiles = [folder_path+file for folder_path in imageFolder for file in os.listdir(folder_path)]
        random.shuffle(self.imageFiles)
        self.flatDictDF = pd.read_csv(dictPath, converters={"best_center": self.safe_literal_eval, "window_location": self.safe_literal_eval, "finding_categories": self.safe_literal_eval})

        if top_c:
            self.filterCategories(top_c)

        self.unique_categories = list(self.flatDictDF["finding_categories"].explode().unique())
        print("{} classes: {}".format(len(self.unique_categories), self.unique_categories))
        print(self.flatDictDF["finding_categories"].value_counts())


    def safe_literal_eval(self, s):
        try:
            string = ast.literal_eval(s)
            return string
        except ValueError as e:
            print(f"Fehler beim Parsen des Strings: {s}")
            raise e

    def __getitem__(self, idx):
        
        imagePath = self.imageFiles[idx]
        original_file_name = self.getOrigFilename(idx)
        data = self.getDataentry(original_file_name)
        loaded_image = self.getImage(imagePath, data)

        labelList = data["finding_categories"].iloc[0]
        labelEnc = self.createHotEncoding(labelList)

        return loaded_image, labelEnc, data.to_dict("list")


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

        if self.relevant_labels:
            label = label[self.relevant_label_indices]
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