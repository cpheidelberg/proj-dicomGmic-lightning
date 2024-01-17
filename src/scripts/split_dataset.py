import pandas as pd
import pickle
import shutil
from tqdm import tqdm
import os


def loadPickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    return data

def flattenDict(data, annotations):
    flatDict = []
    for d in data:
        dictLCC = extractImageDict(d, "L-CC", annotations)
        dictRCC = extractImageDict(d, "R-CC", annotations)
        dictLMLO = extractImageDict(d, "L-MLO", annotations)
        dictRMLO = extractImageDict(d, "R-MLO", annotations)
        flatDict.append(dictLCC)
        flatDict.append(dictRCC)
        flatDict.append(dictLMLO)
        flatDict.append(dictRMLO)

    return flatDict

def extractImageDict(data, view, annotations):
    imgDict = {}
    imgDict["examID"] = data["examID"]
    imgDict["view"] = view
    imgDict["image"] = data[view]
    imgDict["dicom"] = data[view+"_path"]
    imgDict["split"] = getSplitLabel(data, view, annotations)

    return imgDict

def getSplitLabel(data, view, annotations):
    path = data[view+"_path"]

    studyID = path.split('/')[-2]
    imageID = path.split('/')[-1]

    splitCol = annotations[annotations["study_id"] == studyID].loc[annotations["image_id"] == imageID]
    split = splitCol["split"].iloc[0]
    return split


dictPath = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/exam_list.pkl'
imgPath = '../sdsHD/sd18a006/DataBaseMammography/vindr-mammo/1.0.0/output/cropped_images'
labelPath = 'sample_data/annotations/breast-level_annotations.csv'

imageDict = loadPickle(dictPath)
annotations = pd.read_csv(labelPath)
flatDict = flattenDict(imageDict, annotations)

print(len(flatDict))
print(flatDict[0])
for d in tqdm(flatDict):
    shutil.move(os.path.join(imgPath, d["image"][0]+".png"), os.path.join(imgPath, d["split"], d["image"][0]+".png"))