import numpy as np
from tqdm import tqdm
import pandas as pd

import argparse, pickle, os, cv2

import pydicom as dcm
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from highdicom.sr import ComprehensiveSR, ScoordContentItem, GraphicTypeValues, CodedConcept, ContainerContentItem, RelationshipTypeValues, NumContentItem, TextContentItem


def loadPolyline(path:str, exam:str, label:str) -> np.ndarray:

    # create polyline path
    # path = os.path.join(path, f"{exam}_polyline_{label}.txt")
    files = [os.path.join(path, f) for f in os.listdir(path) if f.startswith(f"{exam}_polyline_{label}")]

    polylines = {}
    for i, f in enumerate(files):
        polyline = []
        try:
            with open(f, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("---"):
                    break
                if not line.startswith("Saliency Map"):
                    x, y = map(int, line.strip().split(","))
                    polyline.append((x, y))
        except Exception as error:
            print(exam, "\n\tFailed to load polylin because no polyline was registered. Process with empty polyline", str(error))
        polylines[f"{label}_{i}"] = np.array(polyline)

    return polylines


def get_center_coordinate(polyline):

    x_center = (np.min(polyline[:, 0]) + np.max(polyline[:, 0])) / 2
    y_center = (np.min(polyline[:, 1]) + np.max(polyline[:, 1])) / 2
    center_point = np.array([[int(x_center), int(y_center)]])

    return center_point


def get_polyline_item(prediction, polyline, point, label):
    # Create the point item to store the lesion center
    prediction_item = NumContentItem(
        name=CodedConcept(value='121070',
                            scheme_designator='DCM',
                            meaning="Numeric measurement"),
        value=prediction*100,
        unit=CodedConcept(value="%",
                            scheme_designator="UCUM",
                            meaning="Percent"),
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    prediction_item.ReferencedContentItemIdentifier = [1,1,1,1]

    center_item = ScoordContentItem(
        name=CodedConcept(value='113000',
                            scheme_designator='DCM',
                            meaning=f"Center {label}"),
        graphic_type=GraphicTypeValues.POINT,
        graphic_data=point,
        relationship_type=RelationshipTypeValues.SELECTED_FROM
    )
    center_item.ReferencedContentItemIdentifier = [1,1,1,1]

    # Create the actual polyline item
    polyline_item = ScoordContentItem(
        name=CodedConcept(value='113000', 
                            scheme_designator='DCM',
                            meaning=f"Polyline {label}"),
        graphic_type=GraphicTypeValues.POLYLINE,
        graphic_data=polyline,
        relationship_type=RelationshipTypeValues.SELECTED_FROM,
    )
    polyline_item.ReferencedContentItemIdentifier = [1,1,1,1]
    
    text_item = TextContentItem(
        name=CodedConcept(
            value='121071',
            scheme_designator='DCM',
            meaning="Text"
        ),
        value="Lesion: asymmetry",
        relationship_type=RelationshipTypeValues.CONTAINS
    )
    text_item.ReferencedContentItemIdentifier = [1, 1, 1, 1]

    return prediction_item, center_item, polyline_item, text_item


def get_empty_container(label):
    # Nested 3. layer for DICOM SR
    sub_content_container = ContainerContentItem(
        name=CodedConcept(value='111059', 
                          scheme_designator='DCM',
                          meaning="Single Image Finding"),
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    sub_content_container.ConceptCodeSequence = [CodedConcept(value='F-01796', scheme_designator='SRT', meaning="Mammography breast density")]
    
    # Nested 4./5. layer for DICOM SR with polylines
    polyline_item = TextContentItem(
        name=CodedConcept(
            value='113000', 
            scheme_designator='DCM',
            meaning=f"Polyline {label}"
        ),
        value=f"No polyline found for {label}", 
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    polyline_item.ReferencedContentItemIdentifier = [1, 1, 1, 1]

    # Add polylines into containers regarding required number of layers
    sub_content_container.ContentSequence = [polyline_item]

    return sub_content_container


def get_finding_container(prediction, polyline, label):
    # Nested 3. layer for DICOM SR
    sub_content_container = ContainerContentItem(
        name=CodedConcept(value='111059', 
                          scheme_designator='DCM',
                          meaning="Single Image Finding"),
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    sub_content_container.ConceptCodeSequence = [CodedConcept(value='F-01796', scheme_designator='SRT', meaning="Mammography breast density")]

    # Nested 4./5. layer for DICOM SR with polylines
    prediction = prediction[f"{label}_pred"].iloc[0]
    polyline_sequence = get_polyline_item(prediction, polyline, get_center_coordinate(polyline), label)

    # Add polylines into containers regarding required number of layers
    sub_content_container.ContentSequence = polyline_sequence

    return sub_content_container


def savePolylinesToDicomSR(dicomPath, dicomFile, srFile, prediction, polylines):
    dicomFile = dcm.dcmread(os.path.join(dicomPath, dicomFile))
    pngImage = dicomFile.pixel_array[0]
    
    # Create a primary container for the SR
    root_item = ContainerContentItem(
        name=CodedConcept(
            value='111017',
            scheme_designator='DCM',
            meaning='CAD Processing and Findings Summary'
        )
    )
    root_item.ContentTemplateSequence = [Dataset()]
    root_item.ContentTemplateSequence[0].MappingResource = "DCMR"
    root_item.ContentTemplateSequence[0].TemplateIdentifier = "4000"

    # Nested 1. layer for DICOM SR
    major_content_sequence = Sequence()
    major_content_container = ContainerContentItem(
        name=CodedConcept(value='111017',
            scheme_designator='DCM',
            meaning='CAD Processing and Findings Summary',
        ),
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    major_content_container.ConceptCodeSequence = [CodedConcept(value='111242', scheme_designator='DCM', meaning="All algorithms succeeded; with findings")]

    # Nested 2. layer for DICOM SR
    main_content_container = ContainerContentItem(
        name=CodedConcept(value='111034', 
                          scheme_designator='DCM',
                          meaning="Individual Impression/Recommendation"),
        relationship_type=RelationshipTypeValues.INFERRED_FROM,
        is_content_continuous=False,
    )
    main_content_container.ContentSequence = []

    for i, label in enumerate(["benign", "malignant"]):
        for p in polylines[i]:
            finding_container = get_finding_container(prediction, polylines[i][p], label)
            main_content_container.ContentSequence.append(finding_container)
        if len(polylines[i]) == 0:
            main_content_container.ContentSequence.append(get_empty_container(label))


    # finding_container_ben = get_finding_container(prediction, polylines[0], "benign")
    # finding_container_mal = get_finding_container(prediction, polylines[1], "malignant")
    # main_content_container.ContentSequence = (finding_container_ben, finding_container_mal)
    major_content_container.ContentSequence = [main_content_container]
    major_content_sequence.append(major_content_container)

    root_item.ContentSequence = major_content_sequence
    content_sequence = Sequence()
    content_sequence.append(root_item)

    # create DICOM SR file
    sr = ComprehensiveSR(
        evidence=[dicomFile],
        content=content_sequence,
        series_instance_uid=dcm.uid.generate_uid(),
        series_number=1,
        instance_number=1,
        sop_instance_uid=dcm.uid.generate_uid(),
        manufacturer="Uni Heidelberg",
    )

    # adopt DICOM SR as required by standard
    del sr.SOPClassUID
    sr.SOPClassUID = "1.2.840.10008.5.1.4.1.1.88.50"
    if "PertinentOtherEvidenceSequence" in sr:
        sr.CurrentRequestedProcedureEvidenceSequence = sr.PertinentOtherEvidenceSequence
        del sr.PertinentOtherEvidenceSequence

    # Save the DICOM Dataset
    srPath = os.path.join(dicomPath, srFile)
    sr.save_as(srPath)


def createDicomSr(segPath:str, dicomPath:str, dicomFile:str, prediction:pd.DataFrame, exam: str, image:str):
    """Create a DICOM Structured Report from given polylines."""
    examID = exam[image][0]
    polylineBenign = loadPolyline(segPath, examID, "benign")
    polylineMalignant = loadPolyline(segPath, examID, "malignant")

    polylines = [polylineBenign, polylineMalignant]
    savePolylinesToDicomSR(dicomPath, dicomFile, "outputDicomSR.dcm", prediction, polylines)


def startConversion(segPath:str, resultPath:str, examPath:str, dicomFile:str):
    """Start the conversion process by iterating over the exams and images."""
    print("Start conversion ...")
    with open(examPath, "rb") as f:
        examList = pickle.load(f)
    predictions = pd.read_csv(os.path.join(resultPath, "predictions.csv"))


    print("Loop over images ...")
    for exam in tqdm(examList, unit="exam"): # Loop over exams
        for image in ["L-MLO", "L-CC", "R-MLO", "R-CC"]: # Loop over images
            dicomPath = exam[f"{image}_path"]
            prediction = predictions[predictions["image_index"] == exam[image][0]]
            createDicomSr(segPath, dicomPath, dicomFile, prediction, exam, image)


def main():
    """Main function to handle command line arguments and initiate the conversion process."""
    
    # Retrieve command line arguments
    parser = argparse.ArgumentParser(description='Extract model input from DICOM data')
    parser.add_argument('--segmentation-path', required=True)
    parser.add_argument('--result-path', required=True)
    parser.add_argument('--dicom-file', required=True)
    parser.add_argument('--exam-list-path', required=True)
    args = parser.parse_args()

    segPath = args.segmentation_path
    resultPath = args.result_path
    dicomFile = args.dicom_file
    examPath = args.exam_list_path

    # segPath = 'sample_output/segmentation'
    # resultPath = 'sample_output'
    # examPath = 'sample_output/data.pkl'
    # dicomFile = '1-1.dcm'

    startConversion(segPath, resultPath, examPath, dicomFile)

if __name__ == "__main__":
    main()