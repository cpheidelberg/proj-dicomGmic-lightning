import numpy as np
import pydicom as dcm
from tqdm import tqdm

import argparse, sys, pickle, os, cv2, uuid

from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.sr.codedict import codes
from highdicom.sr import ComprehensiveSR, ScoordContentItem, GraphicTypeValues, CodedConcept, ContainerContentItem, RelationshipTypeValues, ContentSequence
from datetime import datetime


def loadPolyline(path:str, exam:str, label:str) -> np.ndarray:

    # create polyline path
    path = os.path.join(path, f"{exam}_polyline_{label}.txt")

    with open(path, 'r') as f:
        lines = f.readlines()
    polyline = []
    for line in lines:
        if line.startswith("---"):
            break
        if not line.startswith("Saliency Map"):
            x, y = map(int, line.strip().split(","))
            polyline.append((x, y))
    return np.array(polyline)


def savePolylinesToDicomSR(dicomPath, dicomFile, srFile, polylines):
    dicomFile = dcm.dcmread(os.path.join(dicomPath, dicomFile))
    """Save a list of polylines to a DICOM Structured Report."""
    # Create a primary container for the SR
    root_item = ContainerContentItem(
        name=CodedConcept(
            value='111017',
            scheme_designator='DCM',
            meaning='CAD Processing and Findings Summary'
        )
    )

    # Adding ContentTemplateSequence
    root_item.ContentTemplateSequence = [Dataset()]
    root_item.ContentTemplateSequence[0].MappingResource = "DCMR"
    root_item.ContentTemplateSequence[0].TemplateIdentifier = "4000"

    # Nested structure for DICOM SR to visualize polylines
    polyline_content_sequence = Sequence()
    for polyline in polylines:
        polyline_item = ScoordContentItem(
            name=CodedConcept(value='113000', 
                              scheme_designator='DCM', 
                              meaning='Region of Interest'
                            ),
            graphic_type=GraphicTypeValues.POLYLINE,
            graphic_data=polyline,
            relationship_type=RelationshipTypeValues.CONTAINS
        )
        
        polyline_content_sequence.append(polyline_item)
    root_item.ContentSequence = polyline_content_sequence
    content_sequence = Sequence()
    content_sequence.append(root_item)

    # DICOM SR file
    sr = ComprehensiveSR(
        evidence=[dicomFile],
        content=content_sequence,
        series_instance_uid=dcm.uid.generate_uid(),
        series_number=1,
        instance_number=1,
        sop_instance_uid=dcm.uid.generate_uid(),
        manufacturer="YourManufacturer"
    )
    sr.sop_class_uid="1.2.840.10008.5.1.4.1.1.88.50", 
    
    # Require argument adjustment
    if "PertinentOtherEvidenceSequence" in sr:
        sr.CurrentRequestedProcedureEvidenceSequence = sr.PertinentOtherEvidenceSequence
        del sr.PertinentOtherEvidenceSequence

    # Save the DICOM Dataset
    srPath = os.path.join(dicomPath, srFile)
    sr.save_as(srPath)


def createDicomSr(segPath: str, dicomPath:str, dicomFile:str, exam: str, image:str):
    """Create a DICOM Structured Report from given polylines."""
    examID = exam[image][0]
    polylineBegnin = loadPolyline(segPath, examID, "begnin")
    polylineMalignant = loadPolyline(segPath, examID, "malignant")

    polylines = [polylineBegnin, polylineMalignant]
    savePolylinesToDicomSR(dicomPath, dicomFile, "outputDicomSR.dcm", polylines)


def startConversion(segPath:str, examPath:str, dicomFile:str):
    """Start the conversion process by iterating over the exams and images."""
    print("Start conversion ...")
    with open(examPath, "rb") as f:
        examList = pickle.load(f)

    print("Loop over images ...")
    for exam in tqdm(examList, unit="exam"): # Loop over exams
        for image in ["L-MLO", "L-CC", "R-MLO", "R-CC"]: # Loop over images
            dicomPath = exam[f"{image}_path"]
            createDicomSr(segPath, dicomPath, dicomFile, exam, image)


def main():
    """Main function to handle command line arguments and initiate the conversion process."""
    
    # Retrieve command line arguments    
    parser = argparse.ArgumentParser(description='Extract model input from DICOM data')
    parser.add_argument('--segmentation-path', required=True)
    parser.add_argument('--dicom-file', required=True)
    parser.add_argument('--exam-list-path', required=True)
    args = parser.parse_args()

    segPath = args.segmentation_path
    dicomFile = args.dicom_file
    examPath = args.exam_list_path

    startConversion(segPath, examPath, dicomFile)

if __name__ == "__main__":
    main()