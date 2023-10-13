import numpy as np
from tqdm import tqdm

import argparse, pickle, os

import pydicom as dcm
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from highdicom.sr import ComprehensiveSR, ScoordContentItem, GraphicTypeValues, CodedConcept, ContainerContentItem, RelationshipTypeValues


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


def get_center_container(point):

    center_container = ScoordContentItem(
        name=CodedConcept(value='111010', 
                            scheme_designator='DCM',
                            meaning="Center"),
        graphic_type=GraphicTypeValues.POINT,
        graphic_data=point,
        relationship_type=RelationshipTypeValues.HAS_PROPERTIES
    )
    center_item = ScoordContentItem(
        name=CodedConcept(value='113000', 
                            scheme_designator='DCM',
                            meaning="Center"),
        graphic_type=GraphicTypeValues.POINT,
        graphic_data=point,
        relationship_type=RelationshipTypeValues.SELECTED_FROM
    )
    center_item.ReferencedContentItemIdentifier = [1,1,1,1]
    
    # Nest the polyline inside the container's ContentSequence
    center_container.ContentSequence = [center_item]
    return center_container


def get_polyline_container(polyline, label):
    # Create a container for each polyline
    polyline_container = ScoordContentItem(
        name=CodedConcept(value='111041', 
                            scheme_designator='DCM',
                            meaning="Outline"),
        graphic_type=GraphicTypeValues.POLYLINE,
        graphic_data=polyline,
        relationship_type=RelationshipTypeValues.HAS_PROPERTIES
    )

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
    
    # Nest the polyline inside the container's ContentSequence
    polyline_container.ContentSequence = [polyline_item]

    return polyline_container


def savePolylinesToDicomSR(dicomPath, dicomFile, srFile, polylines):
    dicomFile = dcm.dcmread(os.path.join(dicomPath, dicomFile))
    
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

    # Nested 3. layer for DICOM SR
    sub_content_container = ContainerContentItem(
        name=CodedConcept(value='111059', 
                          scheme_designator='DCM',
                          meaning="Single Image Finding"),
        relationship_type=RelationshipTypeValues.CONTAINS,
    )
    sub_content_container.ConceptCodeSequence = [CodedConcept(value='F-01796', scheme_designator='SRT', meaning="Mammography breast density")]

    # Nested 4./5. layer for DICOM SR with polylines
    polyline_sequence = Sequence()
    center_container = get_center_container(np.array([[1748, 2457]]))
    polyline_sequence.append(center_container)
    polyline_container_ben = get_polyline_container(polylines[0], "benign")
    polyline_container_mal = get_polyline_container(polylines[1], "malignant")
    polyline_sequence.append(polyline_container_ben)
    polyline_sequence.append(polyline_container_mal)

    # Add polylines into containers regarding required number of layers
    sub_content_container.ContentSequence = polyline_sequence
    main_content_container.ContentSequence = [sub_content_container]
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