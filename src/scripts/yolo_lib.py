import os
import cv2
import argparse
import pandas as pd
from ultralytics import YOLO
# from src.utils.assess_quality import getEdgeDistance, getEdgeDistanceReal

'''
RUN YOLO MODEL ON RETINAL IMAGE
This file is responsible for running a trained
yolo model on a retinal image, in order to 
detect the optic disc location on the image
and extract information, with special
attention to the width of the bounding box
(Disc Diameter DD)s
'''

# Print obtained bounding box coordinates in different formats
# Parameters:
#   - results: object with info about the image passed on the model
def showResults(results):
    for result in results:
        xywh = result.boxes.xywh  # center-x, center-y, width, height
        print(xywh)
        xywhn = result.boxes.xywhn  # normalized
        print(xywhn)
        xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
        print(xyxy)
        xyxyn = result.boxes.xyxyn  # normalized
        print(xyxyn)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
        print(names)
        confs = result.boxes.conf  # confidence score of each box
        print(confs)

# Returns prediction confidence, bbox center
# coordinates and width from bbox object
# Parameters:
#   - bboxResults: bounding box ultralytics object
def extractInformationFromPred(bboxResults):
    confidence = bboxResults[0].boxes.conf.item()
    centerCoords = [bboxResults[0].boxes.xywh[0][0].item(), bboxResults[0].boxes.xywh[0][1].item()]
    width = bboxResults[0].boxes.xywh[0][2].item()

    return confidence, centerCoords, width

# Given a path to trained weights, load a yolo model
# Parameters:
#   - weightsPath: path to weights .pt to load model
def loadModel(weightsPath):    
    model = YOLO(os.path.join(weightsPath, 'best.pt'))
    return model

# Run the model on an image and return results
# Parameters:
#   - imagePath: path to an image to get structure coordinates
#   - model: model instance
def getCoordinates(imagePath, model):
    _, fileName = os.path.split(imagePath)
    imageName, _ = os.path.split(fileName)

    destPath = './src/data/images'

    model.predict(imagePath, save_txt=False, project=destPath, name=imageName)

    results = model(imagePath)

    return results

# Draw the bounding boxes on both od and fovea
# Parameters:
#   - image: the openCV image to draw the boxes
#   - results: object list, represent the output from the model
#   - color: bbox color
#   - objectName: bbox name
def draw_boxes(image, results, color, objectName=""):
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        label = f"{objectName}{conf:.2f}"

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Font and scale
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # Text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_offset_x = x1
        text_offset_y = y1 - 5 if y1 - 5 > text_height else y1 + text_height + 5

        # Draw the rectangle for the text
        cv2.rectangle(image,
                      (text_offset_x, text_offset_y - text_height - baseline),
                      (text_offset_x + text_width, text_offset_y + baseline),
                      color,
                      -1)  # Filled

        # Write text
        cv2.putText(image,
                    label,
                    (text_offset_x, text_offset_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    lineType=cv2.LINE_AA)

# Get prediction results from running od and fovea detection,
# with minimum confidence of 0.5
# Parameters:
#   - imagePath: path to the image to run the models
#   - odModel: model instance to detect OD
#   - foveaModel: model instance to detect fovea
#   - saveImg: bool, save or not the image
#   - output_path: where to save the image
def getPredictions(imagePath, odModel, saveImg=False, output_path="./src/data/images/test.jpg",):
    image = cv2.imread(imagePath)

    odResults = odModel.predict(imagePath, conf=0.5, max_det=1)

    draw_boxes(image, odResults, color=(0, 170, 0), objectName="Optic Disc: ")

    if saveImg is True:
        cv2.imwrite(output_path, image)

    return odResults

# Given a dataset path, run the model on all the images of the dataset
# Parameters:
#   - args: input arguments
#   - saveImg: whether to save images inferences from models
def runBRSetInferences(args, saveImg=False):
    writeFile = 'retinalInformation'
    labelsPath = '/scratch/diogo.alves/datasets/brset/physionet.org/files/brazilian-ophthalmological/1.0.0/labels.csv'

    odModel = loadModel(args.od_weights)
    foveaModel = loadModel(args.fovea_weights)

    labels = pd.read_csv(labelsPath)
    records = []

    for image in os.listdir(args.data_path):
        imageId, extension = os.path.splitext(image)
        
        if extension == '.jpg':
            imagePath = os.path.join(args.data_path, image)
            
            odInfo, foveaInfo = getPredictions(imagePath, odModel, foveaModel, saveImg)
            if not len(odInfo[0]) or not len(foveaInfo[0]):
                print(f"Could not detect Optic Disc or Fovea for {image}")
                continue

            odConfidence, odCenterCoords, discDiameter = extractInformationFromPred(odInfo)
            foveaConfidence, foveaCenterCoords, _ = extractInformationFromPred(foveaInfo)

            nasalDistance, temporalDistance, theta = getEdgeDistanceReal(os.path.join(args.data_path, image), odInfo[0], foveaInfo[0], saveNasalPoint=False)
            row = labels[labels['image_id'] == imageId]

            if not row.empty:
                if row.iloc[0]['image_field'] == 1:
                    label = 'Adequate'
                else:
                    label = 'Inadequate'
            
                records.append({
                    'image_id': imageId,
                    'quality_label': label,
                    'od_confidence': odConfidence,
                    'fovea_confidence': foveaConfidence,
                    'disc_diameter': discDiameter,
                    'od_center_x': odCenterCoords[0],
                    'od_center_y': odCenterCoords[1],
                    'fovea_center_x': foveaCenterCoords[0],
                    'fovea_center_y': foveaCenterCoords[1],
                    'nasal_distance': nasalDistance,
                    'temporal_distance': temporalDistance,
                    'od_fovea_angle': theta
                })
    
    df = pd.DataFrame(records)
    df.to_csv(f'data/{writeFile}.csv', index=False)

    print(f"Wrote inferences to data/{writeFile}.csv")

# Given an image path, run the model on the specific image
# Parameters:
#   - args: input arguments
#   - saveImg: bool, save or not the image
#   - showResults: bool, enable debug on image run
def runModelOnImage(args, saveImg=True, showResults=False):
    imagePath = os.path.join(args.data_path, f'{args.image}.jpg')
    
    odModel = loadModel(args.od_weights)

    # Run the model on an image to locate optic disc and fovea
    odInfo, foveaInfo = getPredictions(imagePath, odModel, saveImg)
    
    if not len(odInfo[0]) or not len(foveaInfo[0]):
        print(f"Could not detect Optic Disc or Fovea for {args.image}.jpg")
        return
    
    if showResults is True:
        showResults(odInfo)
        showResults(foveaInfo)
    
    # discDiameter = odInfo[0].boxes.xywh[0][2].item()
    # print(f'The optic disc diameter is: {discDiameter:.2f}')

    # Detect the distance between the edges and the optic disc/fovea
    # nasalDistance, temporalDistance, theta = getEdgeDistanceReal(imagePath, odInfo[0], foveaInfo[0])
    # nasalDistance = getEdgeDistance(imagePath, odInfo[0], structure='od')
    # temporalDistance = getEdgeDistance(imagePath, foveaInfo[0], structure='fovea')

    # if nasalDistance < discDiameter:
    #     print(f"The image is inadequate. Criteria: [1] the distance from the OD to the nasal edge ({nasalDistance:.2f}) is lower than 1DD ({discDiameter:.2f}).")

    # if temporalDistance < 2*discDiameter:
    #     print(f"The image is inadequate. Criteria: [2] the distance from the Macular Center to the temporal edge ({temporalDistance:.2f}) is lower than 2DD ({2*discDiameter:.2f}).")

# Main execution, decides how to run the model
def main(args):
    if args.image is not None:
        runModelOnImage(args, saveImg=True)

    elif args.image is None:
        runBRSetInferences(args, saveImg=False)

# python main.py --image img01233 >> "./data/logs/run.log" 2>&1
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assess image field definition (BRSet)")

    parser.add_argument('--image', type=str, default=None, help='Fundus image name')
    parser.add_argument('--data-path', type=str, default='/scratch/diogo.alves/datasets/brset/physionet.org/files/brazilian-ophthalmological/1.0.0/fundus_photos/', help='Fundus image dataset path')
    parser.add_argument('--od-weights', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/src/models/runs/detect/od_baseline2/train_results/weights', help='OD detection weights path')
    parser.add_argument('--fovea-weights', type=str, default='/home/rodrigocm/research/YOLO-on-fundus-images/src/models/runs/detect/fovea/train_results/weights', help='Fovea detection weights path')

    args = parser.parse_args()

    main(args)