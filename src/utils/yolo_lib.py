import os
import cv2
import math
import argparse
import numpy as np
import pandas as pd

from ultralytics import YOLO
from shapely.geometry import LineString, Point, Polygon
# from src.utils.assess_quality import getEdgeDistance, getEdgeDistanceReal
from src.config import RAW_BRSET_LABELS_CSV, RAW_BRSET_IMAGES_DIR, MODELS_DIR, OD_MODEL_PATH, FOVEA_MODEL_PATH

'''
RUN YOLO MODEL ON RETINAL IMAGE
This file is responsible for running a trained
yolo model on a retinal image, in order to 
detect the optic disc location on the image
and extract information, with special
attention to the width of the bounding box
(Disc Diameter DD)s
'''

# Gets bounds for retinal image
# Parameters:
#   - imageInfo: OpenCV image object
def get_retina_bounds(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate bright retinal region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Remove tiny noise
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    # Largest contour = retina
    retina_contour = max(contours, key=cv2.contourArea)

    # Bounding box
    x, _, w, _ = cv2.boundingRect(retina_contour)

    retina_left = x
    retina_right = x + w
    retina_center_x = x + (w / 2)

    return {
        "left": retina_left,
        "right": retina_right,
        "center_x": retina_center_x,
        "width": w,
        "contour": retina_contour
    }

# Determine if the OD/Fovea is on the left, center or right of the image
# Parameters:
#   - imageInfo: object with info about the image passed on the model
#   - imageWidth: width of the full image
def determineStructureDirection(imageInfo, imagePath):
    image = cv2.imread(imagePath)

    retina_info = get_retina_bounds(image)

    if retina_info is None:
        return None

    od_x = imageInfo.boxes.xywh[0][0].item()

    retina_left = retina_info["left"]
    retina_width = retina_info["width"]

    relative_position = (
        (od_x - retina_left) / retina_width
    )

    # 0.0 -> left
    # 0.5 -> center
    # 1 -> right
    return relative_position

# Return BBox needed information from a structure
# Parameters:
#   - imageInfo: bbox information
#   - structure: 'od' or 'fovea'
def getBboxInformation(imageInfo, structure):
    xywh = imageInfo.boxes.xywh

    cx = xywh[0][0].item()
    cy = xywh[0][1].item()

    if structure == 'fovea':
        return np.array([cx, cy])
    
    width = xywh[0][2].item()
    height = xywh[0][3].item()

    return np.array([cx, cy]), width, height

# Calculate the angle theta between the ODCenter
# and foveaCenter
# Parameters:
#   - odCenter: np array with optic disc center coordinates
#   - foveaCenter: np array with fovea center coordinates
def calculateAngleDeg(odCenter, foveaCenter):
    dx = foveaCenter[0] - odCenter[0]
    dy = foveaCenter[1] - odCenter[1]

    angleRad = math.atan2(abs(dy), abs(dx))

    return math.degrees(angleRad)

# Extract nasal point from OD, given a
# Fovea-OD unit vector, pointing to the
# nasal edge
# Parameters:
#   - odCenter: np array with optic disc center coordinates
#   - width: optic disc width given by bbox
#   - height: optic disc height given by bbox
#   - nasalUnitVector: Fovea-OD unit vector
def getNasalPointOnOD(odCenter, width, height, nasalUnitVector):
    cx, cy = odCenter
    ux, uy = nasalUnitVector

    # retorna 1000 pontos, no intervalo [0, 2pi]
    # e calcula equacoes parametricas da elipse
    ts = np.linspace(0, 2*np.pi, 1000)
    xs = (width/2) * np.cos(ts)
    ys = (height/2) * np.sin(ts)

    # projecoes na direção nasal (angulo OD e Fovea)
    projections = xs * ux + ys * uy

    # o maior valor aponta a projeção mais próxima do vetor nasal
    idx = np.argmax(projections)

    x_nasal = cx + xs[idx]
    y_nasal = cy + ys[idx]

    return x_nasal, y_nasal

# Calculate distance from a point to its
# closest edge (temporal or nasal)
# Parameters:
#   - image: cv2 image object
#   - unitVector: nasal or temporal OD-Fovea unit vector
#   - retinalPoint: point from retina to calculate distance from
def calculateDistance(image, unitVector, retinalPoint):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(grayImage, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)   # contorno mais externo da retina

    retina_edge = Polygon(main_contour.squeeze())

    if not retina_edge.contains(Point(retinalPoint)):
        print("Ponto nasal já está na borda da retina ou fora dela — distância = 0")
        return 0.0

    line = LineString([retinalPoint, retinalPoint + 3000 * unitVector])
    intersection = line.intersection(retina_edge.boundary)

    if intersection.geom_type == 'MultiPoint':
        nasalEdgePoint = min(intersection.geoms, key=lambda pt: pt.distance(Point(retinalPoint)))
    elif intersection.geom_type == 'Point':
        nasalEdgePoint = intersection
    else:
        raise ValueError("Interseção inesperada")

    distance = Point(retinalPoint).distance(nasalEdgePoint)
    
    return distance

# Calculate distance from temporal edge
# to the Fovea and from nasal edge to 
# the Optic Disc.
# Parameters:
#   - imagePath: path to the image file
#   - odInfo: BBox information for OD
#   - foveaInfo: BBox information for Fovea
#   - saveNasalPoint: wheter to save or not nasal point image   
#   - outputPath: where to save nasal point image
def getEdgeDistanceReal(imagePath, odInfo, foveaInfo, saveNasalPoint=True, outputPath="./src/data/images"):
    image = cv2.imread(imagePath)

    odCenter, width, height = getBboxInformation(odInfo, structure='od')
    foveaCenter = getBboxInformation(foveaInfo, structure='fovea')
    
    odFoveaVector = foveaCenter - odCenter
    temporalUnitVector = odFoveaVector / np.linalg.norm(odFoveaVector)
    nasalUnitVector = -temporalUnitVector

    nasalX, nasalY = getNasalPointOnOD(odCenter, width, height, nasalUnitVector)
    nasalPoint = np.array([nasalX, nasalY])

    theta = calculateAngleDeg(odCenter, foveaCenter)
    
    if saveNasalPoint is True:
        _, fileName = os.path.split(imagePath)

        name, ext = os.path.splitext(fileName)
        output_filename = f'{name}_nasal{ext}'

        cv2.circle(image, (int(nasalX), int(nasalY)), radius=10, color=(255,0,0), thickness=-1)
        cv2.circle(image, (int(odCenter[0]), int(odCenter[1])), radius=10, color=(255,0,0), thickness=-1)
        cv2.circle(image, (int(foveaCenter[0]), int(foveaCenter[1])), radius=10, color=(255,0,0), thickness=-1)
        cv2.imwrite(os.path.join(outputPath, output_filename), image)

    nasalDistance = calculateDistance(image, nasalUnitVector, nasalPoint)
    temporalDistance = calculateDistance(image, temporalUnitVector, np.array(foveaCenter))

    return nasalDistance, temporalDistance, theta

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
    boxes = bboxResults[0].boxes

    if boxes is None or len(boxes) == 0:
        return None, None, None

    confidence = boxes.conf.item()
    centerCoords = [boxes.xywh[0][0].item(), boxes.xywh[0][1].item()]
    width = boxes.xywh[0][2].item()

    return confidence, centerCoords, width

# Given a path to trained weights, load a yolo model
# Parameters:
#   - weightsPath: path to weights .pt to load model
def loadModel(weightsPath):    
    model = YOLO(weightsPath)
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
def getPredictions(imagePath, odModel, foveaModel, saveImg=False, output_path="",):
    image = cv2.imread(imagePath)

    odResults = odModel.predict(imagePath, conf=0.5, max_det=1)
    foveaResults = foveaModel.predict(imagePath, conf=0.5, max_det=1)

    draw_boxes(image, odResults, color=(0, 170, 0), objectName="Optic Disc: ")
    draw_boxes(image, foveaResults, color=(255, 0, 0), objectName="Fovea: ")

    if saveImg is True:
        cv2.imwrite(output_path, image)

    return odResults, foveaResults

# Given a dataset path, run the model on all the images of the dataset
# Parameters:
#   - args: input arguments
#   - saveImg: whether to save images inferences from models
def runBRSetInferences(args, saveImg=False):
    writeFile = 'retinalInformation'

    odModel = loadModel(args.od_weights)
    foveaModel = loadModel(args.fovea_weights)

    labels = pd.read_csv(RAW_BRSET_LABELS_CSV)
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
    foveaModel = loadModel(args.fovea_weights)

    # Run the model on an image to locate optic disc and fovea
    odInfo, foveaInfo = getPredictions(imagePath, odModel, foveaModel, saveImg)
    
    if not len(odInfo[0]) or not len(foveaInfo[0]):
        print(f"Could not detect Optic Disc or Fovea for {args.image}.jpg")
        return
    
    if showResults is True:
        showResults(odInfo)
        showResults(foveaInfo)
    
    discDiameter = odInfo[0].boxes.xywh[0][2].item()
    print(f'The optic disc diameter is: {discDiameter:.2f}')

    # Detect the distance between the edges and the optic disc/fovea
    nasalDistance, temporalDistance, theta = getEdgeDistanceReal(imagePath, odInfo[0], foveaInfo[0])
    # nasalDistance = getEdgeDistance(imagePath, odInfo[0], structure='od')
    # temporalDistance = getEdgeDistance(imagePath, foveaInfo[0], structure='fovea')

    if nasalDistance < discDiameter:
        print(f"The image is inadequate. Criteria: [1] the distance from the OD to the nasal edge ({nasalDistance:.2f}) is lower than 1DD ({discDiameter:.2f}).")

    if temporalDistance < 2*discDiameter:
        print(f"The image is inadequate. Criteria: [2] the distance from the Macular Center to the temporal edge ({temporalDistance:.2f}) is lower than 2DD ({2*discDiameter:.2f}).")

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
    parser.add_argument('--data-path', type=str, default=RAW_BRSET_IMAGES_DIR, help='Fundus image dataset path')
    parser.add_argument('--od-weights', type=str, default=OD_MODEL_PATH, help='OD detection weights path')
    parser.add_argument('--fovea-weights', type=str, default=FOVEA_MODEL_PATH, help='Fovea detection weights path')

    args = parser.parse_args()

    main(args)