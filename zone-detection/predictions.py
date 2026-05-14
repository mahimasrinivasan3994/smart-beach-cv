import supervision as sv
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
import glob
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2 as cv


def zoneBaseDistribution(file, polygons):
  video_info = sv.VideoInfo.from_video_path(file)

  colors = sv.ColorPalette.default()
  zones = [
      sv.PolygonZone(
          polygon=polygon,
          frame_resolution_wh=video_info.resolution_wh
      )
      for polygon
      in polygons
  ]
  zone_annotators = [
      sv.PolygonZoneAnnotator(
          zone=zone,
          color=colors.by_idx(index),
          thickness=10,
          text_thickness=10,
          text_scale=5
      )
      for index, zone
      in enumerate(zones)
  ]
  box_annotators = [
      sv.BoxAnnotator(
          color=colors.by_idx(index),
          thickness=4,
          text_thickness=4,
          text_scale=2
          )
      for index
      in range(len(polygons))
  ]

  # extract frame
  generator = sv.get_video_frames_generator(file)
  iterator = iter(generator)
  frame = next(iterator)

  # detect
  results = model(frame, size=1280)
  detections = sv.Detections.from_yolov5(results)
  detections = detections[(detections.class_id == 0) & (detections.confidence > 0.25)]

  totalPeople = detections.xyxy.shape[0]
  peopleInWater = 0;
  for place, zone, zone_annotator, box_annotator in zip(['ALL', 'WATER'], zones, zone_annotators, box_annotators):
      mask = zone.trigger(detections=detections)
      detections_filtered = detections[mask]
      if(place == 'WATER'):
        peopleInWater = detections_filtered.xyxy.shape[0]

      frame = box_annotator.annotate(scene=frame, detections=detections_filtered, skip_label=True)
      frame = zone_annotator.annotate(scene=frame)


  print(f'\nTotal people detected: {totalPeople}')
  print(f'\nPeople in Water: {peopleInWater}')
  print(f'\nPeople on Sand: {totalPeople - peopleInWater}\n\n\n')
  
  imageRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
  img = Image.fromarray(imageRGB)
  img.show()


def getDefaultZone(file):
   im = Image.open(file)
   width, height = im.size
   return np.array([[0, 0],[width, 0],[width, height],[0, height],[0, 0]])


if __name__ == '__main__':
    testDataSet = glob.glob("images/*")
    for file in testDataSet:
        result = YOLO("weights/water-segmentation-best.pt").predict(file)[0]
        polygons = []
        polygons.append(getDefaultZone(file))
        for mask in result.masks:
           polygons.append(np.int32( mask.xy[0]))
           
        zoneBaseDistribution(file, polygons)

