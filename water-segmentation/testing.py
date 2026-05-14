from ultralytics import YOLO
from PIL import Image, ImageDraw
import glob


if __name__ == '__main__':
    testDataSet = glob.glob("datasets/test/images/*")
    for file in testDataSet:
        result = YOLO("weights/best.pt").predict(file)[0]
        img = Image.open(file)
        for mask in result.masks:
               polygon = mask.xy[0]
               draw = ImageDraw.Draw(img)
               draw.polygon(polygon,outline=(0,255,0), width=10)
        img.show()