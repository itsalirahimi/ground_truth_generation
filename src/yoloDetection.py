import cv2
import os
from ultralytics import YOLO

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
args, unknown = parser.parse_known_args()

# Load the YOLOv8n model
# model = YOLO('yolov8n.pt')
model = YOLO('yolov8m.pt')

def personDetect(input_image, output_image):
    # Load an image
    input_image_path = input_image
    image = cv2.imread(input_image_path)

    # Check if the image was loaded successfully
    if image is None:
        raise FileNotFoundError(f"Image file '{input_image_path}' not found.")

    # Perform object detection
    results = model(image)

    # Define the person class id (usually 0 for COCO dataset)
    person_class_id = 0

    # Get the first result
    result = results[0]

    # Create a copy of the image to draw bounding boxes
    annotated_image = image.copy()
    detect_person_count = 0
    # Filter detections for persons and draw bounding boxes

    out_rects = []
    output = ""
    for detection in result.boxes.data:
        class_id = int(detection[5])  # Class ID is typically the 6th element

        if class_id == person_class_id:
            detect_person_count += 1
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, detection[:4])
            output = str(x1) + "," + str(y1) + ","
            output += str(x2) + "," + str(y1) + ","
            output += str(x2) + "," + str(y2) + ","
            output += str(x1) + "," + str(y2)
            out_rects.append([(y2-y1), output, (x1, y1, x2, y2)])
            # Draw rectangle on the image
            # cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    print(out_rects)
    if (detect_person_count == 0):
        output = "-1,-1,-1,-1,-1,-1,-1,-1"
    elif (detect_person_count == 1):
        
        output = out_rects[0][1]
        x1, y1, x2, y2 = out_rects[0][2]
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    elif (detect_person_count > 1):
        print ("error! detect more than 1 person -- " + input_image)
        output = out_rects[0][1]
        x1, y1, x2, y2 = out_rects[0][2]


    # Save the output image
    output_image_path = output_image
    cv2.imwrite(output_image_path, annotated_image)

    return (output)


yoloImageDir = os.path.join(args.path, "YOLOImage")
if not os.path.isdir(yoloImageDir):
    os.makedirs(yoloImageDir)

# path = "/home/ali/161618log/telloimg/"
path_in = args.path + "/clearImage/"
path_out = args.path + "/YOLOImage/"
dir_list = os.listdir(path_in)
sort_dir = sorted(dir_list)

output_file = open(args.path +"/YOLOoutput.txt", "w")
for img in sort_dir:
    result_points = personDetect(path_in+img, path_out+img)
    output_file.write(result_points+"\n")
    print(img , result_points)

output_file.close()



