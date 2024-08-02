import rosbag
from cv_bridge import CvBridge
import cv2
import os
import csv
import numpy as np
import time

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-p', '--path', help='The Path to the Bag File.', dest='path')
# args, unknown = parser.parse_known_args()

def remove_all_in_directory(path):
    # Check if the path exists
    if not os.path.exists(path):
        return
    # Iterate over all the items in the directory
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            # Check if it is a file or directory and remove accordingly
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                remove_directory(file_path)  # Recursively remove the directory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def remove_directory(directory):
    # Recursively delete all contents of the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                remove_directory(file_path)  # Recursively remove the subdirectory
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    
    # Remove the now empty directory
    os.rmdir(directory)


def quaternion_to_euler(q):
    # [w,x,y,z]
    q0, q1, q2, q3 = q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * (np.pi / 2)  # use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    # Convert from radians to degrees
    roll = np.degrees(roll)
    pitch = np.degrees(pitch)
    yaw = np.degrees(yaw)

    return roll, pitch, yaw

# Bag File Dir
bag_dir = '/home/ali/VIOT-2/2022-03-10/16-16-18/2022-03-10-16-16-18.bag'
log_dir = os.path.dirname(bag_dir)
# Topics to extract
image_topic = '/tello/camera/image_raw'
odometry_topic = '/tello/odom'

# Directories to save extracted data
image_save_dir = log_dir + '/images'
odometry_filename = log_dir + '/odomPoses.csv'

# Create directories if they do not exist
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)
else:
    remove_all_in_directory(image_save_dir)


# Initialize CvBridge
bridge = CvBridge()

# Open bag file
with rosbag.Bag(bag_dir, 'r') as bag:
    with open(odometry_filename, mode='w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 't_string'])
        
        image_index = 0
        for topic, msg, t in bag.read_messages():

            
            # print (topic)
            # print (t)
            # print ("===================================")
            # print (msg)
            # print (type(t))
            
            # cc = 0 
            if topic == odometry_topic:
                # print(t)
                # print("yesssssssssssssssssssssssssssss")
                # print ("===================================")

                # print (msg)
                time.sleep(0.01)
                # Extract odometry data
                position = msg.pose.pose.position
                orientation = msg.pose.pose.orientation
                print(position)
                
                q = [orientation.w, orientation.x, orientation.y, orientation.z]
                roll, pitch, yaw = quaternion_to_euler(q)
                print(roll, pitch, yaw )
                # # Write odometry data to CSV
                print(t.to_sec())
                csv_writer.writerow([t.to_sec(), position.x, position.y, position.z, roll, pitch, yaw, 't'+str(t.to_sec())])
                cc = 1
            if topic == image_topic:
                # print(t)
                # print("NOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
                # print ("===================================")
                
                try:
                    # Convert ROS Image message to OpenCV image
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    # image_path = os.path.join(image_save_dir, f'image_{image_index:06d}.jpg')
                    print(t)
                    image_path = os.path.join(image_save_dir, str(t.to_sec())+'.jpg')
                    cv2.imwrite(image_path, cv_image)
                    image_index += 1
                    
                    # Show Image
                    cv2.imshow("test", cv_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error converting image: {e}")
            

cv2.destroyAllWindows()
