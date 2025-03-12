import cv2
import xml.etree.ElementTree as ET
import numpy as np

def extract_bbox_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def add_noise(image, noise_level=30):
    h, w, c = image.shape
    noise = np.random.normal(0, noise_level, (h, w, c))
    noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy_image

def extract_objects_with_noise(img_path, xml_path, noise_level=30):
    image = cv2.imread(img_path)
    boxes = extract_bbox_from_xml(xml_path)

    noise_image = add_noise(image, noise_level)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        noise_image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :]  # Replace content inside bbox with original image

    cv2.imshow('Extracted Objects with Noise', noise_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_objects(img_path, xml_path):
    image = cv2.imread(img_path)
    boxes = extract_bbox_from_xml(xml_path)

    mask = np.zeros_like(image)  # Create a black mask

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        mask[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :]  # Copy content inside bbox to mask

    cv2.imshow('Extracted Objects', mask)
    cv2.waitKey(15)
    cv2.destroyAllWindows()
    # cv2.imwrite('/home/zzh/domaingen/1.jpg', mask)

# 请替换以下路径为你的图片和标注文件的路径

def add_blur(image, blur_level=31):
    return cv2.GaussianBlur(image, (blur_level, blur_level), 0)

def extract_objects_with_blur(img_path, xml_path, blur_level=15):
    image = cv2.imread(img_path)
    boxes = extract_bbox_from_xml(xml_path)

    blurred_image = add_blur(image, blur_level)

    for box in boxes:
        xmin, ymin, xmax, ymax = box
        blurred_image[ymin:ymax, xmin:xmax, :] = image[ymin:ymax, xmin:xmax, :]  # Replace content inside bbox with original image

    cv2.imshow('Extracted Objects with Blur', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img_path = '/home/zzh/domaingen/2007_000241.jpg'
xml_path = '/home/zzh/domaingen/2007_000241.xml'

extract_objects_with_blur(img_path, xml_path, blur_level=41)