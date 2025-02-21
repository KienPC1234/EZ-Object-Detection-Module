import os
import json
import numpy as np
import cv2
from ..MISC.Tool import *
from pycocotools.coco import COCO

def polygon_to_rect(cords):
    polygon = np.array(cords, dtype=np.int32)
    polygon = polygon.reshape((-1, 1, 2)) 
    x, y, w, h = cv2.boundingRect(polygon)
    return int(x), int(y), int(w), int(h) 

def extract_from_json(directory):
    directory = os.path.abspath(directory)
    """
    Trích xuất dữ liệu từ file JSON trong thư mục.
    
    Nếu file JSON có định dạng giống LabelMe, sẽ xử lý bằng cách cũ.
    Nếu là file annotation của COCO, sẽ tự động nhận diện và xử lý bằng COCO API.

    Tham số:
      - directory: thư mục chứa file JSON.
      - images_dir: (tuỳ chọn) thư mục chứa ảnh. Nếu có, sẽ nối với tên file ảnh.

    Trả về:
      - result: danh sách các tuple (shape_data, image_path)
          + shape_data: danh sách các bounding box dạng (x, y, w, h)
          + image_path: tên file ảnh (hoặc đường dẫn đầy đủ nếu có images_dir)
    """
    result = []
    
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):  
            continue  # Bỏ qua nếu không phải file JSON
        
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                data = json.load(file)
            
            # Kiểm tra nếu file có định dạng của LabelMe (cách xử lý cũ)
            if "shapes" in data and "imagePath" in data:
                image_path = os.path.basename(data["imagePath"])
                shape_data = []

                for shape in data["shapes"]:
                    if "points" in shape:
                        if shape.get("shape_type") == "rectangle":
                            points = shape["points"]
                            x, y = map(int, points[0])  
                            x2, y2 = map(int, points[1])  
                            w, h = abs(x2 - x), abs(y2 - y)
                            shape_data.append((x, y, w, h))
                        elif shape.get("shape_type") == "polygon":
                            x, y, w, h = polygon_to_rect(shape["points"])
                            shape_data.append((x, y, w, h))
                
                result.append((shape_data, image_path))
            
            # Kiểm tra nếu file có thể là annotation của COCO
            elif "images" in data and "annotations" in data:
                log_print(f"Phát hiện file COCO: {filename}, đang trích xuất dữ liệu...")
                coco = COCO(filepath)
                
                for img_id, img_info in coco.imgs.items():
                    image_file = img_info.get('file_name')

                    shape_data = []
                    ann_ids = coco.getAnnIds(imgIds=img_id)
                    anns = coco.loadAnns(ann_ids)

                    for ann in anns:
                        bbox = ann.get('bbox')
                        if bbox:
                            x, y, w, h = map(int, bbox)
                            shape_data.append((x, y, w, h))
                        elif 'segmentation' in ann and ann['segmentation']:
                            seg = ann['segmentation'][0]
                            x, y, w, h = polygon_to_rect(seg)
                            shape_data.append((x, y, w, h))

                    if shape_data:
                        result.append((shape_data, image_file))
                return result
        
        except Exception as e:
            log_print(f"Lỗi khi xử lý {filename}: {e}",3)
    
    return result

#<----COCO---->



#<----YOLO---->

def parse_json_annotations(dir_path):
    """
    Đọc các file JSON trong thư mục và trích xuất thông tin bbox.
    
    :param dir_path: Đường dẫn thư mục chứa JSON files
    :return: Danh sách [(x, y, w, h, label), imagePath, imageHeight, imageWidth]
    """
    dir_path= os.path.abspath(dir_path)
    results = []
    
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    log_print(f"Lỗi đọc JSON: {file_name}, bỏ qua...",3)
                    continue
            
            # Kiểm tra nếu file có thể là annotation của COCO
            if "images" in data and "annotations" in data:
                log_print(f"Phát hiện file COCO: {file_name}, đang trích xuất dữ liệu...")
                try:
                    coco = COCO(file_path)
                    for ann in coco.dataset.get("annotations", []):
                        if "image_id" not in ann or "bbox" not in ann or "category_id" not in ann:
                            continue
                        image_id = ann["image_id"]
                        image_info = next((img for img in coco.dataset.get("images", []) if img.get("id") == image_id), None)
                        if not image_info or "file_name" not in image_info or "height" not in image_info or "width" not in image_info:
                            continue
                        
                        x, y, w, h = ann["bbox"]
                        label =next((cat["name"] for cat in coco.dataset.get("categories", []) if cat["id"] == ann["category_id"]), None)
                        image_path = image_info["file_name"]
                        image_height = image_info["height"]
                        image_width = image_info["width"]
                        
                        results.append([(x, y, w, h, label), image_path, image_height, image_width])
                    return results
                
                except Exception as e:
                    log_print(f"Lỗi xử lý file COCO {file_name}: {e}",3)
            
            else:
                if "imagePath" not in data or "imageHeight" not in data or "imageWidth" not in data:
                    continue
                
                image_path = data["imagePath"]
                image_height = data["imageHeight"]
                image_width = data["imageWidth"]
                
                for shape in data.get("shapes", []):
                    if "label" not in shape or "points" not in shape or "shape_type" not in shape:
                        continue
                    
                    label = shape["label"]
                    points = shape["points"]
                    shape_type = shape["shape_type"]
                    
                    if shape_type == "rectangle" and len(points) == 2:
                        x, y = points[0]
                        x2, y2 = points[1]
                        w, h = abs(x2 - x), abs(y2 - y)
                    elif shape_type == "polygon" and len(points) > 2:
                        x, y, w, h = polygon_to_rect(points)
                    else:
                        continue
                    
                    if w <= 0 or h <= 0:
                        continue
                    
                    results.append([(x, y, w, h, label), image_path, image_height, image_width])
    
    return results

