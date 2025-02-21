import os
import json
import shutil
from pathlib import Path
from ..MISC.Tool import *
from ..Data.ReadAnnData import *
from pycocotools.coco import COCO
import difflib
import random
import zipfile

def merge_coco_dirs(dir1, dir2, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(str(Path(dir1).parent), "COCO_MERGER")
    else:
        output_dir = abspath(output_dir)
    
    dir1 =abspath(dir1)
    dir2 =abspath(dir2)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    image_counter = 0
    annotation_counter = 0

    def load_and_merge_json(dir_path, image_offset, annotation_offset):
        nonlocal coco_data, image_counter, annotation_counter
        
        for json_file in Path(dir_path).glob("*.json"):
            with open(json_file, "r") as f:
                data = json.load(f)
                
                if not coco_data["categories"]:
                    coco_data["categories"] = data["categories"]
                
                for image in data["images"]:
                    image["id"] += image_offset
                    coco_data["images"].append(image)
                
                for annotation in data["annotations"]:
                    annotation["id"] += annotation_offset
                    annotation["image_id"] += image_offset
                    coco_data["annotations"].append(annotation)

                image_counter += len(data["images"])
                annotation_counter += len(data["annotations"])

    load_and_merge_json(dir1, image_counter, annotation_counter)
    load_and_merge_json(dir2, image_counter, annotation_counter)

    output_json_path = os.path.join(output_dir, "merged_coco.json")
    with open(output_json_path, "w") as outfile:
        json.dump(coco_data, outfile)

    def copy_files(src_dir, output_dir):
        log_print("Đang Sao Chép File Ở Thư Mục "+src_dir+" Vào Thư Mục "+output_dir)
        files = list(Path(src_dir).glob("*"))
        total_files = len(files)
        
        for idx, file in enumerate(files, start=1):
            if file.suffix != ".json":
                shutil.copy(str(file), output_dir)
            log_progress(idx, total_files)
    
    copy_files(dir1, output_dir)
    copy_files(dir2, output_dir)

    print(f"Merge complete. Data saved to {output_json_path} and files copied to {output_dir}.")
    return output_dir


def _find_json_for_folder(DIR, folder_name):
    """
    Tìm file JSON phù hợp trong toàn bộ DIR:
      - Nếu có file chứa trực tiếp tên folder_name (vd: train.json khi tìm train), ưu tiên lấy ngay.
      - Nếu không có, dùng difflib để tìm file có độ tương đồng cao nhất.
    """
    candidate_files = []
    
    # Duyệt toàn bộ DIR để tìm tất cả file JSON
    for root, _, files in os.walk(DIR):
        for file in files:
            if file.lower().endswith('.json'):
                candidate_files.append(os.path.join(root, file))

    if not candidate_files:
        return None

    # **Ưu tiên file có tên chứa chính xác "train" hoặc "valid/val"**
    folder_name_lower = folder_name.lower()
    priority_files = []
    
    for file_path in candidate_files:
        file_base = os.path.splitext(os.path.basename(file_path))[0].lower()
        
        if folder_name_lower == "train" and ("instances_train" in file_base or "train" in file_base):
            return file_path 
        
        if folder_name_lower == "valid" and ("instances_valid" in file_base or "valid" in file_base or "val" in file_base):
            return file_path 

        # Nếu không có file hoàn toàn khớp, đưa vào danh sách chờ để so sánh bằng difflib
        priority_files.append((file_path, file_base))

    # **Dùng difflib để tìm file gần giống nhất nếu chưa chọn được**
    best_file = None
    best_ratio = 0

    for file_path, file_base in priority_files:
        ratio = difflib.SequenceMatcher(None, folder_name_lower, file_base).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_file = file_path

    return best_file if best_ratio > 0.4 else None  # Chỉ chọn file nếu độ tương đồng > 40%

def check_train_valid_json(DIR):
    DIR = abspath(DIR)
    """
    Kiểm tra thư mục DIR chứa các thư mục con 'train' và 'valid':
      - Nếu có file JSON trong thư mục gốc của 'train' và 'valid', thì thông báo.
      - Nếu không có, tìm file JSON gần giống trong toàn bộ DIR và copy vào.
    Trả về dict với đường dẫn file JSON tương ứng hoặc None nếu không tìm thấy.
    """
    trainFolder = os.path.join(DIR, "train")
    validFolder = os.path.join(DIR, "valid")
    
    result = {"train": None, "valid": None}
    
    # Kiểm tra file JSON trực tiếp trong thư mục train và valid
    train_json_direct = [f for f in os.listdir(trainFolder)
                         if f.lower().endswith('.json') and os.path.isfile(os.path.join(trainFolder, f))]
    valid_json_direct = [f for f in os.listdir(validFolder)
                         if f.lower().endswith('.json') and os.path.isfile(os.path.join(validFolder, f))]
    
    if train_json_direct and valid_json_direct:
        log_print("✅ Đã tìm thấy file JSON trong cả thư mục train và valid.")
        result["train"] = os.path.join(trainFolder, train_json_direct[0])
        result["valid"] = os.path.join(validFolder, valid_json_direct[0])
    else:
        # Xử lý cho thư mục train nếu không có file JSON trực tiếp
        if not train_json_direct:
            found_train = _find_json_for_folder(DIR, "train")
            if found_train:
                dest_train = os.path.join(trainFolder, os.path.basename(found_train))
                shutil.copy(found_train, dest_train)
                log_print(f"📂 Copy file JSON {found_train} vào thư mục train.")
                result["train"] = dest_train
            else:
                log_print("⚠️ Không tìm thấy file JSON phù hợp cho thư mục train, vui lòng cho file coco json của train vào thư mục train!",3)
        else:
            result["train"] = os.path.join(trainFolder, train_json_direct[0])
        
        # Xử lý cho thư mục valid nếu không có file JSON trực tiếp
        if not valid_json_direct:
            found_valid = _find_json_for_folder(DIR, "valid")
            if found_valid:
                dest_valid = os.path.join(validFolder, os.path.basename(found_valid))
                shutil.copy(found_valid, dest_valid)
                log_print(f"📂 Copy file JSON {found_valid} vào thư mục valid.")
                result["valid"] = dest_valid
            else:
                log_print("⚠️ Không tìm thấy file JSON phù hợp cho thư mục valid, vui lòng cho file coco json của valid vào thư mục valid!",3)
        else:
            result["valid"] = os.path.join(validFolder, valid_json_direct[0])
    
    return result


def LABELME_to_coco(DIR, COVERT_PATH=None, train_ratio=0.8):
    """
    Chuyển đổi LabelMe annotation từ JSON sang định dạng COCO và chia thành train/val.
    
    :param DIR: Thư mục chứa annotation JSON
    :param COVERT_PATH: Thư mục lưu dataset COCO, mặc định là DIR + "COCO_DATASET"
    :param train_ratio: Tỉ lệ train/val (mặc định 80% train, 20% val)
    """
    DIR = abspath(DIR)
    if COVERT_PATH is None:
        COVERT_PATH = os.path.abspath( "COCO_DATASET")
    else:
        COVERT_PATH = os.path.abspath(COVERT_PATH)
    os.makedirs(COVERT_PATH, exist_ok=True)
    train_dir = os.path.join(COVERT_PATH, "train")
    val_dir = os.path.join(COVERT_PATH, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    annotations = parse_json_annotations(DIR)
    random.shuffle(annotations)
    
    train_size = int(len(annotations) * train_ratio)
    train_data = annotations[:train_size]
    val_data = annotations[train_size:]
    
    def save_coco_json(data, save_path, img_dir):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }
        category_set = {}
        image_set = {}
        ann_id = 1
        
        os.makedirs(img_dir, exist_ok=True)
        total_images = len(data)
        
        for idx, (bbox, img_path, img_h, img_w) in enumerate(data, 1):
            if img_path not in image_set:
                img_id = len(image_set) + 1
                image_set[img_path] = img_id
                coco_format["images"].append({
                    "id": img_id,
                    "file_name": img_path,
                    "height": img_h,
                    "width": img_w
                })
                
                src_img_path = os.path.join(DIR, img_path)
                dst_img_path = os.path.join(img_dir, img_path)
                shutil.copy2(src_img_path, dst_img_path)
                log_progress(idx, total_images)
            else:
                img_id = image_set[img_path]
            
            label = bbox[4]
            if label not in category_set:
                cat_id = len(category_set) + 1
                category_set[label] = cat_id
                coco_format["categories"].append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": "object"
                })
            else:
                cat_id = category_set[label]
            
            x, y, w, h = bbox[:4]
            coco_format["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0
            })
            ann_id += 1
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(coco_format, f, indent=4, ensure_ascii=False)
    
    save_coco_json(train_data, os.path.join(train_dir, "annotations.json"), train_dir)
    save_coco_json(val_data, os.path.join(val_dir, "annotations.json"), val_dir)
    
    log_print(f"Dataset COCO đã lưu tại {COVERT_PATH}", 1)


from pycocotools.coco import COCO

def coco_dataset(train_data_path, train_ann_path, save_path=None, valid_data_path=None, valid_ann_path=None, valid_split=0.2):
    
    if save_path is None:
        save_path = os.path.abspath("COCO_DATASET")
    else:
        save_path = os.path.abspath(save_path)
    
    train_output = os.path.join(save_path, "train")
    valid_output = os.path.join(save_path, "valid")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(valid_output, exist_ok=True)
    
    def clear_folder(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)
    
    clear_folder(train_output)
    clear_folder(valid_output)
    
    def extract_zip(zip_path, extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    if train_data_path.endswith(".zip"):
        log_print("Giải nén dữ liệu train...")
        extract_zip(train_data_path, train_output)
        train_data_path = train_output
    
    if valid_data_path and valid_data_path.endswith(".zip"):
        log_print("Giải nén dữ liệu valid...")
        extract_zip(valid_data_path, valid_output)
        valid_data_path = valid_output
    
    coco_train = COCO(train_ann_path)
    all_train_images = coco_train.getImgIds()
    random.shuffle(all_train_images)
    
    if valid_ann_path:
        coco_valid = COCO(valid_ann_path)
        valid_images = coco_valid.getImgIds()
    else:
        split_idx = int(len(all_train_images) * (1 - valid_split))
        valid_images = all_train_images[split_idx:]
        all_train_images = all_train_images[:split_idx]
    
    def copy_images(image_ids, source_path, dest_path, coco):
        total = len(image_ids)
        for idx, img_id in enumerate(image_ids, 1):
            file_name = coco.loadImgs(img_id)[0]['file_name']
            src = os.path.join(source_path, file_name)
            dst = os.path.join(dest_path, file_name)
            try:
                shutil.copy(src, dst)
            except FileNotFoundError:
                log_print(f"Không tìm thấy file ảnh: {file_name}", 2)
            log_progress(idx, total)
    
    log_print("Đang sao chép ảnh train...")
    copy_images(all_train_images, train_data_path, train_output, coco_train)
    
    log_print("Đang sao chép ảnh valid...")
    if valid_ann_path:
        copy_images(valid_images, valid_data_path, valid_output, coco_valid)
    else:
        copy_images(valid_images, train_data_path, valid_output, coco_train)
    
    def save_annotations(image_ids, coco, output_json):
        new_coco = {
            "info": coco.dataset.get("info", {}),
            "licenses": coco.dataset.get("licenses", []),
            "categories": coco.dataset["categories"],
            "images": [],
            "annotations": []
        }
        
        img_id_map = {old_id: new_id for new_id, old_id in enumerate(image_ids)}
        for img_id in image_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_info["id"] = img_id_map[img_id]
            new_coco["images"].append(img_info)
            anns = coco.getAnnIds(imgIds=[img_id])
            for ann in coco.loadAnns(anns):
                ann["image_id"] = img_id_map[ann["image_id"]]
                new_coco["annotations"].append(ann)
        
        with open(output_json, "w") as f:
            json.dump(new_coco, f, indent=4)
    
    log_print("Đang lưu annotation train...")
    save_annotations(all_train_images, coco_train, os.path.join(train_output, "train.json"))
    
    log_print("Đang lưu annotation valid...")
    if valid_ann_path:
        save_annotations(valid_images, coco_valid, os.path.join(valid_output, "valid.json"))
    else:
        save_annotations(valid_images, coco_train, os.path.join(valid_output, "valid.json"))
    
    log_print("Quá trình xử lý hoàn tất!")
