from ..MISC.Tool import *
from ..Data.ReadAnnData import *
import random
import os
from ..Data.COCO import *
from ultralytics import YOLO

def convert_to_yolo(x, y, w, h, img_width, img_height):
    """
    Chuyển đổi tọa độ bounding box từ dạng pixel sang YOLO format.
    :param x: Tọa độ góc trên trái của bounding box (pixel)
    :param y: Tọa độ góc trên trái của bounding box (pixel)
    :param w: Chiều rộng của vật thể (pixel)
    :param h: Chiều cao của vật thể (pixel)
    :param img_width: Chiều rộng ảnh gốc (pixel)
    :param img_height: Chiều cao ảnh gốc (pixel)
    :return: (x_center, y_center, width, height) theo YOLO format
    """
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    return round(x_center, 6), round(y_center, 6), round(w_norm, 6), round(h_norm, 6)




def coco_to_yolo(TrainPATH, train, val, customTrainYaml):
    TrainPATH = abspath(TrainPATH)
    train_path = os.path.join(TrainPATH, "images", "train")
    val_path = os.path.join(TrainPATH, "images", "val")
    train_label_path = os.path.join(TrainPATH, "labels", "train")
    val_label_path = os.path.join(TrainPATH, "labels", "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    class_names = []

    datasets = {
        "train": (train, train_path, train_label_path),
        "val": (val, val_path, val_label_path)
    }

    for dataset_name, (dataset, image_dest, label_dest) in datasets.items():
        annotations = parse_json_annotations(dataset)
        total_files = len(annotations)  # Tổng số ảnh trong tập train hoặc val
        processed_files = 0  # Số ảnh đã xử lý cho tập train hoặc val

        log_print(f"🔄 Đang chuyển đổi {dataset_name.upper()} ({total_files} ảnh)...")

        for (x, y, w, h, label), image_path, img_h, img_w in annotations:
            if label not in class_names:
                class_names.append(label)

            label_index = class_names.index(label)
            x_center, y_center, w_norm, h_norm = convert_to_yolo(x, y, w, h, img_w, img_h)

            src_image_path = os.path.join(dataset, image_path)
            if not os.path.exists(src_image_path):
                log_print(f"⚠️ Ảnh không tồn tại: {src_image_path}", 3)
                continue

            new_image_path = os.path.join(image_dest, os.path.basename(image_path))
            new_label_path = os.path.join(label_dest, os.path.splitext(os.path.basename(image_path))[0] + ".txt")

            shutil.copy(src_image_path, new_image_path)
            with open(new_label_path, "w") as f:
                f.write(f"{label_index} {x_center} {y_center} {w_norm} {h_norm}\n")

            processed_files += 1
            log_progress(processed_files, total_files)


    # Lưu dataset.yaml vào TrainPATH
    yaml_path = os.path.join(TrainPATH, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}\n" + customTrainYaml)

    log_print("✅ Chuyển đổi COCO sang YOLO hoàn tất!")


def labelMeToYOLO(basepath, TrainPATH, customTrainYaml):
    
    basepath = abspath(basepath)
    TrainPATH = abspath(TrainPATH)
    jsonANN = parse_json_annotations(basepath)
    if not jsonANN:
        log_print("Vui lòng thêm file JSON chứa thông tin ảnh!", 3)
        return

    train_path = os.path.join(TrainPATH, "images", "train")
    val_path = os.path.join(TrainPATH, "images", "val")
    train_label_path = os.path.join(TrainPATH, "labels", "train")
    val_label_path = os.path.join(TrainPATH, "labels", "val")

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)
    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(val_label_path, exist_ok=True)

    class_names = []

    random.shuffle(jsonANN)
    val_size = int(len(jsonANN) * 0.2)
    val_annotations = jsonANN[:val_size]
    train_annotations = jsonANN[val_size:]

    datasets = [
        (train_path, train_label_path, train_annotations),
        (val_path, val_label_path, val_annotations),
    ]

    total_files = len(jsonANN)  # Tổng số ảnh cần xử lý
    processed_files = 0  # Số ảnh đã xử lý

    for image_dest, label_dest, anno_list in datasets:
        for (x, y, w, h, label), image_name, img_h, img_w in anno_list:
            if label not in class_names:
                class_names.append(label)

            label_index = class_names.index(label)
            x_center, y_center, w_norm, h_norm = convert_to_yolo(x, y, w, h, img_w, img_h)

            src_image_path = os.path.join(basepath, image_name)
            if not os.path.exists(src_image_path):
                log_print(f"Ảnh không tồn tại: {src_image_path}", 2)
                continue

            new_image_path = os.path.join(image_dest, image_name)
            new_label_path = os.path.join(label_dest, os.path.splitext(image_name)[0] + ".txt")

            shutil.copy(src_image_path, new_image_path)

            with open(new_label_path, "w") as f:
                f.write(f"{label_index} {x_center} {y_center} {w_norm} {h_norm}\n")

            processed_files += 1
            log_progress(processed_files, total_files)

    sys.stdout.write("\n")  # Xuống dòng sau khi hoàn tất
    yaml_path = os.path.join(TrainPATH, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: images/train\nval: images/val\nnc: {len(class_names)}\nnames: {class_names}\n{customTrainYaml}")

    log_print("✅ Chuyển đổi LabelMe sang YOLO hoàn tất!")


class YOLO_Model:
    def __init__(self, basepath, train_data_path=None, keep_train_path=False):
        """
        Khởi tạo model YOLO.
        
        :param basepath: Đường dẫn gốc chứa dataset.
        :param train_data_path: Đường dẫn tùy chỉnh cho thư mục TrainData. Nếu không đặt, mặc định sẽ là basepath/TrainData.
        :param keep_train_path: Nếu False, sẽ xóa toàn bộ nội dung trong TrainPATH trước khi bắt đầu.
        """
        self.basepath = os.path.abspath(basepath)  # Chuyển về đường dẫn tuyệt đối
        if train_data_path == None:
            self.TrainPATH = os.path.join(self.basepath,"TrainData")
        else:     
            self.TrainPATH = os.path.abspath(train_data_path)
        log_print("Save Train Data Saved In:" + self.TrainPATH)
        if keep_train_path != True:
            self.clear_train_path()


    def clear_train_path(self):
        """Xóa toàn bộ nội dung trong TrainPATH nếu thư mục tồn tại."""
        if os.path.exists(self.TrainPATH):
            shutil.rmtree(self.TrainPATH)
        os.makedirs(self.TrainPATH, exist_ok=True)
        
    def clear_train_path(self):
        """Xóa toàn bộ nội dung trong TrainPATH nếu thư mục tồn tại."""
        if os.path.exists(self.TrainPATH):
            shutil.rmtree(self.TrainPATH)
        os.makedirs(self.TrainPATH, exist_ok=True)

    def create_yolo_dataset_structure(self, dataset_path):
        """
        Tạo cấu trúc thư mục theo định dạng YOLO dataset.

        :param dataset_path: Đường dẫn gốc của dataset
        """
        sub_dirs = [
            "images/train", "images/val",
            "labels/train", "labels/val"
        ]

        for sub_dir in sub_dirs:
            path = os.path.join(dataset_path, sub_dir)
            os.makedirs(path, exist_ok=True)

        yaml_path = os.path.join(dataset_path, "dataset.yaml")
        if not os.path.exists(yaml_path):
            with open(yaml_path, "w") as f:
                f.write("train: images/train\nval: images/val\nnc: 0\nnames: []\n")

        log_print(f"Dataset structure created at: {dataset_path}")

    def SETUP_Dataset(self, customTrainYaml=""):
        """
        Thiết lập dataset theo cấu trúc phù hợp để train YOLO.

        - customTrainYaml (str, optional): Nội dung tùy chỉnh thêm vào file dataset.yaml.  
        """
        self.create_yolo_dataset_structure(self.TrainPATH)
        train_path, valid_path = None, None
        if os.path.isdir(self.basepath):
            subdirs = os.listdir(self.basepath)
            if "train" in subdirs and "valid" in subdirs:
                log_print("Phát Hiện Đây Là 1 COCO DATASET!")
                check_train_valid_json(self.basepath)
                train_path = os.path.join(self.basepath, "train")
                valid_path = os.path.join(self.basepath, "valid")
                coco_to_yolo(self.TrainPATH, train_path, valid_path, customTrainYaml)
            else:
                labelMeToYOLO(self.basepath,self.TrainPATH, customTrainYaml)

    def TrainYOLO(
        self,
        yoloModelPretrain, 
        save_dir=None, 
        name="YOLO_Training",
        device="cpu",  
        resume=False, 
        save_period = 10,
        epochs=50, 
        imgsz=640, 
        batch=16, 
        use_advanced_params=False,  
        **kwargs  
    ):
        """
        Huấn luyện YOLO với GPU tùy chỉnh & hỗ trợ đa GPU.

        - yoloModelPretrain: Model YOLO pretrained (vd: yolov8n.pt)
        - save_dir: Nơi lưu kết quả
        - name: Tên Model (vd: YOLO_Training)
        - device: Chọn GPU hoặc danh sách GPU (vd: "cpu", "cuda", [0,1,2])
        - resume: True để train tiếp từ checkpoint gần nhất
        - save_period: lưu lại checkpoint sau bao nhiêu epochs (vd: 10)
        - epochs (int): Số vòng lặp huấn luyện trên toàn bộ dataset.
        - imgsz (int): Kích thước ảnh đầu vào khi train.
        - batch (int): Số lượng ảnh xử lý cùng lúc trong một batch.
        - use_advanced_params: True để dùng các tham số nâng cao.
        """

        dataset_yaml = os.path.join(self.TrainPATH, "dataset.yaml")

        if save_dir is None:
            save_dir = os.path.join(self.TrainPATH, "runs", name)

        os.makedirs(save_dir, exist_ok=True)

        model = YOLO(yoloModelPretrain)

        if isinstance(device, str) and "," in device:
            device = [int(d) for d in device.split(",")]

        train_params = {
            "data": dataset_yaml,
            "epochs": epochs,
            "save_period": save_period,
            "batch": batch,
            "imgsz": imgsz,
            "device": device,
            "save": True,
            "project": save_dir,
            "name": name,
            "resume": resume
        }

        if use_advanced_params:
            valid_params = {}
            for key, value in kwargs.items():
                if hasattr(model, key):  
                    valid_params[key] = value
                else:
                    log_print(f"Cảnh báo: Model {yoloModelPretrain} không hỗ trợ `{key}`", 2)

            train_params.update(valid_params)
            
        results = model.train(**train_params)
        log_print(f"Training hoàn tất! Kết quả được lưu tại: {save_dir}/{name}")
        return results



