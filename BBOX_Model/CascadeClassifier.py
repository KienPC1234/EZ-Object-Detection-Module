from ..MISC.Tool import *
from ..MISC.binLoader import *
from ..Data.ReadAnnData import *
from ..Data.COCO import *
import os
import shutil
import cv2
from enum import Enum

class FeatureType(Enum):
    """
    Loại đặc trưng sử dụng trong huấn luyện Haar Cascade.
    """
    HAAR = "HAAR"
    LBP = "LBP"
    HOG = "HOG"

class StageType(Enum):
    """
    Kiểu tầng trong quá trình huấn luyện bộ phân loại.
    """
    BOOST = "BOOST"
    WAB = "WAB"

class BoostType(Enum):
    """
    Kiểu thuật toán boosting sử dụng trong huấn luyện.
    """
    GAB = "GAB"
    DAB = "DAB"
    RAB = "RAB"

class ModeType(Enum):
    """
    Chế độ huấn luyện của bộ phân loại.
    """
    BASIC = "BASIC"
    CORE = "CORE"
    ALL = "ALL"




class CascadeTrainer:
    """
    Lớp huấn luyện bộ phân loại Haar Cascade dùng OpenCV.
    """
    def __init__(self, base_dir, num_stages=20,sample_size=(32, 32), train_data_path ="",negative_image_path="",feature_type=FeatureType.HAAR,
                 stage_type=StageType.BOOST, boost_type=BoostType.GAB, mode=ModeType.BASIC):
        """
        Khởi tạo bộ huấn luyện Haar Cascade.
        Lưu ý: Nếu LÀ COCO DATASET THÌ CHO FILE JSON "annotation" VÀO THƯ MỤC CHỨA ẢNH ĐỂ BẮT ĐẦU TRAIN
        :param negative_image_path: Thư mục chứa ảnh Âm
        :param base_dir: Thư mục chứa dữ liệu huấn luyện.
        :param num_stages: Số tầng huấn luyện.
        :param sample_size: Kích thước ảnh mẫu (32x32).
        :param train_data_dir: Ví trí để thư mục cần huấn luyện, mặc định là trong thư mục để dữ liệu, bạn có thể chỉnh tham số này để custom (PATH).
        :param feature_type: Loại đặc trưng sử dụng (HAAR, LBP, HOG).
        :param stage_type: Kiểu tầng huấn luyện (BOOST, WAB).
        :param boost_type: Kiểu boosting (GAB, DAB, RAB).
        :param mode: Chế độ huấn luyện (BASIC, CORE, ALL).
        """
        self.base_dir = os.path.abspath(base_dir)
        self.sample_size =sample_size
        self.feature_type = feature_type
        self.stage_type = stage_type
        self.boost_type = boost_type
        self.num_stages = num_stages
        self.mode = mode 
        if negative_image_path == "":
            log_print("Không Có Thư Mục Ảnh Âm Tính, Đang Dùng Ảnh Mặc Định (Khoảng 3000 Ảnh Âm)",2)
            self.negative_image_dir = os.path.join(get_pak_path(),"bin","haarcascade-negatives-master","images")
        else:     
            self.negative_image_dir = os.path.abspath(negative_image_path)
        log_print("Thư Mục Chứa Ảnh Âm Ở: "+self.negative_image_dir)
        if train_data_path == "":
            self.train_data_dir = os.path.join(base_dir,"TrainData")
        else:     
            self.train_data_dir = os.path.abspath(train_data_path)
        self.positive_images_dir = os.path.join(self.train_data_dir, "positive_images")
        self.negative_images_dir = os.path.join(self.train_data_dir, "negative_images")
        self.positive_info_file = os.path.join(self.train_data_dir, "positive_info.txt")
        self.negative_info_file = os.path.join(self.train_data_dir, "negative_images.txt")
        self.vec_file = os.path.join(self.train_data_dir, "samples.vec")
        self.cascade_dir = os.path.join(self.train_data_dir, "cascade")
        self._setup_directories()
    
    def _setup_directories(self):
        os.makedirs(self.positive_images_dir, exist_ok=True)
        os.makedirs(self.negative_images_dir, exist_ok=True)
        os.makedirs(self.cascade_dir, exist_ok=True)
        
        for file in [self.positive_info_file, self.negative_info_file, self.vec_file]:
            if not os.path.exists(file):
                open(file, 'w').close()
    
    def _is_image_file(self, filename):
        return filename.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".pgm"))
    
    def generate_positive_samples(self):
        cropped_images = []
        sample_width, sample_height = self.sample_size
        datadir = self.base_dir
        image_ann_data = extract_from_json(datadir)
        subdirs = os.listdir(self.base_dir)
        if "train" in subdirs and "valid" in subdirs:
            log_print("Phát Hiện Đây Là 1 COCO DATASET!")
            check_train_valid_json(self.base_dir)
            train_path = os.path.join(self.base_dir, "train")
            valid_path = os.path.join(self.base_dir, "valid")
            datadir = merge_coco_dirs(train_path,valid_path)
        image_ann_data = extract_from_json(datadir)
        # Xóa các file cũ trong thư mục positive_images
        for file in os.listdir(self.positive_images_dir):
            os.remove(os.path.join(self.positive_images_dir, file))

        totalImage  = len(image_ann_data)
        posimg =0
        for annotations, filename in image_ann_data:
            
            img_path = os.path.join(datadir, filename)
            if not os.path.exists(img_path):
                log_print(f"Warning: {img_path} not found!", 2)
                continue

            img = cv2.imread(img_path)
            if img is None:
                log_print(f"Error: Could not read {img_path}", 3)
                continue

            for (x, y, w, h) in annotations:
                cropped = img[y:y+h, x:x+w]

                if cropped.size == 0:
                    log_print(f"Warning: Invalid crop for {filename}, skipping.", 2)
                    continue

                # Tính tỷ lệ resize để phù hợp với sample_size mới
                h_ratio = sample_height / h
                w_ratio = sample_width / w
                scale = min(h_ratio, w_ratio)

                new_w = int(w * scale)
                new_h = int(h * scale)

                resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

                # Tính padding để đạt đúng kích thước (sample_width, sample_height)
                pad_w = (sample_width - new_w) // 2
                pad_h = (sample_height - new_h) // 2

                padded = cv2.copyMakeBorder(resized, pad_h, sample_height - new_h - pad_h,
                                            pad_w, sample_width - new_w - pad_w,
                                            cv2.BORDER_CONSTANT, value=[128, 128, 128])

                # Cập nhật lại tọa độ object sau khi padding
                new_x = pad_w
                new_y = pad_h
                new_w = resized.shape[1]
                new_h = resized.shape[0]

                cropped_images.append((padded, f"positive_{len(cropped_images)}.png", new_x, new_y, new_w, new_h))
            posimg+=1
            log_progress(posimg,totalImage)

        # Ghi file info.txt với tọa độ mới sau khi padding
        with open(self.positive_info_file, "w") as f:
            for i, (padded, filename, new_x, new_y, new_w, new_h) in enumerate(cropped_images):
                save_path = os.path.join(self.positive_images_dir, filename)
                cv2.imwrite(save_path, padded)
                f.write(f"positive_images/{filename} 1 {new_x} {new_y} {new_w} {new_h}\n")
    
        return self.sample_size  # Trả về tuple (width, height)
    
    def prepare_negative_samples(self):
        log_print("Đang Chuẩn Bị Ảnh Âm Tính...")
        negative_source = self.negative_image_dir
        if not os.path.exists(negative_source) or not os.listdir(negative_source):
            log_print("Please add images to 'Negative Images' folder!", 3)
            return
    
        # Xóa ảnh cũ
        for file in os.listdir(self.negative_images_dir):
            os.remove(os.path.join(self.negative_images_dir, file))
    
        negative_files = [file for file in os.listdir(negative_source) if self._is_image_file(file)]
        total_files = len(negative_files)
    
        with open(self.negative_info_file, 'w') as f:
            for idx, file in enumerate(negative_files, 1):
                shutil.copy2(os.path.join(negative_source, file), self.negative_images_dir)
                f.write(f"{os.path.join(self.negative_images_dir, file)}\n")
                log_progress(idx, total_files)  # Hiển thị tiến trình
    
        log_print("Hoàn thành việc chuẩn bị ảnh âm tính!")

    
    def prepare_data(self):
        """ Chuẩn bị ảnh dương tính (positive) và ảnh âm tính (negative) """
        log_print("Bắt đầu chuẩn bị dữ liệu...", 1)
        # Tạo mẫu ảnh dương tính (positive)
        max_size = self.generate_positive_samples()
        log_print(f"Đã chuẩn bị xong ảnh dương tính! Kích thước mẫu: {max_size}")
        # Chuẩn bị ảnh âm tính (negative)
        self.prepare_negative_samples()
        log_print("Đã chuẩn bị xong ảnh âm tính!")
        return max_size
    
    def train(self, CustomTrainARGS=[], CustomSamplesARGS=[]):
        """
        Huấn luyện bộ phân loại Haar Cascade bằng OpenCV.

        - Nếu CustomTrainARGS hoặc CustomSamARGS được cung cấp, sẽ thêm chúng vào tham số chạy.
        - Tạo các mẫu dương (positive samples) từ dữ liệu anotated.
        - Chuẩn bị ảnh tiêu cực (negative samples).
        - Chạy opencv_createsamples.exe để tạo tập dữ liệu huấn luyện.
        - Chạy opencv_traincascade.exe để huấn luyện mô hình.
        """
        
        num_pos = count_image_files(self.positive_images_dir)
        num_neg = count_image_files(self.negative_images_dir)

        ExeRunner("opencv_createsamples.exe", [
            "-info", self.positive_info_file,
            "-num", str(num_pos),
            "-w", str(self.sample_size[0]),
            "-h", str(self.sample_size[1]),
            "-vec", self.vec_file,
            "-inv","-randinv"
        ]+CustomSamplesARGS)

        ExeRunner("opencv_traincascade.exe", [
            "-data", self.cascade_dir,
            "-vec", self.vec_file,
            "-bg", self.negative_info_file,
            "-numPos", str(int(num_pos*0.9)),
            "-numNeg", str(num_neg),
            "-numStages", str(self.num_stages),
            "-w", str(self.sample_size[0]),
            "-h", str(self.sample_size[1]),
            "-featureType", self.feature_type.value,
            "-stageType", self.stage_type.value,
            "-bt", self.boost_type.value,
            "-mode", self.mode.value
        ]+CustomTrainARGS)
        log_print("Training completed!")
        log_print("Casade Saved In: "+self.cascade_dir)
        
        
