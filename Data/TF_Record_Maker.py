import os
import json
import tensorflow as tf
from object_detection.utils import dataset_util
from PIL import Image
from ..MISC.Tool import  *
from ..MISC.pipeline_Maker import *


class PipelineConfigParams:
    def __init__(self,  model_resolution=320, batch_size=32, learning_rate_base=0.08, 
                 total_steps=50000, label_map_path=None, train_input_path=None,  vaild_input_path=None, 
                 custom_config_file=None):
        """
        Lớp chứa các tham số để tạo pipeline.config.

        Tham số:
        - model_resolution (int): Độ phân giải mô hình (320 hoặc 640).
        - batch_size (int): Kích thước batch khi train.
        - learning_rate_base (float): Learning rate ban đầu.
        - total_steps (int): Tổng số bước train.
        - label_map_path (str): Đường dẫn đến file label_map.
        - input_path (str): Đường dẫn đến TFRecord.
        - custom_config_file (str, optional): Nếu có, sử dụng file này làm pipeline.config.
        """
        self.model_resolution = model_resolution
        self.batch_size = batch_size
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.label_map_path = os.path.abspath(label_map_path) if label_map_path else None
        self.train_input_path = os.path.abspath(train_input_path) if train_input_path else None
        self.vaild_input_path = os.path.abspath(vaild_input_path) if vaild_input_path else None
        self.custom_config_file = os.path.abspath(custom_config_file) if custom_config_file else None

def create_tf_example(image_path, annotations, category_mapping)-> str:
    """
    Tạo tf.train.Example từ image và annotations.
    - category_mapping: dict mapping từ original category id -> (new id, category name)
    """
    try:
        with tf.io.gfile.GFile(image_path, 'rb') as fid:
            encoded_image = fid.read()
        image = Image.open(image_path)
        width, height = image.size
    except Exception as e:
        log_print(f"⚠️ Error reading {image_path}: {e}", 3)
        return None

    filename = os.path.basename(image_path).encode('utf8')
    image_format = b'jpeg' if image_path.lower().endswith(('jpg', 'jpeg')) else b'png'

    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    classes_text, classes = [], []

    for ann in annotations:
        x, y, w, h = ann['bbox']
        xmins.append(x / width)
        xmaxs.append((x + w) / width)
        ymins.append(y / height)
        ymaxs.append((y + h) / height)

        old_cat_id = ann['category_id']
        if old_cat_id not in category_mapping:
            continue

        new_cat_id, cat_name = category_mapping[old_cat_id]
        classes_text.append(cat_name.encode('utf8'))
        classes.append(new_cat_id)

    if not classes:
        return None

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



def create_label_map(save_path, global_categories):
    """
    Tạo file label_map.pbtxt và mapping từ original category id sang (new id, category name).
    Lưu ý: new id bắt đầu từ 1 vì 0 dành cho background.
    """
    label_map_path = os.path.join(save_path, "label_map.pbtxt")
    category_mapping = {}  
    seen_names = set()
    new_id = 1  

    for old_id, cat_name in sorted(global_categories.items()):
        if old_id == 0:
            log_print(f"⚠️ Ignored category_id=0 (Invalid) for label_map", 2)
            continue
        if cat_name in seen_names:
            log_print(f"⚠️ Duplicate category name '{cat_name}', skipping duplicate", 2)
            continue
        seen_names.add(cat_name)
        category_mapping[old_id] = (new_id, cat_name)
        new_id += 1

    with open(label_map_path, "w") as f:
        for old_id, (new_id, cat_name) in sorted(category_mapping.items(), key=lambda x: x[1][0]):
            f.write("item {\n")
            f.write(f"  id: {new_id}\n")
            f.write(f"  name: '{cat_name}'\n")
            f.write("}\n")
    
    log_print(f"✅ Saved label map to {label_map_path} with {len(category_mapping)} classes")
    return category_mapping 




def convert_coco_to_tfrecord(input_dir, save_path="TF_DATASET", pipeline_params=None):
    """
    Chuyển đổi COCO dataset thành TFRecord.

    Tham số:
    - input_dir (str): Thư mục chứa dữ liệu COCO.
    - save_path (str): Thư mục lưu TFRecord.
    - pipeline_params (PipelineConfigParams, optional): Tham số pipeline.
    """
    
   
    input_dir = os.path.abspath(input_dir)
    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    
    if pipeline_params == None:
      pipeline_params = PipelineConfigParams()
      
    if pipeline_params.train_input_path==None:
        pipeline_params.train_input_path = os.path.join(save_path,"train.tfrecord")
    if pipeline_params.vaild_input_path==None:
        pipeline_params.vaild_input_path = os.path.join(save_path,"valid.tfrecord")
    if pipeline_params.label_map_path==None:
        pipeline_params.label_map_path= os.path.join(save_path,"label_map.pbtxt")
    
    global_categories = {}
    for subset in ['train', 'valid']:
        images_dir = os.path.join(input_dir, subset)
        json_files = [f for f in os.listdir(images_dir) if f.endswith('.json')]
        if not json_files:
            log_print(f"⚠️ ERROR: No JSON found in {images_dir}",3)
            continue

        json_path = os.path.join(images_dir, json_files[0])
        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        for cat in coco_data.get('categories', []):
            global_categories[cat['id']] = cat['name']

    category_mapping = create_label_map(save_path, global_categories)
    
    for subset in ['train', 'valid']:
        images_dir = os.path.join(input_dir, subset)
        json_files = [f for f in os.listdir(images_dir) if f.endswith('.json')]
        if not json_files:
            continue

        json_path = os.path.join(images_dir, json_files[0])
        output_path = os.path.join(save_path, f'{subset}.tfrecord')

        with open(json_path, 'r') as f:
            coco_data = json.load(f)

        writer = tf.io.TFRecordWriter(output_path)
        count = 0
        total_images = len(coco_data.get('images', []))

        image_annotations = {ann['image_id']: [] for ann in coco_data.get('annotations', [])}
        for ann in coco_data.get('annotations', []):
            image_annotations[ann['image_id']].append(ann)

        for i, img in enumerate(coco_data.get('images', []), 1):
            image_path = os.path.join(images_dir, img['file_name'])
            if not os.path.exists(image_path):
                log_print(f"⚠️ Missing image: {image_path}",2)
                continue

            tf_example = create_tf_example(image_path, image_annotations.get(img['id'], []), category_mapping)
            if tf_example:
                writer.write(tf_example.SerializeToString())
                count += 1
            log_progress(i, total_images)
        
        writer.close()
        print(f"✅ Saved {subset} TFRecord ({count} images) to {output_path}")

    if category_mapping and pipeline_params:
        num_classes = max([new_id for (new_id, _) in category_mapping.values()], default=0)
        return save_path, pipeline_params, num_classes
