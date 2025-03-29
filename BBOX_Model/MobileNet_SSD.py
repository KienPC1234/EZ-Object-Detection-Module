import os
import shutil
from object_detection import model_lib_v2, exporter_lib_v2
from object_detection.protos import pipeline_pb2
from ..Data.TF_Record_Maker import convert_coco_to_tfrecord, PipelineConfigParams
from google.protobuf import text_format
from ..MISC.pipeline_Maker import generate_mobilenet_v2_pipeline
from ..MISC.TFBaseModel import BASE_MODEL
from ..Data.COCO import check_train_valid_json, LABELME_to_coco
from ..MISC.Tool import  *
import tempfile
import tensorflow as tf
import webbrowser
import psutil
import subprocess


def create_mbnetv2_pipeline_config(savepath,params: PipelineConfigParams, num_classes):
    """
    Tạo pipeline.config dựa trên tham số từ PipelineConfigParams.
    
    - Nếu `custom_config_file` được cung cấp, sao chép nó thay vì tạo mới.
    - Nếu không, tự động sinh pipeline.config mới.

    Trả về:
    - Đường dẫn đến file pipeline.config đã tạo hoặc sao chép.
    """
    config_path = os.path.join(savepath, "pipeline.config")
    os.makedirs(savepath, exist_ok=True)

    if params.custom_config_file:
        if not os.path.isfile(params.custom_config_file):
            raise FileNotFoundError(f"Custom config file không tồn tại: {params.custom_config_file}")
        shutil.copy(params.custom_config_file, config_path)
        print(f"📄 Đã sao chép file pipeline từ: {params.custom_config_file}")
        return config_path

    if params.model_resolution == 320:
        fineTunePath = os.path.join(get_pak_path(), "bin", "MobileNetV2_SSD",
                                    "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8",
                                    "checkpoint", "ckpt-0")
    elif params.model_resolution == 640:
        fineTunePath = os.path.join(get_pak_path(), "bin", "MobileNetV2_SSD",
                                    "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8",
                                    "checkpoint", "ckpt-0")
    else:
        raise ValueError("Unsupported model resolution. Choose either 320 or 640.")

    pipeline_config = generate_mobilenet_v2_pipeline(
        resolution=params.model_resolution,
        num_classes=num_classes,
        fine_tune_checkpoint=fineTunePath,
        batch_size=params.batch_size,
        learning_rate_base=params.learning_rate_base,
        total_steps=params.total_steps,
        label_map_path=params.label_map_path,
        train_input_path=params.train_input_path,
        vaild_input_path=params.vaild_input_path
    )

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(pipeline_config)

    log_print(f"✅ Pipeline config đã được tạo tại: {config_path}")
    return config_path




class MobileNetV2_SSD_FPNLite(BASE_MODEL):
    def __init__(self, datapath, savepath=None): 
        super().__init__(datapath,savepath,"MobileNetV2_SSD_FPNLite")

    def prepare_data(self, pipeline_params=None):
        """ Chuyển đổi dataset sang định dạng TFRecord & tạo pipeline.config """
        if os.path.isdir(self.datapath):
            subdirs = os.listdir(self.datapath)
            if "train" in subdirs and "valid" in subdirs:
                log_print("📂 Phát hiện đây là 1 COCO DATASET!")
                check_train_valid_json(self.datapath)
                save_path, pipeline, num_classes = convert_coco_to_tfrecord(
                    self.datapath, self.savepath, pipeline_params)
            else:
                TEMP_COCO = os.path.join(tempfile.gettempdir(), "CocoTEMP")
                log_print("📂 Phát hiện đây là LabelME Dataset, chuyển đổi sang COCO...")
                LABELME_to_coco(self.datapath, TEMP_COCO)
                save_path, pipeline, num_classes = convert_coco_to_tfrecord(
                    TEMP_COCO, self.savepath, pipeline_params)
                shutil.rmtree(TEMP_COCO)

            create_mbnetv2_pipeline_config(save_path, pipeline, num_classes)



        

class MobileNet(BASE_MODEL):
    def __init__(self, datapath, savepath=None): 
        super().__init__(datapath,savepath,"MobileNet")
    
    
    