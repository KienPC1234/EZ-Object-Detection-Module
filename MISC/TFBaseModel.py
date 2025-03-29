

import os
from object_detection import  exporter_lib_v2,model_lib_v2
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from ..MISC.Tool import  *
import tensorflow as tf
import webbrowser
import psutil
import subprocess
from abc import ABC, abstractmethod

class BASE_MODEL:
    def __init__(self, datapath, savepath=None,modelName=None): 
        self.ModelName = "BaseModel" if not modelName else modelName.replace(" ", "_")
        """
        Khởi tạo MobileNetV2 SSD với đường dẫn dữ liệu và nơi lưu mô hình.

        Args:
            datapath (str): Đường dẫn tới dataset (COCO hoặc LabelMe).
            savepath (str, optional): Đường dẫn để lưu model và checkpoint. Mặc định: "MobileNetV2SSD_DATASET".
        """
        self.datapath = os.path.abspath(datapath)
        self.modelcp = None
        self.tensorboard_process = None  # Quản lý TensorBoard

        if savepath is None:
            self.savepath = os.path.abspath(self.ModelName+"_SAVE")
        else:
            self.savepath = os.path.abspath(savepath)

        os.makedirs(self.savepath, exist_ok=True)

    @abstractmethod
    def prepare_data(self, pipeline_params):
        pass
    
    def is_tensorboard_running(self, port=6006):
        """ Kiểm tra xem TensorBoard có đang chạy trên cổng chỉ định không """
        for process in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
            if process.info['name'] and "tensorboard" in process.info['name'].lower():
                if any(f"--port={port}" in cmd for cmd in process.info['cmdline']):
                    return True
        return False

    def start_tensorboard(self, logdir, port=6006):
        """ Khởi động TensorBoard nếu chưa chạy """
        if self.is_tensorboard_running(port):
            log_print(f"⚠️ TensorBoard đã chạy trên cổng {port}, bỏ qua khởi động lại.")
            return

        self.tensorboard_process = subprocess.Popen(
            ["tensorboard", f"--logdir={logdir}", f"--port={port}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        log_print(f"✅ TensorBoard đang chạy tại: http://localhost:{port}")

        try:
            webbrowser.open(f"http://localhost:{port}")
        except:
            log_print("⚠️ Không thể mở trình duyệt tự động, vui lòng truy cập thủ công.")

    def stop_tensorboard(self):
        """ Dừng TensorBoard khi chương trình kết thúc """
        if self.tensorboard_process:
            log_print("🛑 Đang tắt TensorBoard...")
            self.tensorboard_process.terminate()
            self.tensorboard_process.wait()
            log_print("✅ TensorBoard đã dừng.")

    def train(self, modelCheckPointDir=None, openTensorBroad=False, TensorBroadPort=6006,
             checkpoint_every_n=1000, checkpoint_max_to_keep=7, use_tpu=False, **kwargs):
       """ Huấn luyện mô hình """
       if modelCheckPointDir is None:
           MODEL_DIR = os.path.join(self.savepath, "ModelCheckPoint")
       else:
           MODEL_DIR = os.path.abspath(modelCheckPointDir)
       self.modelcp = MODEL_DIR
       if openTensorBroad:
           logdir = os.path.join(MODEL_DIR, "train")
           self.start_tensorboard(logdir, TensorBroadPort)
       if tf.config.list_physical_devices('GPU'):
           device_name = tf.test.gpu_device_name()
           log_print(f"🚀 Training on GPU: {device_name}")
       elif tf.config.list_physical_devices('TPU'):
           log_print("⚡ Training on TPU")
       else:
           log_print("💻 Training on CPU")
       try:
           model_lib_v2.train_loop(
               pipeline_config_path=os.path.join(self.savepath, "pipeline.config"),
               model_dir=MODEL_DIR,
               use_tpu=use_tpu,
               checkpoint_every_n=checkpoint_every_n,
               checkpoint_max_to_keep=checkpoint_max_to_keep,
               **kwargs
           )
           log_print("✅ Đã Hoàn Thành Train!")
       except KeyboardInterrupt:
           log_print("\n🛑 Dừng training do người dùng bấm Ctrl+C!")
       finally:
           self.stop_tensorboard()  
    

    def export_model(self, export_dir=None, checkpoint_path=None, get_latest_checkpoint=True,input_type="image_tensor",**kwargs):
        """ Xuất model đã train ra SavedModel format """

        if export_dir is None:
            export_dir = os.path.join(self.savepath, "exported_model")
        else:
            export_dir = os.path.abspath(export_dir)
        os.makedirs(export_dir, exist_ok=True)

        if checkpoint_path:
            checkpoint_path = os.path.abspath(checkpoint_path)
        elif get_latest_checkpoint:
            checkpoint_dirs = [self.modelcp, os.path.join(self.savepath, "ModelCheckPoint")]
            checkpoint_dirs = [os.path.abspath(d) for d in checkpoint_dirs if d]

            for path in checkpoint_dirs:
                if path and os.path.exists(path):
                    latest_ckpt = tf.train.latest_checkpoint(path)
                    if latest_ckpt:
                        checkpoint_path = latest_ckpt
                        break 


        if checkpoint_path is None:
            log_print("⚠️ Không tìm thấy checkpoint để xuất model!")
            return

        if not checkpoint_path.endswith(".index") and not checkpoint_path.endswith(".data-00000-of-00001"):
            checkpoint_base = checkpoint_path
            index_file = checkpoint_base + ".index"
            data_file = checkpoint_base + ".data-00000-of-00001"

            if os.path.exists(index_file) and os.path.exists(data_file):
                checkpoint_path = checkpoint_base
            else:
                log_print(f"⚠️ Checkpoint {checkpoint_base} không hợp lệ! Thiếu file .index hoặc .data")
                return

        log_print(f"📌 Đang sử dụng checkpoint: {checkpoint_path}")

        pipeline_config_path = os.path.join(self.savepath, "pipeline.config")
        pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()

        with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
            proto_str = f.read()
            text_format.Merge(proto_str, pipeline_config)

        exporter_lib_v2.export_inference_graph(
            input_type=input_type,
            pipeline_config=pipeline_config,  
            trained_checkpoint_dir=os.path.dirname(checkpoint_path),
            output_directory=export_dir,
            **kwargs
        )

        log_print(f"✅ Model đã được xuất tại: {export_dir}")

    def evaluate(self, checkpoint_path=None,get_latest_checkpoint=True, eval_timeout=None, eval_interval=300,**kwargs):
        """
        Đánh giá mô hình với bộ dữ liệu validation.
    
        Args:
            checkpoint_dir (str, optional): Đường dẫn tới checkpoint cần đánh giá. Mặc định: checkpoint mới nhất trong thư mục model.
            eval_timeout (int, optional): Thời gian tối đa chạy evaluation (giây). Nếu None, sẽ chạy mãi.
            eval_interval (int, optional): Khoảng thời gian giữa các lần đánh giá (giây). Mặc định: 300 giây (5 phút).
        """
        if checkpoint_path:
            checkpoint_path = os.path.abspath(checkpoint_path)
        elif get_latest_checkpoint:
            checkpoint_dirs = [self.modelcp, os.path.join(self.savepath, "ModelCheckPoint")]
            checkpoint_dirs = [os.path.abspath(d) for d in checkpoint_dirs if d]

            for path in checkpoint_dirs:
                if path and os.path.exists(path):
                    latest_ckpt = tf.train.latest_checkpoint(path)
                    if latest_ckpt:
                        checkpoint_path = latest_ckpt
                        break 


        if checkpoint_path is None:
            log_print("⚠️ Không tìm thấy checkpoint để xuất model!")
            return

        if not checkpoint_path.endswith(".index") and not checkpoint_path.endswith(".data-00000-of-00001"):
            checkpoint_base = checkpoint_path
            index_file = checkpoint_base + ".index"
            data_file = checkpoint_base + ".data-00000-of-00001"

            if os.path.exists(index_file) and os.path.exists(data_file):
                checkpoint_path = checkpoint_base
            else:
                log_print(f"⚠️ Checkpoint {checkpoint_base} không hợp lệ! Thiếu file .index hoặc .data")
                return

        log_print(f"📌 Đang sử dụng checkpoint: {checkpoint_path}")
        
        try:
            model_lib_v2.eval_continuously(
                pipeline_config_path=os.path.join(self.savepath, "pipeline.config"),
                model_dir=self.savepath,
                checkpoint_dir=os.path.dirname(checkpoint_path),
                wait_interval=eval_interval,
                timeout=eval_timeout,
                **kwargs
            )
            log_print("✅ Đánh giá mô hình hoàn tất!")
        except KeyboardInterrupt:
            log_print("\n🛑 Đánh giá bị hủy bởi người dùng!")
    
    def summary(self):
        """ Hiển thị thông tin tổng quan về mô hình """
        log_print(f"\n📌 **{self.ModelName} Summary**")
        log_print(f"📂 Dataset Path: {self.datapath}")
        log_print(f"💾 Save Path: {self.savepath}")
        log_print(f"📌 Checkpoint Path: {self.modelcp if self.modelcp else 'Chưa có checkpoint nào'}")
        log_print(f"📊 TensorBoard: {'Đang chạy' if self.is_tensorboard_running() else 'Không chạy'}")