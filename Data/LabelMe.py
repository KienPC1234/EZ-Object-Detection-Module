import subprocess
import sys
import os
from ..MISC.Tool import *

def run_labelme():
    """Run App Label Me For Models That Require Multiple Classes"""
    scripts_dir = os.path.join(sys.exec_prefix, 'Scripts')
    labelme_dir = os.path.join(scripts_dir,"labelme")
    try:
        subprocess.run([labelme_dir, "--version"], check=True)
    except subprocess.CalledProcessError:
        log_print("Đang cài đặt labelme bằng pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "labelme"], check=True)

    log_print("Chạy Labelme...")
    labelme_dir = os.path.join(scripts_dir,"labelme")
    log_print(f"LabelMe Ở {labelme_dir}")

    subprocess.run([labelme_dir, "--autosave", "--nodata"])
