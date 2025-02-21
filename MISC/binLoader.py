import os
import json
import subprocess
from .Tool import *
import sys
path = get_pak_path()
path = os.path.join(path,"bin")
config_path = os.path.join(path, "CFG.json")

def load_bin_paths():
    """ Đọc danh sách thư mục chứa file exe từ CFG.json """
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config.get("BinPath", [])
    else:
        log_print("Không Tìm Đượn CFG Json!",3)
    return []




def ExeRunner(FileName, Args=None):
    """ 
    Chạy file exe trong thư mục bin với các đối số (nếu có). 
    Nếu ứng dụng là console app, in trực tiếp ra stdout.
    """
    if Args is None:
        Args = []
    
    bin_dirs = load_bin_paths()  # Lấy danh sách thư mục chứa exe
    for bin_dir in bin_dirs:
        exe_path = os.path.join(path, bin_dir, FileName)
        
        if os.path.exists(exe_path):
            try:
                log_print("Đang Chạy: "+str([exe_path] + Args))
                # Xác định app có phải console app không
                is_console = sys.stdout.isatty()  # True nếu đang chạy trong terminal

                if is_console:
                    # Nếu là console app, in trực tiếp ra stdout
                    process = subprocess.run([exe_path] + Args, check=True)
                else:
                    # Nếu không, chỉ log output
                    process = subprocess.run(
                        [exe_path] + Args, check=True, 
                        capture_output=True, text=True
                    )
                    log_print(process.stdout)

                log_print("Đã Chạy Xong "+FileName+"!")
                return True
            except subprocess.CalledProcessError as e:
                log_print(f"Lỗi khi chạy {FileName}: {e}", 3)
                return False

    log_print(f"Không tìm thấy {FileName} trong bin.", 3)
    return False

