import os
import logging
import shutil
from datetime import datetime
from colorama import init, Fore, Style
import sys
from os.path import abspath
init(autoreset=True)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: Fore.BLUE,      # Xanh dương
        logging.WARNING: Fore.YELLOW, # Vàng
        logging.ERROR: Fore.RED       # Đỏ
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelno, Fore.MAGENTA)  
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{timestamp} {log_color}[{record.levelname}]{Fore.RESET} {record.getMessage()}"


console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())


logger.handlers.clear()
logger.addHandler(console_handler)

def get_pak_path() -> str:
    """Trả về đường dẫn của package"""
    return os.path.dirname(os.path.dirname(__file__))

def log_print(text, mode=1):
    """
    Ghi log với các chế độ:
    - mode = 1: INFO (Màu xanh)
    - mode = 2: WARNING (Màu vàng)
    - mode = 3: ERROR (Màu đỏ)
    """
    if mode == 1:
        logger.info(text)
    elif mode == 2:
        logger.warning(text)
    elif mode == 3:
        logger.error(text)
        exit(0)
    else:
        logger.info(f"[UNKNOWN MODE] {text}")
        
def log_progress(current, total):  
    percentage = (current / total) * 100
    bar_length = 50
    filled_length = int(bar_length * current // total)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    if(current >= total):      
        sys.stdout.write(f"\r{Fore.CYAN}[PROCESSING] |{bar}| {current}/{total} ({percentage:.2f}%) {Style.RESET_ALL}\n")
    else:
        sys.stdout.write(f"\r{Fore.CYAN}[PROCESSING] |{bar}| {current}/{total} ({percentage:.2f}%) {Style.RESET_ALL}")
    sys.stdout.flush()


def count_image_files(directory, valid_extensions={".jpg", ".jpeg", ".png", ".ppm", ".pgm"}):
    """
    Đếm tất cả các tệp ảnh trong một thư mục nào đó.

    :param directory: Đường dẫn tới thư mục cần đếm tệp ảnh.
    :param valid_extensions: Tập hợp các phần mở rộng hợp lệ của tệp ảnh.
    :return: Số lượng tệp ảnh trong thư mục.
    """
    directory = abspath(directory)
    count = 0
    if not os.path.exists(directory):
        log_print(f"⚠️ Thư mục '{directory}' không tồn tại!",3)
        return 0  

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            count += 1
    print(count)
    return count 

def is_image_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".pgm"))


def move_image_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for file_name in os.listdir(src_dir):
        src_path = os.path.join(src_dir, file_name)
        dest_path = os.path.join(dest_dir, file_name)
        if os.path.isfile(src_path) and is_image_file(file_name):
            shutil.move(src_path, dest_path)



    