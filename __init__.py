import configparser
import os
from .MISC.Tool import *

config_path = os.path.join(os.getcwd(), "config.ini")

config = configparser.ConfigParser()
if os.path.exists(config_path):
    config.read(config_path)
    Credits = config.getboolean("settings", "credits", fallback=True)
else:
    Credits = True
    config["settings"] = {"credits": str(Credits)}
    with open(config_path, "w") as configfile:
        config.write(configfile)
if Credits:
    
    log_print(" "+"-" * 86,2)
    log_print(" | Edit 'config.ini' and set 'credits = false' in [settings] to disable this message. |",2)
    log_print(" | Ex: [settings]                                                                     |",2)
    log_print(" |     credits = false                                                                |",2)
    log_print(" "+"-" * 86 + "\n",2)
    
    print("""
    ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░       ░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░ 
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░       ░▒▓█▓▒▒▓█▓▒░  
    ░▒▓███████▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░  ░▒▓█▓▒▒▓█▓▒░  
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▓█▓▒░   
    ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░        ░▒▓█▓▓█▓▒░   
    ░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░       ░▒▓███████▓▒░░▒▓████████▓▒░  ░▒▓██▓▒░    

    Powered By KienCore Develop (Kien TensorFlow)  
                                       
    """)
