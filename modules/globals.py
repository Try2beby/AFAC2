import os
import sys
from loguru import logger
from datetime import datetime  # 导入datetime模块

LOG_DIR = "./logs"

# 获取脚本的文件名（不包括扩展名）
script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# 获取当前日期和时间，格式化为字符串
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")

# 使用脚本的文件名和当前日期时间作为日志文件的文件名
log_filename = f"{script_name}_{current_time}.log"

# 设置日志级别
logger.remove()
logger.add(sys.stderr, level="INFO")

# 创建一个处理器，将日志输出到文件
logger.add(os.path.join(LOG_DIR, log_filename), level="INFO")
