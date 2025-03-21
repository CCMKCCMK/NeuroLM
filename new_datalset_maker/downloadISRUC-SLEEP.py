import os
import requests
import shutil
import tarfile
import zipfile
import rarfile
import concurrent.futures
import patoolib
import time
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 基础URL
# BASE_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/"
# BASE_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupII/"
BASE_URL = "http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/"

# n = 100
# n = 8
n = 10

# 设置下载和解压目录
DOWNLOAD_DIR = "downloads"
EXTRACT_DIR = "extracted"
ORGANIZED_DIR = "organized"

# 创建必要的目录
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(ORGANIZED_DIR, exist_ok=True)

def download_file(url, local_filename):
    """下载文件并显示进度"""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=local_filename) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            return local_filename
    except Exception as e:
        logger.error(f"下载 {url} 时出错: {str(e)}")
        return None

def extract_archive(archive_path, extract_to):
    """解压档案文件"""
    try:
        logger.info(f"正在解压 {archive_path}")
        
        if archive_path.endswith('.rar'):
            patoolib.extract_archive(archive_path, outdir=extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.tar') or '.tar.' in archive_path:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
                
        logger.info(f"成功解压 {archive_path}")
        return True
    except Exception as e:
        logger.error(f"解压 {archive_path} 时出错: {str(e)}")
        return False

def download_and_extract(file_number):
    """下载并解压单个文件"""
    file_url = f"{BASE_URL}{file_number}.rar"
    local_file = os.path.join(DOWNLOAD_DIR, f"{file_number}.rar")
    extract_folder = os.path.join(EXTRACT_DIR, str(file_number))
    
    # 下载文件
    if not os.path.exists(local_file):
        logger.info(f"下载 {file_url}")
        downloaded_file = download_file(file_url, local_file)
        if not downloaded_file:
            return False, file_number
    else:
        logger.info(f"文件 {local_file} 已存在，跳过下载")
    
    # 解压文件
    if not os.path.exists(extract_folder) or not os.listdir(extract_folder):
        os.makedirs(extract_folder, exist_ok=True)
        if not extract_archive(local_file, extract_folder):
            return False, file_number
    else:
        logger.info(f"文件夹 {extract_folder} 已存在，跳过解压")
    
    return True, file_number

def process_extracted_files():
    """处理解压后的文件，按照要求整理文件结构"""
    logger.info("开始整理文件结构")
    
    # 遍历解压目录
    for root, dirs, files in os.walk(EXTRACT_DIR):
        # 跳过根目录
        if root == EXTRACT_DIR:
            continue
        
        # 获取当前文件夹路径
        parts = root.split(os.path.sep)
        
        # 只处理含有.rec和_1.txt文件的文件夹
        rec_files = [f for f in files if f.endswith('.rec')]
        txt_files = [f for f in files if f.endswith('_1.txt')]
        
        if not rec_files or not txt_files:
            continue
        
        # 确定目标文件夹名称
        main_number = parts[1]  # 如 "1", "8", "9", "32"
        
        if len(parts) > 3:  # 如 "extracted/1/1/1" 或 "extracted/1/1/2"
            target_folder = f"{main_number}_{parts[-1]}"
        else:  # 如 "extracted/9/9" 或 "extracted/32/32"
            target_folder = main_number
        
        target_path = os.path.join(ORGANIZED_DIR, target_folder)
        os.makedirs(target_path, exist_ok=True)
        
        # 复制所有文件到目标文件夹
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_path, file)
            shutil.copy2(src_file, dst_file)
            logger.info(f"复制文件: {src_file} -> {dst_file}")
        
        # 创建label.txt文件(复制*_1.txt文件)
        txt_file = next((f for f in files if f.endswith('_1.txt')), None)
        if txt_file:
            src_file = os.path.join(root, txt_file)
            label_file = os.path.join(target_path, 'label.txt')
            shutil.copy2(src_file, label_file)
            logger.info(f"创建label.txt: {src_file} -> {label_file}")

def main():
    # 并行下载和解压
    file_numbers = list(range(1, n+1))
    success_count = 0
    failed_numbers = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_number = {executor.submit(download_and_extract, num): num for num in file_numbers}
        
        for future in concurrent.futures.as_completed(future_to_number):
            success, number = future.result()
            if success:
                success_count += 1
            else:
                failed_numbers.append(number)
    
    logger.info(f"成功处理 {success_count} 个文件")
    if failed_numbers:
        logger.warning(f"失败的文件编号: {failed_numbers}")
    
    # Delete all the downloaded files
    shutil.rmtree(DOWNLOAD_DIR)

    # 整理文件结构
    process_extracted_files()

    # Delete all the extracted files
    shutil.rmtree(EXTRACT_DIR)

    # Rename the organized folder
    os.rename(ORGANIZED_DIR, BASE_URL.split("/")[-2])
    
    logger.info("任务完成!")

if __name__ == "__main__":
    main()