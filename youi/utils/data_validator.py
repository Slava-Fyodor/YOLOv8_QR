import os
import glob
from typing import List, Dict, Tuple


def validate_yolo_dataset(images_dir: str, labels_dir: str = None) -> Dict:
    """
    验证YOLO格式数据集，检查图片与标签是否匹配。
    
    Args:
        images_dir: 图片目录路径
        labels_dir: 标签目录路径，如果为None则自动尝试查找
        
    Returns:
        验证结果字典，包含总图片数、匹配标签数、缺失标签的图片列表等信息
    """
    results = {
        "total_images": 0,
        "matched_labels": 0,
        "missing_labels": 0,
        "missing_images": [],
        "success": False,
        "message": ""
    }
    
    # 验证输入目录
    if not os.path.exists(images_dir):
        results["message"] = f"Каталог изображений не существует: {images_dir}"
        return results
    
    # 如果标签目录为空，尝试查找可能的标签目录
    if labels_dir is None:
        possible_label_dirs = [
            os.path.join(os.path.dirname(images_dir), 'labels'),
            os.path.join(os.path.dirname(os.path.dirname(images_dir)), 'labels', os.path.basename(images_dir)),
            os.path.join(images_dir.replace('images', 'labels')),
            os.path.join(os.path.dirname(images_dir), 'labels', os.path.basename(images_dir))
        ]
        
        for dir_path in possible_label_dirs:
            if os.path.exists(dir_path) and os.listdir(dir_path):
                labels_dir = dir_path
                results["message"] += f"Каталог меток найден автоматически: {labels_dir}\n"
                break
    
    if labels_dir is None or not os.path.exists(labels_dir):
        results["message"] = f"Каталог меток не существует или не найден: {labels_dir}"
        return results
    
    # 查找所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}"), recursive=False))
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}"), recursive=False))
    
    # 提取不带扩展名的图片basename
    image_basenames = [os.path.splitext(os.path.basename(img))[0] for img in image_files]
    results["total_images"] = len(image_files)
    
    if not image_files:
        results["message"] = f"В каталоге изображений не найдено файлов изображений: {images_dir}"
        return results
    
    # 检查每个图片是否有对应的标签
    for img_basename in image_basenames:
        label_path = os.path.join(labels_dir, f"{img_basename}.txt")
        if os.path.exists(label_path):
            results["matched_labels"] += 1
        else:
            results["missing_labels"] += 1
            results["missing_images"].append(img_basename)
    
    # 根据匹配率判断数据集有效性
    match_rate = results["matched_labels"] / results["total_images"] if results["total_images"] > 0 else 0
    
    if match_rate >= 0.9:
        results["success"] = True
        results["message"] = f"Набор данных корректен: найдено {results['total_images']} изображений, из них {results['matched_labels']} имеют соответствующие метки (доля совпадений: {match_rate:.2%})"
    elif match_rate > 0:
        results["success"] = True  # 部分成功也算成功，但会给出警告
        results["message"] = f"Набор данных частично корректен: найдено {results['total_images']} изображений, но только {results['matched_labels']} имеют соответствующие метки (доля совпадений: {match_rate:.2%})"
    else:
        results["success"] = False
        results["message"] = f"Набор данных некорректен: найдено {results['total_images']} изображений, но ни одно из них не имеет соответствующей метки"
    
    return results


def find_best_match_for_yolo(base_dir: str) -> Dict:
    """
    在给定目录下查找最佳的YOLO格式数据集配置。
    
    Args:
        base_dir: 要搜索的基础目录
        
    Returns:
        包含找到的最佳训练和验证集目录的字典
    """
    results = {
        "train_images": None,
        "train_labels": None,
        "val_images": None,
        "val_labels": None,
        "message": "",
        "success": False
    }
    
    # 检查是否是标准的YOLO结构
    standard_train_images = os.path.join(base_dir, 'images', 'train')
    standard_train_labels = os.path.join(base_dir, 'labels', 'train')
    standard_val_images = os.path.join(base_dir, 'images', 'val')
    standard_val_labels = os.path.join(base_dir, 'labels', 'val')
    
    # 首先检查标准目录结构
    if (os.path.exists(standard_train_images) and os.path.exists(standard_train_labels) and
        os.path.exists(standard_val_images) and os.path.exists(standard_val_labels)):
        # 验证标准目录的有效性
        train_validation = validate_yolo_dataset(standard_train_images, standard_train_labels)
        val_validation = validate_yolo_dataset(standard_val_images, standard_val_labels)
        
        if train_validation["success"] and val_validation["success"]:
            results["train_images"] = standard_train_images
            results["train_labels"] = standard_train_labels
            results["val_images"] = standard_val_images
            results["val_labels"] = standard_val_labels
            results["success"] = True
            results["message"] = "Найдена стандартная структура каталогов YOLO, проверка пройдена"
            return results
    
    # 查找可能的图像目录
    image_dirs = []
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name in ['images', 'img', 'train', 'val']:
                path = os.path.join(root, dir_name)
                # 检查是否包含图片
                if has_image_files(path):
                    image_dirs.append(path)
    
    # 对于找到的每个图像目录，尝试找到匹配的标签目录
    for img_dir in image_dirs:
        # 跳过已经处理过的目录
        if results["train_images"] and img_dir == results["train_images"]:
            continue
        if results["val_images"] and img_dir == results["val_images"]:
            continue
        
        # 判断是训练集还是验证集
        dir_name = os.path.basename(img_dir)
        parent_name = os.path.basename(os.path.dirname(img_dir))
        
        # 判断是否为训练目录
        is_train = ('train' in dir_name.lower() or 
                    (parent_name == 'images' and dir_name == 'train'))
        
        # 判断是否为验证目录
        is_val = ('val' in dir_name.lower() or 
                  'valid' in dir_name.lower() or 
                  'test' in dir_name.lower() or
                  (parent_name == 'images' and dir_name == 'val'))
        
        # 如果不能确定是训练还是验证，暂时跳过
        if not is_train and not is_val:
            continue
        
        # 尝试验证此目录
        validation = validate_yolo_dataset(img_dir)
        
        if validation["success"]:
            if is_train and not results["train_images"]:
                results["train_images"] = img_dir
                results["train_labels"] = validation.get("found_labels_dir")
                results["message"] += f"Найден каталог обучающих изображений: {img_dir}\n"
            elif is_val and not results["val_images"]:
                results["val_images"] = img_dir
                results["val_labels"] = validation.get("found_labels_dir")
                results["message"] += f"Найден каталог валидационных изображений: {img_dir}\n"
    
    # 如果只找到训练集，但没有验证集，使用部分训练集作为验证集
    if results["train_images"] and not results["val_images"]:
        results["val_images"] = results["train_images"]
        results["val_labels"] = results["train_labels"]
        results["message"] += "Отдельный валидационный набор не найден, обучающий набор будет использован как валидационный\n"
    
    # 确定是否找到足够的数据
    if results["train_images"]:
        results["success"] = True
        results["message"] += "Набор данных успешно найден и проверен\n"
    else:
        results["message"] += "Не удалось найти корректный обучающий набор данных\n"
    
    return results


def has_image_files(directory: str) -> bool:
    """
    检查目录是否包含图片文件
    
    Args:
        directory: 要检查的目录
        
    Returns:
        如果包含图片文件返回True，否则返回False
    """
    if not os.path.exists(directory):
        return False
        
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    for ext in image_extensions:
        # 检查大小写扩展名
        if glob.glob(os.path.join(directory, f"*{ext}")) or glob.glob(os.path.join(directory, f"*{ext.upper()}")):
            return True
    
    return False


def inspect_dataset_structure(base_dir: str) -> str:
    """
    检查并报告数据集的目录结构
    
    Args:
        base_dir: 要检查的基础目录
        
    Returns:
        描述数据集结构的文本报告
    """
    report = []
    report.append(f"Проверяемый каталог: {base_dir}")
    
    if not os.path.exists(base_dir):
        return f"Каталог не существует: {base_dir}"
    
    # 统计文件和目录
    dir_counts = {}
    file_extensions = {}
    
    # 最多扫描3层
    max_depth = 3
    current_depth = 0
    
    def scan_directory(directory, depth):
        if depth > max_depth:
            return
            
        try:
            items = os.listdir(directory)
            
            # 分类文件和目录
            dirs = [item for item in items if os.path.isdir(os.path.join(directory, item))]
            files = [item for item in items if os.path.isfile(os.path.join(directory, item))]
            
            # 记录此级别目录的数量
            depth_key = f"Уровень {depth}"
            if depth_key not in dir_counts:
                dir_counts[depth_key] = []
            dir_counts[depth_key].append((directory, len(dirs), len(files)))
            
            # 记录文件扩展名
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in file_extensions:
                    file_extensions[ext] = 0
                file_extensions[ext] += 1
            
            # 递归扫描子目录
            for dir_name in dirs:
                scan_directory(os.path.join(directory, dir_name), depth + 1)
                
        except Exception as e:
            report.append(f"Ошибка при сканировании каталога {directory}: {str(e)}")
    
    # 开始扫描
    scan_directory(base_dir, current_depth)
    
    # 生成报告
    report.append("\nСтруктура каталогов:")
    for depth, dirs in sorted(dir_counts.items()):
        report.append(f"\n{depth}:")
        for dir_info in dirs:
            dir_path, subdirs, files = dir_info
            rel_path = os.path.relpath(dir_path, base_dir)
            if rel_path == '.':
                rel_path = 'Корневой каталог'
            report.append(f"  {rel_path}: {subdirs} подкаталог(ов), {files} файл(ов)")
    
    # 报告文件类型
    report.append("\nСтатистика по типам файлов:")
    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
        report.append(f"  {ext}: {count} файл(ов)")
    
    # 检查是否有常见数据集结构
    common_structures = [
        (os.path.join(base_dir, 'images', 'train'), "Обучающие изображения YOLO"),
        (os.path.join(base_dir, 'labels', 'train'), "Обучающие метки YOLO"),
        (os.path.join(base_dir, 'images', 'val'), "Валидационные изображения YOLO"),
        (os.path.join(base_dir, 'labels', 'val'), "Валидационные метки YOLO"),
        (os.path.join(base_dir, 'train', 'images'), "Альтернативные обучающие изображения"),
        (os.path.join(base_dir, 'val', 'images'), "Альтернативные валидационные изображения"),
        (os.path.join(base_dir, 'annotations'), "Аннотации COCO"),
        (os.path.join(base_dir, 'Annotations'), "Аннотации VOC"),
        (os.path.join(base_dir, 'JPEGImages'), "Изображения VOC"),
        (os.path.join(base_dir, 'ImageSets', 'Main'), "Разделение наборов VOC")
    ]
    
    report.append("\nОбнаруженные структуры набора данных:")
    for path, desc in common_structures:
        if os.path.exists(path):
            # 计算文件数
            file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
            report.append(f"  Найдено {desc}: {path} ({file_count} файл(ов))")
    
    return "\n".join(report)