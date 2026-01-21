#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dolphin和MinerU VLM布局识别结果比对与融合脚本

功能：
1. 以Dolphin为主进行布局识别（Dolphin的检测结果作为主要结果）
2. 使用MinerU VLM对Dolphin的结果进行检查和修正
3. 生成最终的布局识别结果
4. 输出布局可视化PDF文档

注意：本脚本仅进行布局分析，不进行后续内容提取操作
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import cv2
import numpy as np

# 设置环境变量，消除 tokenizers 并行处理的警告
# 这个警告在多进程环境中会出现，不影响功能，但可以通过设置环境变量来消除
if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 默认模型存储路径（可以通过环境变量覆盖）
DEFAULT_MODEL_ROOT = os.getenv("MINERU_MODEL_ROOT", os.path.expanduser("~/mineru_models"))
DEFAULT_MINERU_VLM_PATH = os.path.join(DEFAULT_MODEL_ROOT, "vlm")
DEFAULT_DOLPHIN_PATH = os.getenv("DOLPHIN_MODEL_PATH", "/home/hsr/Dolphin/hf_model")

# 添加Dolphin路径
dolphin_path = os.getenv("DOLPHIN_PATH", "/home/hsr/Dolphin")
if os.path.exists(dolphin_path) and dolphin_path not in sys.path:
    sys.path.insert(0, dolphin_path)

try:
    from demo_layout import DOLPHIN
    from utils.utils import parse_layout_string, process_coordinates, convert_pdf_to_images
    DOLPHIN_AVAILABLE = True
except ImportError as e:
    print(f"警告: Dolphin模块导入失败: {e}")
    print("提示: 请确保已安装pymupdf: pip install pymupdf")
    DOLPHIN_AVAILABLE = False

try:
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock
    from mineru.utils.pdf_image_tools import load_images_from_pdf
    from mineru.utils.enum_class import ImageType
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    MINERU_AVAILABLE = True
except ImportError as e:
    print(f"错误: MinerU模块导入失败: {e}")
    MINERU_AVAILABLE = False
    sys.exit(1)

from loguru import logger


# 模型单例类，用于复用MinerU和Dolphin模型
class ModelSingleton:
    _mineru_client = None
    _mineru_backend = None
    _mineru_model_path = None
    _dolphin_model = None
    _dolphin_model_path = None
    _dolphin_load_failed = False  # 标记Dolphin模型是否加载失败，避免重复尝试
    
    @classmethod
    def get_mineru_client(cls, backend: str, model_path: str = None):
        """获取MinerU客户端（单例）"""
        # 处理model_path：如果是None且不是http-client，使用默认路径或自动下载
        actual_model_path = model_path
        if actual_model_path is None and backend != "http-client":
            # 优先使用默认路径，如果不存在则自动下载
            if os.path.exists(DEFAULT_MINERU_VLM_PATH):
                actual_model_path = DEFAULT_MINERU_VLM_PATH
                logger.info(f"使用本地MinerU VLM模型路径: {actual_model_path}")
            else:
                # 检查 HuggingFace 缓存目录（auto_download_and_get_model_root_path 会优先使用缓存）
                logger.info("未找到本地MinerU VLM模型，检查 HuggingFace 缓存或自动下载...")
                os.makedirs(DEFAULT_MODEL_ROOT, exist_ok=True)
                # auto_download_and_get_model_root_path 会：
                # 1. 先检查 HuggingFace 缓存（~/.cache/huggingface/hub/）
                # 2. 如果缓存中有模型，直接返回缓存路径（不会重新下载）
                # 3. 如果缓存中没有，才会下载到缓存目录
                downloaded_path = auto_download_and_get_model_root_path(DEFAULT_MODEL_ROOT, "vlm")
                actual_model_path = downloaded_path
                logger.info(f"MinerU VLM模型路径: {actual_model_path}")
                # 检查是否是 HuggingFace 缓存路径
                if '/.cache/huggingface/' in actual_model_path or '/huggingface/hub/' in actual_model_path:
                    logger.info(f"提示：模型存储在 HuggingFace 缓存目录中，这是正常的")
                    logger.info(f"提示：模型不会自动删除，后续运行将直接使用缓存，无需重新下载")
                else:
                    logger.info(f"提示：后续运行将直接使用此路径，无需重新下载")
        
        # 如果模型已加载且参数相同，直接返回（使用actual_model_path进行比较）
        if (cls._mineru_client is not None and 
            cls._mineru_backend == backend and 
            cls._mineru_model_path == actual_model_path):
            logger.debug("复用已加载的MinerU VLM模型")
            return cls._mineru_client
        
        # 如果参数不同或未加载，重新加载
        if cls._mineru_client is None:
            logger.info(f"正在加载MinerU VLM模型 (backend: {backend}, path: {actual_model_path})...")
        else:
            # 模型已加载但参数不同，需要重新加载（这种情况应该很少发生）
            logger.warning(
                f"模型参数已变化，重新加载MinerU VLM模型 "
                f"(旧: backend={cls._mineru_backend}, path={cls._mineru_model_path}; "
                f"新: backend={backend}, path={actual_model_path})..."
            )
        cls._mineru_client = MinerUClient(
            backend=backend,
            model_path=actual_model_path
        )
        cls._mineru_backend = backend
        cls._mineru_model_path = actual_model_path  # 保存实际路径，而不是None
        logger.info("MinerU VLM模型加载完成")
        return cls._mineru_client
    
    @classmethod
    def get_dolphin_model(cls, model_path: str):
        """获取Dolphin模型（单例）"""
        if not DOLPHIN_AVAILABLE:
            return None
        
        # 如果之前加载失败，直接返回None，避免重复尝试
        if cls._dolphin_load_failed:
            return None
        
        # 如果模型已加载且路径相同，直接返回
        if cls._dolphin_model is not None and cls._dolphin_model_path == model_path:
            logger.debug("复用已加载的Dolphin模型")
            return cls._dolphin_model
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"Dolphin模型路径不存在: {model_path}")
            logger.warning("Dolphin模型路径不存在，将仅使用MinerU VLM结果")
            cls._dolphin_load_failed = True
            return None
        
        # 如果路径不同或未加载，重新加载
        logger.info(f"正在加载Dolphin模型 (path: {model_path})...")
        try:
            cls._dolphin_model = DOLPHIN(model_path)
            cls._dolphin_model_path = model_path
            cls._dolphin_load_failed = False  # 加载成功，重置失败标志
            logger.info("Dolphin模型加载完成")
            return cls._dolphin_model
        except Exception as e:
            logger.error(f"Dolphin模型加载失败: {e}")
            logger.warning("Dolphin模型加载失败，将仅使用MinerU VLM结果")
            logger.info("后续处理将跳过Dolphin模型，不再重复尝试加载")
            cls._dolphin_model = None
            cls._dolphin_model_path = None
            cls._dolphin_load_failed = True  # 标记加载失败，避免后续重复尝试
            return None


# Dolphin标签到MinerU ContentBlock类型的映射
DOLPHIN_TO_MINERU_TYPE_MAP = {
    'para': 'text',
    'title': 'title',
    'table': 'table',
    'figure': 'image',
    'formula': 'equation',
    'code': 'code',
    'list': 'list',
    'header': 'header',
    'footer': 'footer',
    'page_number': 'page_number',
    'fnote': 'page_footnote',
    'distorted_page': 'text',
    'sec_0': 'title',
    'sec_1': 'title',
    'author': 'text',
    'paper_abstract': 'text',
    'watermark': 'aside_text',
    'meta_num': 'text',
}


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个bbox的IoU"""
    x1_min = max(bbox1[0], bbox2[0])
    y1_min = max(bbox1[1], bbox2[1])
    x1_max = min(bbox1[2], bbox2[2])
    y1_max = min(bbox1[3], bbox2[3])
    
    if x1_max <= x1_min or y1_max <= y1_min:
        return 0.0
    
    intersection = (x1_max - x1_min) * (y1_max - y1_min)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
    
    return intersection / union


def calculate_bbox_area(bbox: List[float]) -> float:
    """计算bbox的面积"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
    """将绝对坐标的bbox归一化到[0,1]范围"""
    x1, y1, x2, y2 = bbox
    x1_norm = max(0.0, min(1.0, x1 / img_width))
    y1_norm = max(0.0, min(1.0, y1 / img_height))
    x2_norm = max(0.0, min(1.0, x2 / img_width))
    y2_norm = max(0.0, min(1.0, y2 / img_height))
    return [x1_norm, y1_norm, x2_norm, y2_norm]


def convert_dolphin_to_content_blocks(
    dolphin_results: List[Tuple[List[float], str, List[str]]],
    image: Image.Image
) -> List[ContentBlock]:
    """将Dolphin的结果转换为ContentBlock列表"""
    img_width, img_height = image.size
    blocks = []
    
    for coords, label, tags in dolphin_results:
        # 将Dolphin的坐标转换为原图坐标再归一化
        if DOLPHIN_AVAILABLE:
            try:
                x1, y1, x2, y2 = process_coordinates(coords, image)
            except Exception:
                continue
        else:
            x1, y1, x2, y2 = coords
        
        normalized_bbox = normalize_bbox([x1, y1, x2, y2], img_width, img_height)
        mineru_type = DOLPHIN_TO_MINERU_TYPE_MAP.get(label.lower(), 'text')
        
        block = ContentBlock(
            type=mineru_type,
            bbox=normalized_bbox,
            angle=None,
            content=None
        )
        blocks.append(block)
    
    return blocks


def is_bbox_covered(
    bbox: List[float],
    existing_blocks: List[ContentBlock],
    coverage_threshold: float = 0.5
) -> bool:
    """
    检查bbox是否被已存在的blocks覆盖
    
    参数:
        bbox: 待检查的bbox
        existing_blocks: 已存在的blocks列表
        coverage_threshold: 覆盖阈值，如果IoU超过此值则认为被覆盖
    
    返回:
        如果被覆盖返回True，否则返回False
    """
    for existing_block in existing_blocks:
        iou = calculate_iou(bbox, existing_block.bbox)
        if iou > coverage_threshold:
            return True
    return False


def compare_and_merge_layouts(
    dolphin_blocks: List[ContentBlock],
    mineru_blocks: List[ContentBlock],
    mineru_client: MinerUClient = None,
    image: Image.Image = None,
    iou_threshold: float = 0.7,
    type_match_threshold: float = 0.5,
    coverage_threshold: float = 0.5
) -> List[ContentBlock]:
    """
    比对和融合Dolphin和MinerU VLM的布局识别结果（以Dolphin为主，MinerU VLM仅用于检查修正）
    
    策略（以Dolphin为主，MinerU VLM检查修正）：
    1. 以Dolphin的结果作为主要结果（所有Dolphin block都会保留）
    2. 对于每个Dolphin block，使用MinerU VLM进行检查和修正：
       - 如果找到匹配的MinerU block（IoU > threshold）：
         * 分类检查修正：
           - 如果Dolphin识别为table：分类必须保持table（以Dolphin为准，不修正）
           - 如果分类一致：使用Dolphin的分类（保持不变）
           - 如果分类不一致且Dolphin不是table：
             * table vs text冲突：修正为table（避免表格被误识别为文本）
             * 其他情况：保持Dolphin的分类（以Dolphin为主，不修正）
         * bbox检查修正：
           - 如果Dolphin识别为table：由MinerU检测是否需要扩大覆盖范围（如果MinerU更大则扩大，否则保持Dolphin）
           - 如果不是表格，则以小的bbox为准（避免包含多余区域）
       - 如果未找到匹配：直接使用Dolphin的block（不做修正）
    3. 不添加MinerU独有的block（因为Dolphin是主检测方法，MinerU只用于修正）
    """
    merged_blocks = []
    used_mineru_indices = set()
    
    # 以Dolphin为主：遍历所有Dolphin block，使用MinerU VLM进行检查修正
    for i, dolphin_block in enumerate(dolphin_blocks):
        best_match_idx = -1
        best_iou = 0.0
        
        # 寻找IoU最高的MinerU block进行检查
        for j, mineru_block in enumerate(mineru_blocks):
            if j in used_mineru_indices:
                continue
            
            iou = calculate_iou(dolphin_block.bbox, mineru_block.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
        
        # 如果找到匹配的MinerU block（IoU超过阈值），进行检查修正
        if best_match_idx >= 0 and best_iou > iou_threshold:
            mineru_block = mineru_blocks[best_match_idx]
            
            # 分类检查修正策略（以Dolphin为主）：
            # 1. 如果Dolphin识别为table或header，分类必须保持（以Dolphin为准，不修正）
            #    - MinerU的header分类有问题，必须用Dolphin的header分类覆盖
            # 2. 如果两个分类一致，使用Dolphin的分类（保持不变）
            # 3. 如果分类不一致且Dolphin不是table/header：
            #    - table vs text冲突：如果MinerU识别为table，修正为table（避免表格被误识别为文本）
            #    - header冲突：如果MinerU识别为header但Dolphin不是，保持Dolphin的分类（MinerU的header分类有问题）
            #    - 其他情况：保持Dolphin的分类（以Dolphin为主，不修正）
            if dolphin_block.type in ['table', 'header']:
                # Dolphin识别为table或header，分类必须保持（以Dolphin为准）
                # 特别是header：MinerU的header分类有问题，必须用Dolphin的header分类覆盖
                selected_type = dolphin_block.type
                if dolphin_block.type == 'header' and mineru_block.type != 'header':
                    logger.debug(f"保持header分类: MinerU分类={mineru_block.type}，使用Dolphin的header分类（MinerU的header分类有问题）")
            elif dolphin_block.type == mineru_block.type:
                selected_type = dolphin_block.type  # 分类一致，保持Dolphin的分类
            else:
                # 分类不一致的情况（Dolphin不是table/header）
                types_set = {dolphin_block.type, mineru_block.type}
                if 'table' in types_set and 'text' in types_set:
                    # table和text冲突时，如果MinerU识别为table，修正为table（避免表格被误识别为文本）
                    if mineru_block.type == 'table':
                        selected_type = 'table'
                        logger.debug(f"修正分类: Dolphin={dolphin_block.type} -> MinerU=table -> 修正为table")
                    else:
                        selected_type = dolphin_block.type
                elif 'header' in types_set:
                    # header冲突：如果MinerU识别为header但Dolphin不是，保持Dolphin的分类（MinerU的header分类有问题）
                    if mineru_block.type == 'header':
                        # MinerU识别为header，但Dolphin不是，保持Dolphin的分类（MinerU的header分类有问题）
                        selected_type = dolphin_block.type
                        logger.debug(f"保持Dolphin分类: MinerU识别为header但Dolphin不是，保持Dolphin的分类={dolphin_block.type}（MinerU的header分类有问题）")
                    else:
                        selected_type = dolphin_block.type
                else:
                    # 其他情况保持Dolphin的分类（以Dolphin为主，不修正）
                    selected_type = dolphin_block.type
            
            # bbox检查修正策略（改进版）：
            # - 表格：更激进的扩大策略，检查所有MinerU的table block，选择最大的bbox（确保完整覆盖）
            # - header：如果Dolphin识别为header，保持header分类，但用MinerU的bbox修正定位（Dolphin定位可能有问题）
            # - 其他：如果Dolphin的定位覆盖有问题，使用MinerU的bbox进行修正（选择更合理的bbox）
            dolphin_area = calculate_bbox_area(dolphin_block.bbox)
            mineru_area = calculate_bbox_area(mineru_block.bbox)
            
            if selected_type == 'table':
                # 表格：更激进的扩大策略
                # 1. 先检查匹配的MinerU block
                # 2. 再检查所有MinerU的table block，看是否有更大的覆盖范围
                best_table_bbox = dolphin_block.bbox
                best_table_area = dolphin_area
                
                # 检查匹配的MinerU block
                if mineru_area > dolphin_area:
                    best_table_bbox = mineru_block.bbox
                    best_table_area = mineru_area
                
                # 检查所有MinerU的table block，看是否有更大的覆盖范围
                for other_mineru_block in mineru_blocks:
                    if other_mineru_block.type == 'table':
                        other_area = calculate_bbox_area(other_mineru_block.bbox)
                        # 检查是否有重叠（IoU > 0.3），如果有重叠且更大，考虑使用
                        iou_with_other = calculate_iou(dolphin_block.bbox, other_mineru_block.bbox)
                        if iou_with_other > 0.3 and other_area > best_table_area:
                            best_table_bbox = other_mineru_block.bbox
                            best_table_area = other_area
                            logger.debug(f"表格bbox扩大: 发现更大的MinerU table bbox (IoU={iou_with_other:.2f}, 面积={other_area:.4f})")
                
                selected_bbox = best_table_bbox
                if best_table_area > dolphin_area:
                    logger.debug(f"表格bbox扩大: 使用更大的bbox (Dolphin面积={dolphin_area:.4f}, 最终面积={best_table_area:.4f})")
                else:
                    logger.debug(f"表格bbox保持: 使用Dolphin的bbox (面积={dolphin_area:.4f})")
            elif selected_type == 'header':
                # header：如果Dolphin识别为header，保持header分类，但用MinerU的bbox修正定位
                # Dolphin的header分类更准确，但定位覆盖可能有问题，用MinerU修正
                if mineru_area > dolphin_area * 0.8 and mineru_area < dolphin_area * 1.5:
                    # MinerU的bbox在合理范围内，使用MinerU的bbox修正定位
                    selected_bbox = mineru_block.bbox
                    logger.debug(f"header bbox修正: 使用MinerU的bbox修正Dolphin的定位 (Dolphin面积={dolphin_area:.4f}, MinerU面积={mineru_area:.4f})")
                else:
                    # MinerU的bbox不合理，使用Dolphin的bbox
                    selected_bbox = dolphin_block.bbox
                    logger.debug(f"header bbox保持: MinerU的bbox不合理 (Dolphin面积={dolphin_area:.4f}, MinerU面积={mineru_area:.4f})")
            else:
                # 其他类型：如果Dolphin的定位覆盖有问题，使用MinerU的bbox进行修正
                # 选择更合理的bbox（面积适中，不是太大也不是太小）
                area_ratio = mineru_area / dolphin_area if dolphin_area > 0 else 1.0
                if 0.7 <= area_ratio <= 1.3:
                    # MinerU的bbox比较合理（在Dolphin的70%-130%范围内），使用MinerU的bbox
                    selected_bbox = mineru_block.bbox
                    logger.debug(f"非表格bbox修正: 使用MinerU的bbox修正Dolphin的定位覆盖问题 (面积比={area_ratio:.2f})")
                else:
                    # MinerU的bbox不合理，使用Dolphin的bbox
                    selected_bbox = dolphin_block.bbox
            
            # 创建修正后的block（以Dolphin为主，但可能被MinerU修正）
            merged_block = ContentBlock(
                type=selected_type,  # 分类：Dolphin识别为table时保持table，其他情况可能被MinerU修正
                bbox=selected_bbox,   # bbox：Dolphin识别为table时由MinerU检测是否需要扩大，非表格以小的为准
                angle=dolphin_block.angle,  # 保留Dolphin的其他属性
                content=dolphin_block.content
            )
            merged_blocks.append(merged_block)
            used_mineru_indices.add(best_match_idx)
        else:
            # 没有找到匹配的MinerU block，直接使用Dolphin的block（不做修正）
            merged_blocks.append(dolphin_block)
    
    # 注意：不添加MinerU独有的block，因为Dolphin是主检测方法，MinerU只用于修正
    
    return merged_blocks


def visualize_layout_pdf(
    images: List[Image.Image],
    blocks_list: List[List[ContentBlock]],
    output_path: str,
    alpha: float = 0.3
) -> None:
    """生成布局可视化PDF"""
    
    # 块类型颜色映射（BGR格式）
    type_colors = {
        'text': (200, 255, 200),
        'title': (200, 200, 255),
        'table': (255, 200, 200),
        'image': (255, 255, 200),
        'code': (255, 200, 255),
        'equation': (200, 255, 255),
        'header': (220, 220, 220),
        'footer': (220, 220, 220),
        'page_number': (180, 180, 180),
        'page_footnote': (240, 240, 200),
    }
    default_color = (200, 200, 200)
    
    visualized_images = []
    
    for image, blocks in zip(images, blocks_list):
        img_array = np.array(image.convert('RGB'))
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_width, img_height = image.size
        
        overlay = img_bgr.copy()
        
        # 绘制每个block
        for idx, block in enumerate(blocks):
            bbox_norm = block.bbox
            x1 = int(bbox_norm[0] * img_width)
            y1 = int(bbox_norm[1] * img_height)
            x2 = int(bbox_norm[2] * img_width)
            y2 = int(bbox_norm[3] * img_height)
            
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(x1 + 1, min(x2, img_width))
            y2 = max(y1 + 1, min(y2, img_height))
            
            color = type_colors.get(block.type.lower(), default_color)
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            label_text = f"{idx + 1}: {block.type}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, font, font_scale, thickness
            )
            
            text_x = x1
            text_y = y1 - 5
            if text_y - text_height < 0:
                text_y = y1 + text_height + 5
            
            cv2.rectangle(
                img_bgr,
                (text_x - 2, text_y - text_height - 2),
                (text_x + text_width + 2, text_y + baseline + 2),
                (255, 255, 255),
                -1
            )
            
            cv2.putText(
                img_bgr,
                label_text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )
        
        result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        pil_result = Image.fromarray(result_rgb)
        visualized_images.append(pil_result)
    
    # 保存为PDF
    if visualized_images:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        visualized_images[0].save(
            output_path,
            "PDF",
            resolution=100.0,
            save_all=True,
            append_images=visualized_images[1:] if len(visualized_images) > 1 else []
        )
        logger.info(f"布局可视化PDF已保存到: {output_path}")


def process_document(
    input_path: str,
    output_dir: str,
    dolphin_model_path: str,
    mineru_backend: str = "transformers",
    mineru_model_path: str = None,
    iou_threshold: float = 0.7,
    type_match_threshold: float = 0.5,
    skip_dolphin: bool = False,
    skip_mineru: bool = False
) -> None:
    """
    处理单个文档（以Dolphin为主，MinerU VLM检查修正）
    
    注意：本函数仅进行布局分析，不进行后续内容提取操作
    """
    
    # 加载PDF并转换为图片
    # 注意：为了与Dolphin demo保持一致，优先使用Dolphin的PDF转换方式
    # Dolphin的convert_pdf_to_images会将最长边缩放到896像素，这会影响识别结果
    # 使用相同的PDF转换方式可以确保结果一致
    images_pil_list = []
    if DOLPHIN_AVAILABLE and not skip_dolphin:
        try:
            logger.info("使用Dolphin的PDF转换方式（与Dolphin demo保持一致，target_size=896）...")
            images_pil_list = convert_pdf_to_images(input_path, target_size=896)
            if not images_pil_list:
                raise Exception("Dolphin PDF转换失败")
            logger.info(f"Dolphin PDF转换完成，共 {len(images_pil_list)} 页")
        except Exception as e:
            logger.warning(f"Dolphin PDF转换失败: {e}，使用MinerU的PDF转换方式...")
            with open(input_path, 'rb') as f:
                pdf_bytes = f.read()
            images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
            images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    else:
        # 使用MinerU的PDF转换方式
        logger.info("使用MinerU的PDF转换方式...")
        with open(input_path, 'rb') as f:
            pdf_bytes = f.read()
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    
    output_file_name = Path(input_path).stem
    logger.info(f"正在处理文档: {input_path}，共 {len(images_pil_list)} 页")
    
    # 1. 使用Dolphin进行布局识别（主要方法，使用单例，避免重复加载）
    dolphin_results_list = []
    if skip_dolphin:
        logger.warning("跳过Dolphin布局识别（skip_dolphin=True）")
        if not skip_mineru:
            logger.warning("警告：Dolphin被跳过，将仅使用MinerU VLM结果（非推荐模式）")
        dolphin_results_list = [[] for _ in images_pil_list]
    elif DOLPHIN_AVAILABLE:
        logger.info("正在使用Dolphin进行布局识别（主要方法）...")
        # 尝试获取已加载的Dolphin模型（如果之前加载失败，这里会返回None且不会重复尝试）
        dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
        if dolphin_model is None:
            logger.error("Dolphin模型不可用，无法继续处理（Dolphin是主要方法）")
            if not skip_mineru:
                logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
            dolphin_results_list = [[] for _ in images_pil_list]
        else:
            try:
                for idx, image in enumerate(images_pil_list):
                    logger.info(f"  Dolphin识别页面 {idx + 1}/{len(images_pil_list)}...")
                    layout_str = dolphin_model.chat("Parse the reading order of this document.", image)
                    dolphin_results = parse_layout_string(layout_str)
                    
                    # 检查结果是否有效（与Dolphin demo保持一致）
                    if not dolphin_results or not (layout_str.startswith("[") and layout_str.endswith("]")):
                        logger.warning(f"  页面 {idx + 1} Dolphin识别结果无效，使用默认distorted_page")
                        # 使用默认的distorted_page（与Dolphin demo保持一致）
                        img_width, img_height = image.size
                        dolphin_results = [([0, 0, img_width, img_height], 'distorted_page', [])]
                    
                    dolphin_results_list.append(dolphin_results)
                logger.info(f"Dolphin识别完成，共识别 {sum(len(r) for r in dolphin_results_list)} 个blocks")
            except Exception as e:
                logger.error(f"Dolphin布局识别失败: {e}")
                if not skip_mineru:
                    logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
                dolphin_results_list = [[] for _ in images_pil_list]
    else:
        logger.error("Dolphin不可用，无法继续处理（Dolphin是主要方法）")
        if not skip_mineru:
            logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
        dolphin_results_list = [[] for _ in images_pil_list]
    
    # 2. 使用MinerU VLM进行布局识别（辅助方法，用于验证和补充）
    mineru_results = []
    mineru_client = None
    if skip_mineru:
        logger.info("跳过MinerU VLM布局识别（skip_mineru=True）")
        mineru_results = [[] for _ in images_pil_list]
    else:
        logger.info("正在使用MinerU VLM进行布局识别（辅助方法）...")
        try:
            mineru_client = ModelSingleton.get_mineru_client(mineru_backend, mineru_model_path)
            mineru_results = mineru_client.batch_layout_detect(images=images_pil_list)
            logger.info(f"MinerU VLM识别完成，共识别 {sum(len(r) for r in mineru_results)} 个blocks")
        except Exception as e:
            logger.warning(f"MinerU VLM布局识别失败: {e}")
            logger.warning("将仅使用Dolphin结果（MinerU VLM作为辅助方法，失败不影响主流程）")
            import traceback
            traceback.print_exc()
            mineru_results = [[] for _ in images_pil_list]
    
    # 检查是否有可用的结果
    if not dolphin_results_list or all(len(r) == 0 for r in dolphin_results_list):
        if skip_mineru or not mineru_results or all(len(r) == 0 for r in mineru_results):
            logger.error("=" * 80)
            logger.error("错误：Dolphin和MinerU VLM都不可用，无法进行布局识别")
            logger.error("=" * 80)
            logger.error("请确保：")
            logger.error("1. Dolphin模型路径正确且模型已加载")
            logger.error("2. 如果使用MinerU VLM，确保模型已正确加载")
            logger.error("3. 不要同时使用 --skip-dolphin 和 --skip-mineru")
            logger.error("=" * 80)
            raise RuntimeError("Dolphin和MinerU VLM都不可用，无法进行布局识别")
    
    # 3. 比对和融合结果（以Dolphin为主，MinerU VLM检查修正）
    if skip_mineru:
        logger.info("正在处理布局识别结果（仅使用Dolphin，跳过MinerU VLM）...")
    else:
        logger.info("正在比对和融合布局识别结果（以Dolphin为主，MinerU VLM检查修正）...")
    merged_results = []
    
    for page_idx, (dolphin_results, mineru_blocks, image_pil) in enumerate(
        zip(dolphin_results_list, mineru_results, images_pil_list)
    ):
        # 转换Dolphin结果为ContentBlock
        dolphin_blocks = convert_dolphin_to_content_blocks(dolphin_results, image_pil)
        
        # 比对和融合（以Dolphin为主，MinerU VLM检查修正）
        merged_blocks = compare_and_merge_layouts(
            dolphin_blocks,  # 以Dolphin为主
            mineru_blocks,   # MinerU VLM检查修正
            mineru_client=mineru_client,
            image=image_pil,
            iou_threshold=iou_threshold,
            type_match_threshold=type_match_threshold
        )
        
        merged_results.append(merged_blocks)
        logger.info(
            f"页面 {page_idx + 1}: Dolphin {len(dolphin_blocks)}个blocks, "
            f"MinerU VLM {len(mineru_blocks)}个blocks, "
            f"融合后 {len(merged_blocks)}个blocks"
        )
    
    # 4. 生成可视化PDF
    output_pdf_path = os.path.join(output_dir, f"{output_file_name}_layout.pdf")
    
    logger.info("正在生成布局可视化PDF...")
    visualize_layout_pdf(images_pil_list, merged_results, output_pdf_path)
    
    # 5. 保存结果JSON
    output_json_path = os.path.join(output_dir, f"{output_file_name}_layout.json")
    results_json = []
    for page_idx, blocks in enumerate(merged_results):
        page_results = []
        for block in blocks:
            page_results.append({
                "type": block.type,
                "bbox": block.bbox,
                "angle": block.angle,
                "content": block.content
            })
        results_json.append({
            "page": page_idx + 1,
            "blocks": page_results
        })
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2)
    
    logger.info(f"布局识别结果JSON已保存到: {output_json_path}")
    logger.info(f"处理完成: {input_path}")




def main():
    parser = argparse.ArgumentParser(description="Dolphin和MinerU VLM布局识别结果比对与融合（以Dolphin为主）")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="输入PDF文件路径或包含PDF文件的目录"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--dolphin-model-path",
        default=DEFAULT_DOLPHIN_PATH,
        help=f"Dolphin模型路径（默认: {DEFAULT_DOLPHIN_PATH}）"
    )
    parser.add_argument(
        "--mineru-backend",
        default="transformers",
        choices=["transformers", "vllm-engine", "vllm-async-engine", "lmdeploy-engine", "http-client"],
        help="MinerU VLM后端（默认: transformers，即vlm-transformers）"
    )
    parser.add_argument(
        "--mineru-model-path",
        default=None,
        help="MinerU模型路径（默认: 自动下载）"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.7,
        help="IoU阈值，用于判断block是否一致（默认: 0.7）"
    )
    parser.add_argument(
        "--type-match-threshold",
        type=float,
        default=0.5,
        help="类型匹配阈值，用于处理不一致的block（默认: 0.5）"
    )
    parser.add_argument(
        "--skip-dolphin",
        action="store_true",
        default=False,
        help="跳过Dolphin模型加载（不推荐，Dolphin是主要方法）"
    )
    parser.add_argument(
        "--skip-mineru",
        action="store_true",
        default=False,
        help="跳过MinerU VLM模型加载（MinerU VLM是辅助方法，可以跳过）"
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 收集输入文件
    input_path = Path(args.input)
    if input_path.is_file():
        input_files = [input_path]
    elif input_path.is_dir():
        input_files = list(input_path.glob("*.pdf"))
    else:
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)
    
    if not input_files:
        logger.error(f"未找到PDF文件: {args.input}")
        sys.exit(1)
    
    logger.info(f"找到 {len(input_files)} 个PDF文件")
    
    # 预加载模型（第一次加载，后续文档处理时复用）
    # 注意：先获取实际的model_path，确保后续调用时使用相同的路径
    actual_mineru_model_path = args.mineru_model_path
    if actual_mineru_model_path is None and args.mineru_backend != "http-client":
        # 优先使用默认路径，如果不存在则自动下载
        if os.path.exists(DEFAULT_MINERU_VLM_PATH):
            actual_mineru_model_path = DEFAULT_MINERU_VLM_PATH
            logger.info(f"使用本地MinerU VLM模型路径: {actual_mineru_model_path}")
        else:
            # 检查 HuggingFace 缓存目录（auto_download_and_get_model_root_path 会优先使用缓存）
            logger.info("未找到本地MinerU VLM模型，检查 HuggingFace 缓存或自动下载...")
            os.makedirs(DEFAULT_MODEL_ROOT, exist_ok=True)
            # auto_download_and_get_model_root_path 会：
            # 1. 先检查 HuggingFace 缓存（~/.cache/huggingface/hub/）
            # 2. 如果缓存中有模型，直接返回缓存路径（不会重新下载）
            # 3. 如果缓存中没有，才会下载到缓存目录
            downloaded_path = auto_download_and_get_model_root_path(DEFAULT_MODEL_ROOT, "vlm")
            actual_mineru_model_path = downloaded_path
            logger.info(f"MinerU VLM模型路径: {actual_mineru_model_path}")
            # 检查是否是 HuggingFace 缓存路径
            if '/.cache/huggingface/' in actual_mineru_model_path or '/huggingface/hub/' in actual_mineru_model_path:
                logger.info(f"提示：模型存储在 HuggingFace 缓存目录中，这是正常的")
                logger.info(f"提示：模型不会自动删除，后续运行将直接使用缓存，无需重新下载")
            else:
                logger.info(f"提示：后续运行将直接使用此路径，无需重新下载")
    
    logger.info("正在预加载模型（仅加载一次，后续复用）...")
    
    # 先加载Dolphin模型（主要方法，必需）
    dolphin_model = None
    if args.skip_dolphin:
        logger.warning("跳过Dolphin模型加载（--skip-dolphin选项，不推荐）")
        if not args.skip_mineru:
            logger.warning("警告：Dolphin是主要方法，跳过后将仅使用MinerU VLM（非推荐模式）")
    elif DOLPHIN_AVAILABLE:
        # 检查Dolphin模型路径是否存在
        dolphin_model_path = args.dolphin_model_path
        if not os.path.exists(dolphin_model_path):
            logger.error(f"Dolphin模型路径不存在: {dolphin_model_path}")
            logger.error("Dolphin是主要方法，模型路径不存在将导致处理失败")
            logger.info(f"提示：请确保Dolphin模型已下载到: {dolphin_model_path}")
            logger.info("或者使用 --dolphin-model-path 指定正确的模型路径")
            # 标记为失败，避免后续重复检查
            ModelSingleton._dolphin_load_failed = True
            if not args.skip_mineru:
                logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
        else:
            try:
                dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
                if dolphin_model is not None:
                    logger.info("Dolphin模型预加载完成（主要方法）")
                else:
                    logger.error("Dolphin模型不可用，无法继续处理（Dolphin是主要方法）")
                    if not args.skip_mineru:
                        logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
            except Exception as e:
                logger.error(f"Dolphin模型预加载失败: {e}")
                logger.error("Dolphin是主要方法，加载失败将导致处理失败")
                if not args.skip_mineru:
                    logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
                import traceback
                traceback.print_exc()
    else:
        logger.error("Dolphin不可用，无法继续处理（Dolphin是主要方法）")
        if not args.skip_mineru:
            logger.warning("将尝试仅使用MinerU VLM结果（非推荐模式）")
    
    # 尝试加载MinerU VLM模型（辅助方法，可选）
    if args.skip_mineru:
        logger.info("跳过MinerU VLM模型加载（--skip-mineru选项，MinerU VLM是辅助方法）")
    else:
        try:
            ModelSingleton.get_mineru_client(args.mineru_backend, actual_mineru_model_path)
            logger.info("MinerU VLM模型预加载完成（辅助方法）")
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
                logger.warning("=" * 80)
                logger.warning("GPU显存不足，无法加载MinerU VLM模型（辅助方法）")
                logger.warning("=" * 80)
                logger.warning(f"错误详情: {error_msg}")
                logger.warning("")
                logger.warning("MinerU VLM是辅助方法，加载失败不影响主流程")
                logger.warning("将仅使用Dolphin结果（推荐模式）")
                logger.warning("=" * 80)
                logger.warning("提示：可以使用 --skip-mineru 选项跳过MinerU VLM模型加载")
            else:
                logger.warning(f"MinerU VLM模型预加载失败（辅助方法）: {e}")
                logger.warning("MinerU VLM是辅助方法，加载失败不影响主流程")
                logger.warning("将仅使用Dolphin结果（推荐模式）")
        except Exception as e:
            logger.warning(f"MinerU VLM模型预加载失败（辅助方法）: {e}")
            logger.warning("MinerU VLM是辅助方法，加载失败不影响主流程")
            logger.warning("将仅使用Dolphin结果（推荐模式）")
    
    logger.info("模型预加载完成，开始处理文档...")
    
    # 处理每个文件（复用已加载的模型）
    for input_file in input_files:
        try:
            process_document(
                str(input_file),
                args.output,
                args.dolphin_model_path,
                args.mineru_backend,
                actual_mineru_model_path,  # 使用实际的model_path，确保单例匹配
                args.iou_threshold,
                args.type_match_threshold,
                args.skip_dolphin,
                args.skip_mineru
            )
        except Exception as e:
            logger.error(f"处理文件 {input_file} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("所有文件处理完成")


if __name__ == "__main__":
    main()

