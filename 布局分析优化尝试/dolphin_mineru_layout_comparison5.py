#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dolphin和MinerU VLM布局识别与内容提取脚本

功能：
1. 使用Dolphin进行全页面布局识别（主要方法）
2. 使用MinerU VLM基于Dolphin的布局结果进行内容提取
3. Header重新分类：基于位置、高度、面积、宽高比、周围内容重新分类header blocks
4. 生成最终的布局识别结果和内容提取结果
5. 输出布局可视化PDF文档
6. 处理时间统计：输出processing_time.json

注意：
- 布局分析：使用Dolphin进行全页面布局识别（主要方法）
- 内容提取：使用MinerU VLM基于Dolphin的布局结果进行内容提取，生成MinerU标准格式输出
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime
from PIL import Image
import cv2
import numpy as np

# 设置环境变量，消除 tokenizers 并行处理的警告
# 这个警告在多进程环境中会出现，不影响功能，但可以通过设置环境变量来消除
if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 默认模型存储路径（可以通过环境变量覆盖）
# 如果设置了MINERU_VLM_MODEL_PATH环境变量，直接使用它
DEFAULT_MINERU_VLM_PATH = os.getenv("MINERU_VLM_MODEL_PATH", None)

if DEFAULT_MINERU_VLM_PATH is None:
    # 检查HuggingFace缓存路径（用户可能已经下载了模型）
    hf_cache_path = "/home/hsr/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/snapshots/879e58bdd9566632b27a8a81f0e2961873311f67"
    if os.path.exists(hf_cache_path):
        DEFAULT_MINERU_VLM_PATH = hf_cache_path
    else:
        # 否则使用默认路径（会自动下载）
        DEFAULT_MODEL_ROOT = os.getenv("MINERU_MODEL_ROOT", os.path.expanduser("~/mineru_models"))
        DEFAULT_MINERU_VLM_PATH = os.path.join(DEFAULT_MODEL_ROOT, "vlm")
else:
    # 如果用户通过环境变量设置了路径，也设置DEFAULT_MODEL_ROOT（用于auto_download函数）
    DEFAULT_MODEL_ROOT = os.path.dirname(DEFAULT_MINERU_VLM_PATH) if os.path.dirname(DEFAULT_MINERU_VLM_PATH) else os.path.expanduser("~/mineru_models")
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
    from mineru.utils.enum_class import ImageType, MakeMode
    from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    from mineru.backend.vlm.model_output_to_middle_json import result_to_middle_json
    from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
    from mineru.data.data_reader_writer import FileBasedDataWriter
    MINERU_AVAILABLE = True
except ImportError as e:
    print(f"错误: MinerU模块导入失败: {e}")
    MINERU_AVAILABLE = False
    sys.exit(1)

from loguru import logger


# 模型单例类，用于复用MinerU和Dolphin模型（本地加载模式）
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
                # 使用默认的模型根目录（不是HuggingFace缓存路径）
                model_root_for_download = os.getenv("MINERU_MODEL_ROOT", os.path.expanduser("~/mineru_models"))
                os.makedirs(model_root_for_download, exist_ok=True)
                # auto_download_and_get_model_root_path 会：
                # 1. 先检查 HuggingFace 缓存（~/.cache/huggingface/hub/）
                # 2. 如果缓存中有模型，直接返回缓存路径（不会重新下载）
                # 3. 如果缓存中没有，才会下载到缓存目录
                downloaded_path = auto_download_and_get_model_root_path(model_root_for_download, "vlm")
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
        """获取Dolphin模型（本地加载，单例模式）"""
        if not DOLPHIN_AVAILABLE:
            return None
        
        # 如果之前加载失败，直接返回None，避免重复尝试
        if cls._dolphin_load_failed:
            return None
        
        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            logger.error(f"Dolphin模型路径不存在: {model_path}")
            logger.warning("Dolphin模型路径不存在，将仅使用MinerU VLM结果")
            cls._dolphin_load_failed = True
            return None
        
        # 如果模型已加载且路径相同，直接返回（单例模式，在同一次运行中复用）
        if cls._dolphin_model is not None and cls._dolphin_model_path == model_path:
            logger.debug("复用已加载的Dolphin模型（本地模式，同次运行中复用）")
            return cls._dolphin_model
        
        # 如果路径不同或未加载，重新加载
        logger.info(f"正在加载Dolphin模型到本地内存 (path: {model_path})...")
        try:
            cls._dolphin_model = DOLPHIN(model_path)
            cls._dolphin_model_path = model_path
            cls._dolphin_load_failed = False  # 加载成功，重置失败标志
            logger.info("Dolphin模型加载完成（本地模式）")
            logger.info("提示：模型已加载到内存，本次运行中处理多个文档时会复用该模型")
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
    'tab': 'table',  # Dolphin输出的表格标签是'tab'，需要映射到'table'
    'figure': 'image',
    'fig': 'image',  # Dolphin可能输出'fig'作为图片标签
    'image': 'image',  # 直接是'image'标签
    'img': 'image',  # 可能是'img'标签
    'picture': 'image',  # 可能是'picture'标签
    'formula': 'equation',
    'equ': 'equation',  # Dolphin输出的公式标签可能是'equ'
    'code': 'code',
    'list': 'list',
    'header': 'header',
    'footer': 'footer',
    'foot': 'footer',  # Dolphin输出的页脚标签是'foot'
    'page_number': 'page_number',
    'fnote': 'page_footnote',
    'distorted_page': 'text',
    'sec_0': 'title',
    'sec_1': 'title',
    'sec_2': 'title',  # Dolphin可能输出sec_2, sec_3等
    'sec_3': 'title',
    'sec_4': 'title',
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


def check_previous_page_ends_with_table(
    previous_blocks: List[ContentBlock],
    image: Image.Image,
    footer_threshold: float = 0.85
) -> bool:
    """
    检查上一页（除页脚外，y > 0.85）是否以表格结尾
    
    参数:
        previous_blocks: 上一页的blocks列表
        image: 上一页的图片（用于获取尺寸）
        footer_threshold: 页脚阈值（默认0.85，即y > 0.85的区域被认为是页脚区域）
    
    返回:
        如果上一页（除页脚外）以表格结尾返回True，否则返回False
    """
    if not previous_blocks:
        return False
    
    img_width, img_height = image.size
    
    # 找到除页脚外（y <= 0.85）的所有blocks
    non_footer_blocks = []
    for block in previous_blocks:
        x1, y1, x2, y2 = block.bbox
        # 转换为像素坐标
        y1_px = y1 * img_height
        y2_px = y2 * img_height
        
        # 如果block的底部（y2）在页脚区域（y > 0.85）之外，则认为是非页脚block
        if y2_px <= footer_threshold * img_height:
            non_footer_blocks.append(block)
    
    if not non_footer_blocks:
        return False
    
    # 找到y坐标最大的block（最下方的block）
    bottom_block = max(non_footer_blocks, key=lambda b: b.bbox[3])
    
    # 检查最下方的block是否是表格
    return bottom_block.type == 'table'


def reclassify_header_blocks(
    blocks: List[ContentBlock],
    image: Image.Image,
    position_threshold: float = 0.15,
    height_threshold: float = 0.1,
    area_threshold: float = 0.1,
    aspect_ratio_threshold: float = 3.0,
    nearby_content_threshold: float = 0.2
) -> List[ContentBlock]:
    """
    重新分类header blocks
    
    判断标准：
    - 位置：y < 0.15（页面顶部 15%）
    - 高度：< 页面高度的 10%
    - 面积：> 页面面积的 10%
    - 宽高比：> 3.0
    - 周围内容：检查下方 20% 区域是否有其他 blocks
    
    分类逻辑：
    - 面积大且高度不小 → `title`
    - 宽且在上方，周围有内容 → `title`
    - 不在顶部 → `text`
    - 小且在上方 → 保持 `header`
    
    参数:
        blocks: ContentBlock列表
        image: PIL图片
        position_threshold: 位置阈值（默认0.15，即页面顶部15%）
        height_threshold: 高度阈值（默认0.1，即页面高度的10%）
        area_threshold: 面积阈值（默认0.1，即页面面积的10%）
        aspect_ratio_threshold: 宽高比阈值（默认3.0）
        nearby_content_threshold: 周围内容检查阈值（默认0.2，即下方20%区域）
    
    返回:
        重新分类后的blocks列表
    """
    img_width, img_height = image.size
    page_area = img_width * img_height
    
    reclassified_blocks = []
    header_corrected_count = 0
    
    for block in blocks:
        if block.type != 'header':
            # 非header类型，直接保留
            reclassified_blocks.append(block)
            continue
        
        x1, y1, x2, y2 = block.bbox
        # 转换为像素坐标
        x1_px = x1 * img_width
        y1_px = y1 * img_height
        x2_px = x2 * img_width
        y2_px = y2 * img_height
        
        width_px = x2_px - x1_px
        height_px = y2_px - y1_px
        area_px = width_px * height_px
        
        # 计算归一化值
        y1_norm = y1  # 已经是归一化坐标
        height_norm = height_px / img_height
        area_norm = area_px / page_area
        aspect_ratio = width_px / height_px if height_px > 0 else 0.0
        
        # 检查各项条件
        is_top_position = y1_norm < position_threshold
        is_small_height = height_norm < height_threshold
        is_large_area = area_norm > area_threshold
        is_wide = aspect_ratio > aspect_ratio_threshold
        
        # 检查下方区域是否有其他blocks
        has_nearby_content = False
        y2_norm = y2  # 已经是归一化坐标
        bottom_y = y2_norm
        check_bottom = bottom_y + nearby_content_threshold
        for other_block in blocks:
            if other_block == block:
                continue
            other_x1, other_y1, other_x2, other_y2 = other_block.bbox
            # 检查是否在下方区域内
            if bottom_y < other_y1 < check_bottom:
                has_nearby_content = True
                break
        
        # 分类逻辑
        if not is_top_position:
            # 不在顶部 → `text`
            new_block = ContentBlock(
                type='text',
                bbox=block.bbox,
                angle=block.angle,
                content=block.content
            )
            reclassified_blocks.append(new_block)
            header_corrected_count += 1
            logger.debug(
                f"header重新分类: 不在顶部 -> text "
                f"(y1={y1_norm:.3f}, height={height_norm:.3f}, area={area_norm:.3f})"
            )
        elif is_large_area and not is_small_height:
            # 面积大且高度不小 → `title`
            new_block = ContentBlock(
                type='title',
                bbox=block.bbox,
                angle=block.angle,
                content=block.content
            )
            reclassified_blocks.append(new_block)
            header_corrected_count += 1
            logger.debug(
                f"header重新分类: 面积大且高度不小 -> title "
                f"(y1={y1_norm:.3f}, height={height_norm:.3f}, area={area_norm:.3f})"
            )
        elif is_wide and is_top_position and has_nearby_content:
            # 宽且在上方，周围有内容 → `title`
            new_block = ContentBlock(
                type='title',
                bbox=block.bbox,
                angle=block.angle,
                content=block.content
            )
            reclassified_blocks.append(new_block)
            header_corrected_count += 1
            logger.debug(
                f"header重新分类: 宽且在上方，周围有内容 -> title "
                f"(y1={y1_norm:.3f}, height={height_norm:.3f}, aspect_ratio={aspect_ratio:.2f}, has_nearby={has_nearby_content})"
            )
        else:
            # 小且在上方 → 保持 `header`
            reclassified_blocks.append(block)
            logger.debug(
                f"header保持: 小且在上方 "
                f"(y1={y1_norm:.3f}, height={height_norm:.3f}, area={area_norm:.3f})"
            )
    
    if header_corrected_count > 0:
        logger.info(f"header重新分类完成，共修正 {header_corrected_count} 个header blocks")
    
    return reclassified_blocks


def verify_header_blocks_after_extraction(
    extract_results: List[List[ContentBlock]],
    images_pil_list: List[Image.Image],
    header_top_threshold: float = 0.15,
    header_height_threshold: float = 0.1,
    content_similarity_threshold: float = 0.7
) -> List[List[ContentBlock]]:
    """
    在内容提取后验证识别为header的blocks是否真的是页眉
    
    页眉特征：
    1. 位置：通常在页面顶部（y坐标 < header_top_threshold * 页面高度）
    2. 高度：页眉高度通常较小（高度 < header_height_threshold * 页面高度）
    3. 内容重复性：页眉内容在多页中可能重复或相似
    
    参数:
        extract_results: 内容提取后的结果（包含content）
        images_pil_list: 所有页面的PIL图片列表
        header_top_threshold: 页眉顶部位置阈值（默认0.15，即页面高度的15%）
        header_height_threshold: 页眉高度阈值（默认0.1，即页面高度的10%）
        content_similarity_threshold: 内容相似度阈值（默认0.7）
    
    返回:
        验证后的结果（header类型可能被修正为text）
    """
    verified_results = []
    
    # 收集所有header blocks的内容，用于检查重复性
    header_contents = []
    for page_idx, blocks in enumerate(extract_results):
        for block in blocks:
            if block.type == 'header' and block.content:
                header_contents.append((page_idx, block.content))
    
    # 计算内容相似度（简单的字符串相似度）
    def calculate_similarity(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        # 使用简单的字符重叠度
        str1_clean = str1.strip().lower()
        str2_clean = str2.strip().lower()
        if not str1_clean or not str2_clean:
            return 0.0
        # 如果完全相同，返回1.0
        if str1_clean == str2_clean:
            return 1.0
        # 计算字符集合的相似度
        set1 = set(str1_clean)
        set2 = set(str2_clean)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    # 验证每个页面的header blocks
    header_corrected_count = 0
    for page_idx, (blocks, image_pil) in enumerate(zip(extract_results, images_pil_list)):
        verified_blocks = []
        img_width, img_height = image_pil.size
        
        for block in blocks:
            if block.type == 'header':
                x1, y1, x2, y2 = block.bbox
                # 转换为像素坐标
                y1_px = y1 * img_height
                y2_px = y2 * img_height
                height_px = y2_px - y1_px
                
                # 检查位置和高度
                is_top_position = y1_px < (header_top_threshold * img_height)
                is_small_height = height_px < (header_height_threshold * img_height)
                
                # 检查内容重复性
                has_similar_content = False
                if block.content:
                    for other_page_idx, other_content in header_contents:
                        if other_page_idx != page_idx:
                            similarity = calculate_similarity(block.content, other_content)
                            if similarity > content_similarity_threshold:
                                has_similar_content = True
                                break
                
                # 页眉验证：至少满足位置或高度条件，且最好有内容重复性
                # 如果位置在顶部，或者（位置合理且内容重复），则认为是页眉
                is_likely_header = is_top_position or (is_small_height and has_similar_content)
                
                if is_likely_header:
                    # 确认是页眉，保持header类型
                    verified_blocks.append(block)
                    logger.debug(
                        f"页面 {page_idx + 1}: 确认header block "
                        f"(位置: y1={y1_px:.1f}, 高度={height_px:.1f}, "
                        f"顶部位置={is_top_position}, 小高度={is_small_height}, "
                        f"内容重复={has_similar_content}, 内容='{block.content[:50] if block.content else ''}')"
                    )
                else:
                    # 不是页眉，修正为text
                    corrected_block = ContentBlock(
                        type='text',
                        bbox=block.bbox,
                        angle=block.angle,
                        content=block.content
                    )
                    verified_blocks.append(corrected_block)
                    header_corrected_count += 1
                    logger.info(
                        f"页面 {page_idx + 1}: header block被修正为text "
                        f"(位置: y1={y1_px:.1f}, 高度={height_px:.1f}, "
                        f"顶部位置={is_top_position}, 小高度={is_small_height}, "
                        f"内容重复={has_similar_content}, 内容='{block.content[:50] if block.content else ''}')"
                    )
            else:
                # 非header类型，直接保留
                verified_blocks.append(block)
        
        verified_results.append(verified_blocks)
    
    if header_corrected_count > 0:
        logger.info(f"页眉验证完成，共修正 {header_corrected_count} 个header blocks为text")
    else:
        logger.info("页眉验证完成，所有header blocks均通过验证")
    
    return verified_results


def merge_layouts_expert_model(
    mineru_blocks: List[ContentBlock],
    dolphin_blocks: List[ContentBlock],
    image: Image.Image,
    iou_threshold: float = 0.7
) -> List[ContentBlock]:
    """
    专家模型融合逻辑：按block类型动态调整两个模型的权重
    
    融合策略：
    1. 表格类型：任一模型识别为表格就判定为表格
       - 如果两个模型都识别为表格，使用更大的bbox
       - 否则使用识别为表格的那个模型的bbox
    2. title vs text：Dolphin识别为title，MinerU识别为text，则判定为title（title更信任Dolphin）
    3. 其他类型（image、equation、code等）：同等信任，优先使用MinerU的结果
    4. 边界框定位：除了表格（两个模型都识别时用更大的），其他情况都用MinerU的bbox
    
    参数:
        mineru_blocks: MinerU VLM的blocks列表
        dolphin_blocks: Dolphin的blocks列表
        image: PIL图片
        iou_threshold: IoU阈值（默认0.7）
    
    返回:
        融合后的blocks列表
    """
    merged_blocks = []
    used_dolphin_indices = set()
    
    # 以MinerU为主，遍历所有MinerU blocks
    for mineru_block in mineru_blocks:
        # 寻找IoU最高的Dolphin block进行匹配
        best_match_idx = -1
        best_iou = 0.0
        best_dolphin_block = None
        
        for j, dolphin_block in enumerate(dolphin_blocks):
            if j in used_dolphin_indices:
                continue
            
            iou = calculate_iou(mineru_block.bbox, dolphin_block.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
                best_dolphin_block = dolphin_block
        
        # 如果找到匹配的Dolphin block（IoU > threshold）
        if best_match_idx >= 0 and best_iou > iou_threshold and best_dolphin_block:
            dolphin_block = best_dolphin_block
            
            # 融合策略：按类型动态调整
            # 1. 表格类型：任一模型识别为表格就判定为表格
            if mineru_block.type == 'table' or dolphin_block.type == 'table':
                selected_type = 'table'
                
                # 如果两个模型都识别为表格，使用更大的bbox
                if mineru_block.type == 'table' and dolphin_block.type == 'table':
                    mineru_area = calculate_bbox_area(mineru_block.bbox)
                    dolphin_area = calculate_bbox_area(dolphin_block.bbox)
                    if dolphin_area > mineru_area:
                        selected_bbox = dolphin_block.bbox
                        logger.debug(
                            f"表格融合: 两个模型都识别为表格，使用Dolphin的更大bbox "
                            f"(MinerU面积={mineru_area:.4f}, Dolphin面积={dolphin_area:.4f})"
                        )
                    else:
                        selected_bbox = mineru_block.bbox
                        logger.debug(
                            f"表格融合: 两个模型都识别为表格，使用MinerU的更大bbox "
                            f"(MinerU面积={mineru_area:.4f}, Dolphin面积={dolphin_area:.4f})"
                        )
                elif mineru_block.type == 'table':
                    # MinerU识别为表格，Dolphin不是，使用MinerU的bbox
                    selected_bbox = mineru_block.bbox
                    logger.debug(
                        f"表格融合: MinerU识别为表格，Dolphin={dolphin_block.type}，使用MinerU的bbox"
                    )
                else:
                    # Dolphin识别为表格，MinerU不是，使用Dolphin的bbox
                    selected_bbox = dolphin_block.bbox
                    logger.debug(
                        f"表格融合: Dolphin识别为表格，MinerU={mineru_block.type}，使用Dolphin的bbox"
                    )
            
            # 2. title vs text：Dolphin识别为title，MinerU识别为text，则判定为title
            elif dolphin_block.type == 'title' and mineru_block.type == 'text':
                selected_type = 'title'
                selected_bbox = mineru_block.bbox  # 边界框定位信任MinerU
                logger.debug(
                    f"类型融合: Dolphin=title, MinerU=text -> 判定为title（title更信任Dolphin）"
                )
            
            # 3. 其他类型：同等信任，优先使用MinerU的结果
            else:
                selected_type = mineru_block.type  # 优先使用MinerU的结果
                selected_bbox = mineru_block.bbox  # 边界框定位信任MinerU
                if mineru_block.type != dolphin_block.type:
                    logger.debug(
                        f"类型融合: MinerU={mineru_block.type}, Dolphin={dolphin_block.type} -> "
                        f"使用MinerU的结果（其他类型优先MinerU）"
                    )
            
            merged_block = ContentBlock(
                type=selected_type,
                bbox=selected_bbox,
                angle=mineru_block.angle,
                content=mineru_block.content
            )
            merged_blocks.append(merged_block)
            used_dolphin_indices.add(best_match_idx)
        else:
            # 没有匹配的Dolphin block，直接使用MinerU的结果
            merged_blocks.append(mineru_block)
    
    # 添加Dolphin独有的blocks（如果Dolphin识别为表格，但MinerU没有识别到）
    for j, dolphin_block in enumerate(dolphin_blocks):
        if j in used_dolphin_indices:
            continue
        
        # 如果Dolphin识别为表格，检查是否与已有blocks重叠
        is_overlapping = False
        for merged_block in merged_blocks:
            iou = calculate_iou(dolphin_block.bbox, merged_block.bbox)
            if iou > iou_threshold:
                is_overlapping = True
                break
        
        # 如果Dolphin识别为表格且不与已有blocks重叠，添加它
        if dolphin_block.type == 'table' and not is_overlapping:
            merged_blocks.append(dolphin_block)
            logger.debug(f"添加Dolphin独有的table block")
    
    return merged_blocks


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
           - 如果Dolphin识别为table、title或image：分类必须保持（以Dolphin为准，不修正）
             * Dolphin识别为标题的都按Dolphin的来（包括title、sec_0、sec_1等）
             * image类型必须保持，避免被误判为其他类型（特别是text）
           - 如果分类一致：使用Dolphin的分类（保持不变）
           - 如果分类不一致：
             * image类型保护：如果任一识别为image，优先使用image类型（避免图片被误判为text）
             * header冲突：如果一个识别成header，另一个识别成text或其他类型，按识别成text或其他类型的来
             * table vs text冲突：如果MinerU识别为table，修正为table（避免表格被误识别为文本）
             * 其他情况：保持Dolphin的分类（以Dolphin为主，不修正）
         * bbox检查修正：
           - 如果Dolphin识别为table：如果MinerU更大则扩大，否则保持Dolphin
           - 如果Dolphin识别为header：用MinerU的bbox修正定位（Dolphin定位可能有问题）
           - 其他类型：如果Dolphin的定位覆盖有问题，使用MinerU的bbox进行修正
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
            # 1. 如果Dolphin识别为table、title或image，分类必须保持（以Dolphin为准，不修正）
            #    - Dolphin识别为标题的都按Dolphin的来（包括title、sec_0、sec_1等）
            #    - image类型必须保持，避免被误判为其他类型
            # 2. 如果两个分类一致，使用Dolphin的分类（保持不变）
            # 3. 如果分类不一致：
            #    - image类型保护：如果任一识别为image，优先使用image类型（避免图片被误判为text）
            #    - header冲突：如果一个识别成header，另一个识别成text或其他类型，按识别成text或其他类型的来
            #    - table vs text冲突：如果MinerU识别为table，修正为table（避免表格被误识别为文本）
            #    - 其他情况：保持Dolphin的分类（以Dolphin为主，不修正）
            if dolphin_block.type in ['table', 'title', 'image']:
                # Dolphin识别为table、title或image，分类必须保持（以Dolphin为准）
                # 特别是title和image：必须保持Dolphin的分类
                selected_type = dolphin_block.type
                if dolphin_block.type == 'title' and mineru_block.type != 'title':
                    logger.debug(f"保持title分类: MinerU分类={mineru_block.type}，使用Dolphin的title分类（Dolphin识别为标题的都按Dolphin的来）")
                elif dolphin_block.type == 'image' and mineru_block.type != 'image':
                    logger.debug(f"保持image分类: MinerU分类={mineru_block.type}，使用Dolphin的image分类（保护image类型，避免被误判）")
            elif dolphin_block.type == mineru_block.type:
                selected_type = dolphin_block.type  # 分类一致，保持Dolphin的分类
            else:
                # 分类不一致的情况
                types_set = {dolphin_block.type, mineru_block.type}
                # 优先保护image类型：如果任一识别为image，使用image类型（避免image被误判为text）
                if 'image' in types_set:
                    selected_type = 'image'
                    if dolphin_block.type != 'image' and mineru_block.type == 'image':
                        logger.info(f"image类型保护修正: Dolphin={dolphin_block.type} -> MinerU=image -> 修正为image（避免图片被误判为text）")
                    else:
                        logger.debug(f"image类型保护: Dolphin={dolphin_block.type}, MinerU={mineru_block.type} -> 使用image类型")
                elif 'header' in types_set:
                    # header冲突：如果一个识别成header，另一个识别成text或其他类型，按识别成text或其他类型的来
                    if dolphin_block.type == 'header':
                        # Dolphin识别为header，但MinerU识别为其他类型，使用MinerU的分类
                        selected_type = mineru_block.type
                        logger.debug(f"header冲突修正: Dolphin=header -> MinerU={mineru_block.type} -> 使用MinerU的分类（按识别成text或其他类型的来）")
                    else:
                        # MinerU识别为header，但Dolphin识别为其他类型，使用Dolphin的分类
                        selected_type = dolphin_block.type
                        logger.debug(f"header冲突修正: MinerU=header -> Dolphin={dolphin_block.type} -> 使用Dolphin的分类（按识别成text或其他类型的来）")
                elif 'table' in types_set and 'text' in types_set:
                    # table和text冲突时，如果MinerU识别为table，修正为table（避免表格被误识别为文本）
                    if mineru_block.type == 'table':
                        selected_type = 'table'
                        logger.debug(f"修正分类: Dolphin={dolphin_block.type} -> MinerU=table -> 修正为table")
                    else:
                        selected_type = dolphin_block.type
                else:
                    # 其他情况保持Dolphin的分类（以Dolphin为主，不修正）
                    selected_type = dolphin_block.type
            
            # bbox检查修正策略（原有策略）：
            # - 表格：如果MinerU更大则扩大，否则保持Dolphin
            # - header：如果Dolphin识别为header，保持header分类，但用MinerU的bbox修正定位（Dolphin定位可能有问题）
            # - 其他：如果Dolphin的定位覆盖有问题，使用MinerU的bbox进行修正（选择更合理的bbox）
            dolphin_area = calculate_bbox_area(dolphin_block.bbox)
            mineru_area = calculate_bbox_area(mineru_block.bbox)
            
            if selected_type == 'table':
                # 表格：如果MinerU更大则扩大，否则保持Dolphin
                if mineru_area > dolphin_area:
                    selected_bbox = mineru_block.bbox
                    logger.debug(f"表格bbox扩大: 使用MinerU的bbox (Dolphin面积={dolphin_area:.4f}, MinerU面积={mineru_area:.4f})")
                else:
                    selected_bbox = dolphin_block.bbox
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
                bbox=selected_bbox,   # bbox：Dolphin识别为table时如果MinerU更大则扩大，否则保持Dolphin
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


def get_fused_layout_blocks(
    images_pil_list: List[Image.Image],
    dolphin_model_path: str,
    mineru_client: MinerUClient,
    iou_threshold: float = 0.7,
    skip_dolphin: bool = False
) -> List[List[ContentBlock]]:
    """
    获取融合后的布局blocks（可被MinerU项目调用）
    
    Args:
        images_pil_list: PIL图片列表
        dolphin_model_path: Dolphin模型路径
        mineru_client: MinerU VLM客户端
        iou_threshold: IoU阈值
        skip_dolphin: 是否跳过Dolphin（仅使用MinerU）
    
    Returns:
        融合后的布局blocks列表（每页一个ContentBlock列表）
    """
    merged_results = []
    
    # 1. 获取Dolphin布局结果
    dolphin_results_list = []
    if skip_dolphin or not DOLPHIN_AVAILABLE:
        logger.info("跳过Dolphin布局识别")
        dolphin_results_list = [[] for _ in images_pil_list]
    else:
        logger.info("正在使用Dolphin进行布局识别...")
        dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
        if dolphin_model is None:
            logger.warning("Dolphin模型不可用，将仅使用MinerU VLM结果")
            dolphin_results_list = [[] for _ in images_pil_list]
        else:
            try:
                for idx, image in enumerate(images_pil_list):
                    logger.debug(f"  Dolphin识别页面 {idx + 1}/{len(images_pil_list)}...")
                    layout_str = dolphin_model.chat("Parse the reading order of this document.", image)
                    dolphin_results = parse_layout_string(layout_str)
                    
                    if not dolphin_results or not (layout_str.startswith("[") and layout_str.endswith("]")):
                        logger.warning(f"  页面 {idx + 1} Dolphin识别结果无效，使用默认distorted_page")
                        img_width, img_height = image.size
                        dolphin_results = [([0, 0, img_width, img_height], 'distorted_page', [])]
                    
                    dolphin_results_list.append(dolphin_results)
            except Exception as e:
                logger.error(f"Dolphin布局识别失败: {e}")
                dolphin_results_list = [[] for _ in images_pil_list]
    
    # 2. 获取MinerU VLM布局结果
    logger.info("正在使用MinerU VLM进行布局识别...")
    try:
        mineru_results = mineru_client.batch_layout_detect(images=images_pil_list)
    except Exception as e:
        logger.error(f"MinerU VLM布局识别失败: {e}")
        mineru_results = [[] for _ in images_pil_list]
    
    # 3. 融合布局
    logger.info("正在融合布局识别结果...")
    for page_idx, (dolphin_results, mineru_blocks, image_pil) in enumerate(
        zip(dolphin_results_list, mineru_results, images_pil_list)
    ):
        # 转换Dolphin结果为ContentBlock
        dolphin_blocks = convert_dolphin_to_content_blocks(dolphin_results, image_pil)
        
        # 比对和融合
        merged_blocks = compare_and_merge_layouts(
            dolphin_blocks,
            mineru_blocks,
            mineru_client=mineru_client,
            image=image_pil,
            iou_threshold=iou_threshold
        )
        
        merged_results.append(merged_blocks)
        logger.debug(
            f"页面 {page_idx + 1}: Dolphin {len(dolphin_blocks)}个blocks, "
            f"MinerU VLM {len(mineru_blocks)}个blocks, "
            f"融合后 {len(merged_blocks)}个blocks"
        )
    
    return merged_results


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
    处理单个文档（Dolphin布局分析 + MinerU VLM内容提取）
    
    功能：
    1. 布局分析：使用Dolphin进行全页面布局识别（主要方法）
    2. 内容提取：如果启用MinerU VLM（未使用--skip-mineru），会基于Dolphin的布局结果进行内容提取
    """
    
    # 加载PDF并转换为图片
    # 采用Dolphin的文档处理方式
    # Dolphin的convert_pdf_to_images会将最长边缩放到896像素
    images_pil_list = []
    pdf_bytes = None
    images_list = None
    pdf_doc = None
    
    if DOLPHIN_AVAILABLE:
        try:
            logger.info("使用Dolphin的PDF转换方式（target_size=896）...")
            images_pil_list = convert_pdf_to_images(input_path, target_size=896)
            if not images_pil_list:
                raise Exception("Dolphin PDF转换失败")
            logger.info(f"Dolphin PDF转换完成，共 {len(images_pil_list)} 页")
            # 为了内容提取，也需要加载PDF字节和images_list
            with open(input_path, 'rb') as f:
                pdf_bytes = f.read()
            images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        except Exception as e:
            logger.error(f"Dolphin PDF转换失败: {e}")
            raise
    else:
        logger.error("Dolphin不可用，无法进行PDF转换")
        raise RuntimeError("Dolphin不可用，无法进行PDF转换")
    
    output_file_name = Path(input_path).stem
    # 为每个文件创建独立的输出目录
    file_output_dir = os.path.join(output_dir, output_file_name)
    os.makedirs(file_output_dir, exist_ok=True)
    logger.info(f"正在处理文档: {input_path}，共 {len(images_pil_list)} 页")
    logger.info(f"输出目录: {file_output_dir}")
    
    # 开始布局识别阶段计时
    import time
    layout_start_time = time.time()
    
    # 1. 使用Dolphin进行布局识别（全页面布局识别，主要方法）
    dolphin_results_list = []
    merged_results = []
    mineru_client = None
    
    if skip_dolphin or not DOLPHIN_AVAILABLE:
        logger.error("Dolphin是布局分析的主要方法，不可跳过或不可用")
        if not DOLPHIN_AVAILABLE:
            logger.error("Dolphin模块不可用，请检查安装")
        raise RuntimeError("Dolphin是布局分析的主要方法，必须可用")
    else:
        logger.info("正在使用Dolphin进行全页面布局识别（主要方法）...")
        dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
        if dolphin_model is None:
            logger.error("Dolphin模型加载失败，无法继续处理")
            raise RuntimeError("Dolphin模型加载失败")
        
        try:
            # 对所有页面使用Dolphin进行布局识别
            for page_idx, image in enumerate(images_pil_list):
                logger.debug(f"  Dolphin识别页面 {page_idx + 1}/{len(images_pil_list)}...")
                layout_str = dolphin_model.chat("Parse the reading order of this document.", image)
                dolphin_results = parse_layout_string(layout_str)
                
                if not dolphin_results or not (layout_str.startswith("[") and layout_str.endswith("]")):
                    logger.warning(f"  页面 {page_idx + 1} Dolphin识别结果无效，使用默认distorted_page")
                    img_width, img_height = image.size
                    dolphin_results = [([0, 0, img_width, img_height], 'distorted_page', [])]
                
                dolphin_results_list.append(dolphin_results)
                
                # 转换Dolphin结果为ContentBlock
                dolphin_blocks = convert_dolphin_to_content_blocks(dolphin_results, image)
                merged_results.append(dolphin_blocks)
            
            logger.info(f"Dolphin布局识别完成，共识别 {sum(len(r) for r in merged_results)} 个blocks")
        except Exception as e:
            logger.error(f"Dolphin布局识别失败: {e}")
            logger.error("Dolphin是主要方法，识别失败将导致处理失败")
            import traceback
            traceback.print_exc()
            raise
    
    # 2. 如果需要内容提取，加载MinerU VLM客户端
    if not skip_mineru:
        logger.info("正在加载MinerU VLM客户端用于内容提取...")
        try:
            mineru_client = ModelSingleton.get_mineru_client(mineru_backend, mineru_model_path)
            logger.info("MinerU VLM客户端加载完成，将用于内容提取")
        except Exception as e:
            logger.warning(f"MinerU VLM客户端加载失败: {e}")
            logger.warning("将跳过内容提取，仅输出布局识别结果")
            mineru_client = None
    
    # 结束布局识别阶段计时
    layout_end_time = time.time()
    layout_processing_time = layout_end_time - layout_start_time
    logger.info(f"布局识别阶段完成，耗时: {layout_processing_time:.2f} 秒")
    
    # 检查是否有可用的结果
    if not merged_results or all(len(r) == 0 for r in merged_results):
        logger.error("=" * 80)
        logger.error("错误：布局识别结果为空，无法继续处理")
        logger.error("=" * 80)
        raise RuntimeError("布局识别结果为空，无法继续处理")
    
    # 输出布局识别结果统计
    for page_idx, merged_blocks in enumerate(merged_results):
        dolphin_count = len(dolphin_results_list[page_idx]) if page_idx < len(dolphin_results_list) else 0
        logger.info(
            f"页面 {page_idx + 1}: Dolphin识别 {dolphin_count}个blocks, "
            f"最终布局 {len(merged_blocks)}个blocks"
        )
    
    
    # 4. 生成可视化PDF
    output_pdf_path = os.path.join(file_output_dir, f"{output_file_name}_layout.pdf")
    
    logger.info("正在生成布局可视化PDF...")
    visualize_layout_pdf(images_pil_list, merged_results, output_pdf_path)
    
    # 5. 保存结果JSON
    output_json_path = os.path.join(file_output_dir, f"{output_file_name}_layout.json")
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
    
    # 6. 如果启用了MinerU VLM，进行内容提取和后处理（使用Dolphin的布局结果）
    # 开始内容提取阶段计时
    extraction_start_time = time.time()
    
    if not skip_mineru and mineru_client is not None and pdf_bytes is not None and images_list is not None:
        logger.info("正在使用Dolphin的布局结果进行内容提取和后处理...")
        try:
            # 使用Dolphin的布局结果进行内容提取（会自动进行post_process后处理）
            extract_results = mineru_client.batch_two_step_extract(
                images=images_pil_list,
                fused_layout_blocks=merged_results  # 使用Dolphin的布局结果
            )
            
            # Header重新分类（在内容提取之后，基于布局信息）
            logger.info("正在重新分类header blocks（基于位置、高度、面积、宽高比、周围内容）...")
            extract_results_before_header_reclassify = extract_results
            extract_results = []
            for page_idx, (blocks, image_pil) in enumerate(zip(extract_results_before_header_reclassify, images_pil_list)):
                reclassified_blocks = reclassify_header_blocks(blocks, image_pil)
                extract_results.append(reclassified_blocks)
            
            # 生成MinerU标准格式输出（包括MagicModel处理、表格跨页合并、标题分级等）
            logger.info("正在生成MinerU标准格式输出（包括后处理）...")
            local_image_dir = os.path.join(file_output_dir, f"{output_file_name}_images")
            local_md_dir = file_output_dir
            os.makedirs(local_image_dir, exist_ok=True)
            
            image_writer = FileBasedDataWriter(local_image_dir)
            # result_to_middle_json会进行：
            # 1. MagicModel处理（分类不同类型的blocks：image, table, title, code等）
            # 2. 图片/表格/公式的截图处理
            # 3. 表格跨页合并
            # 4. LLM优化标题分级（如果启用）
            middle_json = result_to_middle_json(extract_results, images_list, pdf_doc, image_writer)
            
            # 保存middle.json
            middle_json_path = os.path.join(file_output_dir, f"{output_file_name}_middle.json")
            with open(middle_json_path, 'w', encoding='utf-8') as f:
                json.dump(middle_json, f, ensure_ascii=False, indent=2)
            logger.info(f"MinerU中间JSON已保存到: {middle_json_path}")
            
            # 保存model.json（原始输出，包含content）
            model_json_path = os.path.join(file_output_dir, f"{output_file_name}_model.json")
            model_json = []
            for page_idx, blocks in enumerate(extract_results):
                page_blocks = []
                for block in blocks:
                    page_blocks.append({
                        "type": block.type,
                        "bbox": block.bbox,
                        "angle": block.angle,
                        "content": block.content
                    })
                model_json.append({
                    "page": page_idx + 1,
                    "blocks": page_blocks
                })
            with open(model_json_path, 'w', encoding='utf-8') as f:
                json.dump(model_json, f, ensure_ascii=False, indent=2)
            logger.info(f"MinerU模型原始输出JSON已保存到: {model_json_path}")
            
            # 使用union_make生成content_list.json（完整的MinerU处理流程）
            logger.info("正在生成content_list.json...")
            pdf_info = middle_json["pdf_info"]
            image_dir = os.path.basename(local_image_dir)
            content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            content_list_path = os.path.join(file_output_dir, f"{output_file_name}_content_list.json")
            with open(content_list_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            logger.info(f"MinerU内容列表JSON已保存到: {content_list_path}")
            
            # 使用union_make生成Markdown文件（完整的MinerU处理流程）
            logger.info("正在生成Markdown文件...")
            md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            md_path = os.path.join(file_output_dir, f"{output_file_name}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content_str)
            logger.info(f"Markdown文件已保存到: {md_path}")
            
            logger.info("内容提取和后处理完成")
        except Exception as e:
            logger.warning(f"内容提取和后处理失败: {e}")
            import traceback
            traceback.print_exc()
            logger.warning("将继续处理，但不会生成内容提取结果")
    
    logger.info(f"处理完成: {input_path}")




def main():
    parser = argparse.ArgumentParser(
        description="Dolphin布局分析 + MinerU VLM内容提取（Dolphin进行布局识别，MinerU进行内容提取）"
    )
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
        help="跳过Dolphin模型加载（不推荐，Dolphin是布局分析的主要方法，必需）"
    )
    parser.add_argument(
        "--skip-mineru",
        action="store_true",
        default=False,
        help="跳过MinerU VLM模型加载（MinerU VLM用于内容提取，可以跳过，但会只输出布局结果）"
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
            logger.info("提示：如果这是首次下载，可能需要较长时间，请耐心等待...")
            # 使用默认的模型根目录（不是HuggingFace缓存路径）
            model_root_for_download = os.getenv("MINERU_MODEL_ROOT", os.path.expanduser("~/mineru_models"))
            os.makedirs(model_root_for_download, exist_ok=True)
            # auto_download_and_get_model_root_path 会：
            # 1. 先检查 HuggingFace 缓存（~/.cache/huggingface/hub/）
            # 2. 如果缓存中有模型，直接返回缓存路径（不会重新下载）
            # 3. 如果缓存中没有，才会下载到缓存目录（可能需要较长时间）
            logger.info("正在检查/下载MinerU VLM模型（这可能需要几分钟）...")
            try:
                downloaded_path = auto_download_and_get_model_root_path(model_root_for_download, "vlm")
                actual_mineru_model_path = downloaded_path
                logger.info(f"MinerU VLM模型路径: {actual_mineru_model_path}")
            except Exception as e:
                logger.error(f"MinerU VLM模型检查/下载失败: {e}")
                logger.warning("如果持续失败，建议使用 --skip-mineru 跳过MinerU VLM内容提取")
                logger.warning("注意：跳过MinerU VLM将只输出Dolphin的布局识别结果，不进行内容提取")
                raise
            # 检查是否是 HuggingFace 缓存路径
            if '/.cache/huggingface/' in actual_mineru_model_path or '/huggingface/hub/' in actual_mineru_model_path:
                logger.info(f"提示：模型存储在 HuggingFace 缓存目录中，这是正常的")
                logger.info(f"提示：模型不会自动删除，后续运行将直接使用缓存，无需重新下载")
            else:
                logger.info(f"提示：后续运行将直接使用此路径，无需重新下载")
    
    logger.info("正在预加载模型（仅加载一次，后续复用）...")
    
    # 先加载Dolphin模型（布局分析的主要方法，必需）
    dolphin_model = None
    if args.skip_dolphin:
        logger.error("跳过Dolphin模型加载（--skip-dolphin选项）")
        logger.error("Dolphin是布局分析的主要方法，必须加载，无法跳过")
        sys.exit(1)
    elif DOLPHIN_AVAILABLE:
        # 检查Dolphin模型路径是否存在
        dolphin_model_path = args.dolphin_model_path
        if not os.path.exists(dolphin_model_path):
            logger.error(f"Dolphin模型路径不存在: {dolphin_model_path}")
            logger.error("Dolphin是布局分析的主要方法，模型路径不存在将导致处理失败")
            logger.info(f"提示：请确保Dolphin模型已下载到: {dolphin_model_path}")
            logger.info("或者使用 --dolphin-model-path 指定正确的模型路径")
            sys.exit(1)
        else:
            try:
                dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
                if dolphin_model is not None:
                    logger.info("Dolphin模型预加载完成（布局分析的主要方法）")
                else:
                    logger.error("Dolphin模型不可用，无法继续处理")
                    logger.error("Dolphin是布局分析的主要方法，必须可用")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"Dolphin模型预加载失败: {e}")
                logger.error("Dolphin是布局分析的主要方法，加载失败将导致处理失败")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    else:
        logger.error("Dolphin不可用，无法继续处理")
        logger.error("Dolphin是布局分析的主要方法，必须可用")
        sys.exit(1)
    
    # 尝试加载MinerU VLM模型（内容提取方法，可选）
    if args.skip_mineru:
        logger.info("跳过MinerU VLM模型加载（--skip-mineru选项）")
        logger.info("将仅进行布局识别，不进行内容提取")
    else:
        try:
            ModelSingleton.get_mineru_client(args.mineru_backend, actual_mineru_model_path)
            logger.info("MinerU VLM模型预加载完成（用于内容提取）")
        except RuntimeError as e:
            error_msg = str(e)
            if "CUDA out of memory" in error_msg or "out of memory" in error_msg.lower():
                logger.warning("=" * 80)
                logger.warning("GPU显存不足，无法加载MinerU VLM模型（用于内容提取）")
                logger.warning("=" * 80)
                logger.warning(f"错误详情: {error_msg}")
                logger.warning("")
                logger.warning("MinerU VLM用于内容提取，加载失败将跳过内容提取")
                logger.warning("将仅输出Dolphin的布局识别结果")
                logger.warning("=" * 80)
                logger.warning("提示：可以使用 --skip-mineru 选项跳过MinerU VLM模型加载")
            else:
                logger.warning(f"MinerU VLM模型预加载失败（用于内容提取）: {e}")
                logger.warning("MinerU VLM用于内容提取，加载失败将跳过内容提取")
                logger.warning("将仅输出Dolphin的布局识别结果")
        except Exception as e:
            logger.warning(f"MinerU VLM模型预加载失败（用于内容提取）: {e}")
            logger.warning("MinerU VLM用于内容提取，加载失败将跳过内容提取")
            logger.warning("将仅输出Dolphin的布局识别结果")
    
    logger.info("模型预加载完成，开始处理文档...")
    
    # 计时起点：模型加载完成后开始计时
    import time
    start_time = time.time()
    start_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"处理开始时间: {start_time_str}")
    
    # 处理每个文件（复用已加载的模型）
    processed_files_count = 0
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
        else:
            processed_files_count += 1
    
    # 计时终点：所有文件处理完成后结束计时
    end_time = time.time()
    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_processing_time = end_time - start_time
    
    logger.info(f"处理结束时间: {end_time_str}")
    logger.info(f"总处理时间: {total_processing_time:.2f} 秒")
    logger.info(f"处理的文件数量: {processed_files_count}")
    
    # 保存处理时间统计到JSON文件
    processing_time_json = {
        "total_processing_time_seconds": round(total_processing_time, 2),
        "total_files_processed": processed_files_count,
        "start_time": start_time_str,
        "end_time": end_time_str
    }
    
    processing_time_path = os.path.join(args.output, "processing_time.json")
    with open(processing_time_path, 'w', encoding='utf-8') as f:
        json.dump(processing_time_json, f, ensure_ascii=False, indent=2)
    logger.info(f"处理时间统计已保存到: {processing_time_path}")
    
    logger.info("所有文件处理完成")


if __name__ == "__main__":
    main()

