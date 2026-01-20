#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image
import cv2
import numpy as np

if 'TOKENIZERS_PARALLELISM' not in os.environ:
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

DEFAULT_MODEL_ROOT = os.getenv("MINERU_MODEL_ROOT", os.path.expanduser("~/mineru_models"))
DEFAULT_MINERU_VLM_PATH = os.path.join(DEFAULT_MODEL_ROOT, "vlm")
DEFAULT_DOLPHIN_PATH = os.getenv("DOLPHIN_MODEL_PATH", "/home/hsr/Dolphin/hf_model")

dolphin_path = os.getenv("DOLPHIN_PATH", "/home/hsr/Dolphin")
if os.path.exists(dolphin_path) and dolphin_path not in sys.path:
    sys.path.insert(0, dolphin_path)

try:
    from demo_layout import DOLPHIN
    from utils.utils import parse_layout_string, process_coordinates, convert_pdf_to_images
    DOLPHIN_AVAILABLE = True
except ImportError as e:
    print(f"警告: Dolphin模块导入失败: {e}")
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


class ModelSingleton:
    _mineru_client = None
    _mineru_backend = None
    _mineru_model_path = None
    _dolphin_model = None
    _dolphin_model_path = None
    _dolphin_load_failed = False
    
    @classmethod
    def get_mineru_client(cls, backend: str, model_path: str = None):
        actual_model_path = model_path
        if actual_model_path is None and backend != "http-client":
            if os.path.exists(DEFAULT_MINERU_VLM_PATH):
                actual_model_path = DEFAULT_MINERU_VLM_PATH
                logger.info(f"使用本地MinerU VLM模型路径: {actual_model_path}")
            else:
                logger.info("未找到本地MinerU VLM模型，检查 HuggingFace 缓存或自动下载...")
                os.makedirs(DEFAULT_MODEL_ROOT, exist_ok=True)
                downloaded_path = auto_download_and_get_model_root_path(DEFAULT_MODEL_ROOT, "vlm")
                actual_model_path = downloaded_path
                logger.info(f"MinerU VLM模型路径: {actual_model_path}")
        
        if (cls._mineru_client is not None and 
            cls._mineru_backend == backend and 
            cls._mineru_model_path == actual_model_path):
            logger.debug("复用已加载的MinerU VLM模型")
            return cls._mineru_client
        
        if cls._mineru_client is None:
            logger.info(f"正在加载MinerU VLM模型 (backend: {backend}, path: {actual_model_path})...")
        else:
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
        cls._mineru_model_path = actual_model_path
        logger.info("MinerU VLM模型加载完成")
        return cls._mineru_client
    
    @classmethod
    def get_dolphin_model(cls, model_path: str):
        if not DOLPHIN_AVAILABLE:
            return None
        
        if cls._dolphin_load_failed:
            return None
        
        if not os.path.exists(model_path):
            logger.error(f"Dolphin模型路径不存在: {model_path}")
            logger.warning("Dolphin模型路径不存在，将仅使用MinerU VLM结果")
            cls._dolphin_load_failed = True
            return None
        
        if cls._dolphin_model is not None and cls._dolphin_model_path == model_path:
            logger.debug("复用已加载的Dolphin模型")
            return cls._dolphin_model
        
        logger.info(f"正在加载Dolphin模型到本地内存 (path: {model_path})...")
        try:
            cls._dolphin_model = DOLPHIN(model_path)
            cls._dolphin_model_path = model_path
            cls._dolphin_load_failed = False
            logger.info("Dolphin模型加载完成")
            return cls._dolphin_model
        except Exception as e:
            logger.error(f"Dolphin模型加载失败: {e}")
            cls._dolphin_model = None
            cls._dolphin_model_path = None
            cls._dolphin_load_failed = True
            return None


DOLPHIN_TO_MINERU_TYPE_MAP = {
    'para': 'text',
    'title': 'title',
    'table': 'table',
    'tab': 'table',
    'figure': 'image',
    'fig': 'image',
    'image': 'image',
    'img': 'image',
    'picture': 'image',
    'formula': 'equation',
    'equ': 'equation',
    'code': 'code',
    'list': 'list',
    'header': 'header',
    'footer': 'footer',
    'foot': 'footer',
    'page_number': 'page_number',
    'fnote': 'page_footnote',
    'distorted_page': 'text',
    'sec_0': 'title',
    'sec_1': 'title',
    'sec_2': 'title',
    'sec_3': 'title',
    'sec_4': 'title',
    'author': 'text',
    'paper_abstract': 'text',
    'watermark': 'aside_text',
    'meta_num': 'text',
}


def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
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
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def normalize_bbox(bbox: List[float], img_width: int, img_height: int) -> List[float]:
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
    img_width, img_height = image.size
    blocks = []
    
    for coords, label, tags in dolphin_results:
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
    for existing_block in existing_blocks:
        iou = calculate_iou(bbox, existing_block.bbox)
        if iou > coverage_threshold:
            return True
    return False


def verify_header_blocks_after_extraction(
    extract_results: List[List[ContentBlock]],
    images_pil_list: List[Image.Image],
    header_top_threshold: float = 0.15,
    header_height_threshold: float = 0.1,
    content_similarity_threshold: float = 0.7
) -> List[List[ContentBlock]]:
    verified_results = []
    
    header_contents = []
    for page_idx, blocks in enumerate(extract_results):
        for block in blocks:
            if block.type == 'header' and block.content:
                header_contents.append((page_idx, block.content))
    
    def calculate_similarity(str1: str, str2: str) -> float:
        if not str1 or not str2:
            return 0.0
        str1_clean = str1.strip().lower()
        str2_clean = str2.strip().lower()
        if not str1_clean or not str2_clean:
            return 0.0
        if str1_clean == str2_clean:
            return 1.0
        set1 = set(str1_clean)
        set2 = set(str2_clean)
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    header_corrected_count = 0
    for page_idx, (blocks, image_pil) in enumerate(zip(extract_results, images_pil_list)):
        verified_blocks = []
        img_width, img_height = image_pil.size
        
        for block in blocks:
            if block.type == 'header':
                x1, y1, x2, y2 = block.bbox
                y1_px = y1 * img_height
                y2_px = y2 * img_height
                height_px = y2_px - y1_px
                
                is_top_position = y1_px < (header_top_threshold * img_height)
                is_small_height = height_px < (header_height_threshold * img_height)
                
                has_similar_content = False
                if block.content:
                    for other_page_idx, other_content in header_contents:
                        if other_page_idx != page_idx:
                            similarity = calculate_similarity(block.content, other_content)
                            if similarity > content_similarity_threshold:
                                has_similar_content = True
                                break
                
                is_likely_header = is_top_position or (is_small_height and has_similar_content)
                
                if is_likely_header:
                    verified_blocks.append(block)
                else:
                    corrected_block = ContentBlock(
                        type='text',
                        bbox=block.bbox,
                        angle=block.angle,
                        content=block.content
                    )
                    verified_blocks.append(corrected_block)
                    header_corrected_count += 1
            else:
                verified_blocks.append(block)
        
        verified_results.append(verified_blocks)
    
    if header_corrected_count > 0:
        logger.info(f"页眉验证完成，共修正 {header_corrected_count} 个header blocks为text")
    else:
        logger.info("页眉验证完成，所有header blocks均通过验证")
    
    return verified_results


def compare_and_merge_layouts(
    dolphin_blocks: List[ContentBlock],
    mineru_blocks: List[ContentBlock],
    mineru_client: MinerUClient = None,
    image: Image.Image = None,
    iou_threshold: float = 0.7,
    type_match_threshold: float = 0.5,
    coverage_threshold: float = 0.5
) -> List[ContentBlock]:
    merged_blocks = []
    used_mineru_indices = set()
    
    for i, dolphin_block in enumerate(dolphin_blocks):
        best_match_idx = -1
        best_iou = 0.0
        
        for j, mineru_block in enumerate(mineru_blocks):
            if j in used_mineru_indices:
                continue
            
            iou = calculate_iou(dolphin_block.bbox, mineru_block.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match_idx = j
        
        if best_match_idx >= 0 and best_iou > iou_threshold:
            mineru_block = mineru_blocks[best_match_idx]
            
            if dolphin_block.type in ['table', 'title', 'image']:
                selected_type = dolphin_block.type
            elif dolphin_block.type == mineru_block.type:
                selected_type = dolphin_block.type
            else:
                types_set = {dolphin_block.type, mineru_block.type}
                if 'image' in types_set:
                    selected_type = 'image'
                elif 'header' in types_set:
                    if dolphin_block.type == 'header':
                        selected_type = mineru_block.type
                    else:
                        selected_type = dolphin_block.type
                elif 'table' in types_set and 'text' in types_set:
                    if mineru_block.type == 'table':
                        selected_type = 'table'
                    else:
                        selected_type = dolphin_block.type
                else:
                    selected_type = dolphin_block.type
            
            dolphin_area = calculate_bbox_area(dolphin_block.bbox)
            mineru_area = calculate_bbox_area(mineru_block.bbox)
            
            if selected_type == 'table':
                best_table_bbox = dolphin_block.bbox
                best_table_area = dolphin_area
                
                if mineru_area > dolphin_area:
                    best_table_bbox = mineru_block.bbox
                    best_table_area = mineru_area
                
                for other_mineru_block in mineru_blocks:
                    if other_mineru_block.type == 'table':
                        other_area = calculate_bbox_area(other_mineru_block.bbox)
                        iou_with_other = calculate_iou(dolphin_block.bbox, other_mineru_block.bbox)
                        if iou_with_other > 0.3 and other_area > best_table_area:
                            best_table_bbox = other_mineru_block.bbox
                            best_table_area = other_area
                
                selected_bbox = best_table_bbox
            elif selected_type == 'header':
                if mineru_area > dolphin_area * 0.8 and mineru_area < dolphin_area * 1.5:
                    selected_bbox = mineru_block.bbox
                else:
                    selected_bbox = dolphin_block.bbox
            else:
                area_ratio = mineru_area / dolphin_area if dolphin_area > 0 else 1.0
                if 0.7 <= area_ratio <= 1.3:
                    selected_bbox = mineru_block.bbox
                else:
                    selected_bbox = dolphin_block.bbox
            
            merged_block = ContentBlock(
                type=selected_type,
                bbox=selected_bbox,
                angle=dolphin_block.angle,
                content=dolphin_block.content
            )
            merged_blocks.append(merged_block)
            used_mineru_indices.add(best_match_idx)
        else:
            merged_blocks.append(dolphin_block)
    
    return merged_blocks


def visualize_layout_pdf(
    images: List[Image.Image],
    blocks_list: List[List[ContentBlock]],
    output_path: str,
    alpha: float = 0.3
) -> None:
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
    merged_results = []
    
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
    
    logger.info("正在使用MinerU VLM进行布局识别...")
    try:
        mineru_results = mineru_client.batch_layout_detect(images=images_pil_list)
    except Exception as e:
        logger.error(f"MinerU VLM布局识别失败: {e}")
        mineru_results = [[] for _ in images_pil_list]
    
    logger.info("正在融合布局识别结果...")
    for page_idx, (dolphin_results, mineru_blocks, image_pil) in enumerate(
        zip(dolphin_results_list, mineru_results, images_pil_list)
    ):
        dolphin_blocks = convert_dolphin_to_content_blocks(dolphin_results, image_pil)
        
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
    images_pil_list = []
    pdf_bytes = None
    images_list = None
    pdf_doc = None
    
    if DOLPHIN_AVAILABLE and not skip_dolphin:
        try:
            logger.info("使用Dolphin的PDF转换方式...")
            images_pil_list = convert_pdf_to_images(input_path, target_size=896)
            if not images_pil_list:
                raise Exception("Dolphin PDF转换失败")
            logger.info(f"Dolphin PDF转换完成，共 {len(images_pil_list)} 页")
            with open(input_path, 'rb') as f:
                pdf_bytes = f.read()
            images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        except Exception as e:
            logger.warning(f"Dolphin PDF转换失败: {e}，使用MinerU的PDF转换方式...")
            with open(input_path, 'rb') as f:
                pdf_bytes = f.read()
            images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
            images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    else:
        logger.info("使用MinerU的PDF转换方式...")
        with open(input_path, 'rb') as f:
            pdf_bytes = f.read()
        images_list, pdf_doc = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        images_pil_list = [image_dict["img_pil"] for image_dict in images_list]
    
    output_file_name = Path(input_path).stem
    file_output_dir = os.path.join(output_dir, output_file_name)
    os.makedirs(file_output_dir, exist_ok=True)
    logger.info(f"正在处理文档: {input_path}，共 {len(images_pil_list)} 页")
    logger.info(f"输出目录: {file_output_dir}")
    
    dolphin_results_list = []
    if skip_dolphin:
        logger.warning("跳过Dolphin布局识别")
        dolphin_results_list = [[] for _ in images_pil_list]
    elif DOLPHIN_AVAILABLE:
        logger.info("正在使用Dolphin进行布局识别...")
        dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
        if dolphin_model is None:
            logger.error("Dolphin模型不可用")
            dolphin_results_list = [[] for _ in images_pil_list]
        else:
            try:
                for idx, image in enumerate(images_pil_list):
                    logger.info(f"  Dolphin识别页面 {idx + 1}/{len(images_pil_list)}...")
                    layout_str = dolphin_model.chat("Parse the reading order of this document.", image)
                    dolphin_results = parse_layout_string(layout_str)
                    
                    if not dolphin_results or not (layout_str.startswith("[") and layout_str.endswith("]")):
                        logger.warning(f"  页面 {idx + 1} Dolphin识别结果无效，使用默认distorted_page")
                        img_width, img_height = image.size
                        dolphin_results = [([0, 0, img_width, img_height], 'distorted_page', [])]
                    
                    dolphin_results_list.append(dolphin_results)
                logger.info(f"Dolphin识别完成，共识别 {sum(len(r) for r in dolphin_results_list)} 个blocks")
            except Exception as e:
                logger.error(f"Dolphin布局识别失败: {e}")
                dolphin_results_list = [[] for _ in images_pil_list]
    else:
        logger.error("Dolphin不可用")
        dolphin_results_list = [[] for _ in images_pil_list]
    
    mineru_results = []
    mineru_client = None
    if skip_mineru:
        logger.info("跳过MinerU VLM布局识别")
        mineru_results = [[] for _ in images_pil_list]
    else:
        logger.info("正在使用MinerU VLM进行布局识别...")
        try:
            mineru_client = ModelSingleton.get_mineru_client(mineru_backend, mineru_model_path)
            mineru_results = mineru_client.batch_layout_detect(images=images_pil_list)
            logger.info(f"MinerU VLM识别完成，共识别 {sum(len(r) for r in mineru_results)} 个blocks")
        except Exception as e:
            logger.warning(f"MinerU VLM布局识别失败: {e}")
            mineru_results = [[] for _ in images_pil_list]
    
    if not dolphin_results_list or all(len(r) == 0 for r in dolphin_results_list):
        if skip_mineru or not mineru_results or all(len(r) == 0 for r in mineru_results):
            logger.error("Dolphin和MinerU VLM都不可用，无法进行布局识别")
            raise RuntimeError("Dolphin和MinerU VLM都不可用，无法进行布局识别")
    
    if skip_mineru:
        logger.info("正在处理布局识别结果（仅使用Dolphin）...")
    else:
        logger.info("正在比对和融合布局识别结果...")
    merged_results = []
    
    for page_idx, (dolphin_results, mineru_blocks, image_pil) in enumerate(
        zip(dolphin_results_list, mineru_results, images_pil_list)
    ):
        dolphin_blocks = convert_dolphin_to_content_blocks(dolphin_results, image_pil)
        
        merged_blocks = compare_and_merge_layouts(
            dolphin_blocks,
            mineru_blocks,
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
    
    output_pdf_path = os.path.join(file_output_dir, f"{output_file_name}_layout.pdf")
    
    logger.info("正在生成布局可视化PDF...")
    visualize_layout_pdf(images_pil_list, merged_results, output_pdf_path)
    
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
    
    if not skip_mineru and mineru_client is not None and pdf_bytes is not None and images_list is not None:
        logger.info("正在使用融合布局进行内容提取和后处理...")
        try:
            extract_results = mineru_client.batch_two_step_extract(
                images=images_pil_list,
                fused_layout_blocks=merged_results
            )
            
            logger.info("正在验证header blocks...")
            extract_results = verify_header_blocks_after_extraction(extract_results, images_pil_list)
            
            logger.info("正在生成MinerU标准格式输出...")
            local_image_dir = os.path.join(file_output_dir, f"{output_file_name}_images")
            local_md_dir = file_output_dir
            os.makedirs(local_image_dir, exist_ok=True)
            
            image_writer = FileBasedDataWriter(local_image_dir)
            middle_json = result_to_middle_json(extract_results, images_list, pdf_doc, image_writer)
            
            middle_json_path = os.path.join(file_output_dir, f"{output_file_name}_middle.json")
            with open(middle_json_path, 'w', encoding='utf-8') as f:
                json.dump(middle_json, f, ensure_ascii=False, indent=2)
            logger.info(f"MinerU中间JSON已保存到: {middle_json_path}")
            
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
            
            logger.info("正在生成content_list.json...")
            pdf_info = middle_json["pdf_info"]
            image_dir = os.path.basename(local_image_dir)
            content_list = vlm_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            content_list_path = os.path.join(file_output_dir, f"{output_file_name}_content_list.json")
            with open(content_list_path, 'w', encoding='utf-8') as f:
                json.dump(content_list, f, ensure_ascii=False, indent=2)
            logger.info(f"MinerU内容列表JSON已保存到: {content_list_path}")
            
            logger.info("正在生成Markdown文件...")
            md_content_str = vlm_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            md_path = os.path.join(file_output_dir, f"{output_file_name}.md")
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content_str)
            logger.info(f"Markdown文件已保存到: {md_path}")
            
            logger.info("内容提取和后处理完成")
        except Exception as e:
            logger.warning(f"内容提取和后处理失败: {e}")
    
    logger.info(f"处理完成: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dolphin和MinerU VLM布局识别结果比对与融合"
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
        help="MinerU VLM后端（默认: transformers）"
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
        help="IoU阈值（默认: 0.7）"
    )
    parser.add_argument(
        "--type-match-threshold",
        type=float,
        default=0.5,
        help="类型匹配阈值（默认: 0.5）"
    )
    parser.add_argument(
        "--skip-dolphin",
        action="store_true",
        default=False,
        help="跳过Dolphin模型加载"
    )
    parser.add_argument(
        "--skip-mineru",
        action="store_true",
        default=False,
        help="跳过MinerU VLM模型加载"
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
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
    
    actual_mineru_model_path = args.mineru_model_path
    if actual_mineru_model_path is None and args.mineru_backend != "http-client":
        if os.path.exists(DEFAULT_MINERU_VLM_PATH):
            actual_mineru_model_path = DEFAULT_MINERU_VLM_PATH
            logger.info(f"使用本地MinerU VLM模型路径: {actual_mineru_model_path}")
        else:
            logger.info("未找到本地MinerU VLM模型，检查 HuggingFace 缓存或自动下载...")
            os.makedirs(DEFAULT_MODEL_ROOT, exist_ok=True)
            try:
                downloaded_path = auto_download_and_get_model_root_path(DEFAULT_MODEL_ROOT, "vlm")
                actual_mineru_model_path = downloaded_path
                logger.info(f"MinerU VLM模型路径: {actual_mineru_model_path}")
            except Exception as e:
                logger.error(f"MinerU VLM模型检查/下载失败: {e}")
                raise
    
    logger.info("正在预加载模型...")
    
    dolphin_model = None
    if args.skip_dolphin:
        logger.warning("跳过Dolphin模型加载")
    elif DOLPHIN_AVAILABLE:
        dolphin_model_path = args.dolphin_model_path
        if not os.path.exists(dolphin_model_path):
            logger.error(f"Dolphin模型路径不存在: {dolphin_model_path}")
            ModelSingleton._dolphin_load_failed = True
        else:
            try:
                dolphin_model = ModelSingleton.get_dolphin_model(dolphin_model_path)
                if dolphin_model is not None:
                    logger.info("Dolphin模型预加载完成")
            except Exception as e:
                logger.error(f"Dolphin模型预加载失败: {e}")
    else:
        logger.error("Dolphin不可用")
    
    if args.skip_mineru:
        logger.info("跳过MinerU VLM模型加载")
    else:
        try:
            ModelSingleton.get_mineru_client(args.mineru_backend, actual_mineru_model_path)
            logger.info("MinerU VLM模型预加载完成")
        except Exception as e:
            logger.warning(f"MinerU VLM模型预加载失败: {e}")
    
    logger.info("模型预加载完成，开始处理文档...")
    
    for input_file in input_files:
        try:
            process_document(
                str(input_file),
                args.output,
                args.dolphin_model_path,
                args.mineru_backend,
                actual_mineru_model_path,
                args.iou_threshold,
                args.type_match_threshold,
                args.skip_dolphin,
                args.skip_mineru
            )
        except Exception as e:
            logger.error(f"处理文件 {input_file} 时出错: {e}")
            continue
    
    logger.info("所有文件处理完成")


if __name__ == "__main__":
    main()