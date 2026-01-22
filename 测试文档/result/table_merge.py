# Copyright (c) Opendatalab. All rights reserved.

from loguru import logger
from bs4 import BeautifulSoup
import re
from typing import List, Dict, Tuple, Optional, Set

from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from mineru.utils.enum_class import BlockType, SplitFlag


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters"""
    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return ''.join(result)


def normalize_text(text: str) -> str:
    """标准化文本：移除空白字符，转换为半角"""
    if not text:
        return ""
    text = full_to_half(text)
    # 移除多余空白字符，保留单个空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def calculate_table_total_columns(soup) -> int:
    """计算表格的总列数，考虑合并单元格"""
    rows = soup.find_all("tr")
    if not rows:
        return 0
    
    # 使用矩阵跟踪单元格占用
    max_cols = 0
    occupied = {}
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        cells = row.find_all(["td", "th"])
        
        if row_idx not in occupied:
            occupied[row_idx] = {}
        
        for cell in cells:
            # 找到第一个可用列
            while col_idx in occupied[row_idx]:
                col_idx += 1
            
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            
            # 标记被占用的位置
            for r in range(row_idx, row_idx + rowspan):
                if r not in occupied:
                    occupied[r] = {}
                for c in range(col_idx, col_idx + colspan):
                    occupied[r][c] = True
            
            col_idx += colspan
            max_cols = max(max_cols, col_idx)
    
    return max_cols


def get_table_structure(soup) -> Dict:
    """获取表格的详细结构信息"""
    rows = soup.find_all("tr")
    if not rows:
        return {}
    
    info = {
        "total_rows": len(rows),
        "total_columns": calculate_table_total_columns(soup),
        "has_merged_cells": False,
        "column_patterns": [],
        "row_heights": [],  # 每行的列数
        "cell_span_info": [],  # 单元格合并信息
        "header_rows": 0,
        "data_rows": 0,
    }
    
    # 分析每行结构
    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        row_info = {
            "visual_cells": len(cells),
            "actual_cols": 0,
            "cells": []
        }
        
        for cell in cells:
            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
            text = normalize_text(cell.get_text())
            
            cell_info = {
                "colspan": colspan,
                "rowspan": rowspan,
                "text": text,
                "is_merged": colspan > 1 or rowspan > 1
            }
            
            row_info["cells"].append(cell_info)
            row_info["actual_cols"] += colspan
            
            if colspan > 1 or rowspan > 1:
                info["has_merged_cells"] = True
        
        info["row_heights"].append(row_info["actual_cols"])
        info["cell_span_info"].append(row_info)
    
    # 检测表头
    info["header_rows"] = detect_header_rows(info["cell_span_info"])
    info["data_rows"] = info["total_rows"] - info["header_rows"]
    
    return info


def detect_header_rows(cell_span_info: List[Dict]) -> int:
    """检测表头行数"""
    if not cell_span_info:
        return 0
    
    # 表头通常在前几行，具有一些特征
    header_rows = 0
    
    # 检查前3行
    for i in range(min(3, len(cell_span_info))):
        row_info = cell_span_info[i]
        cells = row_info["cells"]
        
        # 表头特征
        header_features = 0
        for cell in cells:
            text = cell["text"]
            
            # 特征1：短文本（通常表头较短）
            if text and len(text) < 20:
                header_features += 1
            
            # 特征2：包含常见表头关键词
            header_keywords = ["序号", "编号", "名称", "单位", "数量", "金额", "日期", 
                             "No", "ID", "Name", "Date", "Time", "Quantity", "Price"]
            if any(keyword in text for keyword in header_keywords):
                header_features += 2
            
            # 特征3：单元格合并较多（表头常合并单元格）
            if cell["colspan"] > 1 or cell["rowspan"] > 1:
                header_features += 1
        
        # 如果特征足够多，认为是表头
        if header_features >= len(cells):
            header_rows += 1
        else:
            break
    
    return header_rows


def calculate_structure_similarity(struct1: Dict, struct2: Dict) -> float:
    """计算两个表格的结构相似度"""
    if not struct1 or not struct2:
        return 0.0
    
    similarities = []
    weights = []
    
    # 1. 列数相似度（权重0.4）
    if struct1["total_columns"] > 0 and struct2["total_columns"] > 0:
        col_sim = 1 - abs(struct1["total_columns"] - struct2["total_columns"]) / max(
            struct1["total_columns"], struct2["total_columns"], 1
        )
        similarities.append(col_sim * 0.4)
        weights.append(0.4)
    
    # 2. 表头行数相似度（权重0.2）
    header_sim = 1 if struct1["header_rows"] == struct2["header_rows"] else 0.5
    similarities.append(header_sim * 0.2)
    weights.append(0.2)
    
    # 3. 行数比例相似度（权重0.2）
    if struct1["total_rows"] > 0 and struct2["total_rows"] > 0:
        row_sim = min(struct1["total_rows"], struct2["total_rows"]) / max(
            struct1["total_rows"], struct2["total_rows"], 1
        )
        similarities.append(row_sim * 0.2)
        weights.append(0.2)
    
    # 4. 合并单元格特征相似度（权重0.2）
    merge_sim = 1 if struct1["has_merged_cells"] == struct2["has_merged_cells"] else 0.5
    similarities.append(merge_sim * 0.2)
    weights.append(0.2)
    
    # 计算加权平均
    if weights:
        total_weight = sum(weights)
        return sum(similarities) / total_weight if total_weight > 0 else 0.0
    return 0.0


def compare_table_headers(soup1, soup2, max_header_rows=3) -> Tuple[int, bool, List]:
    """比较两个表格的表头，返回匹配信息"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    
    if not rows1 or not rows2:
        return 0, False, []
    
    header_rows = 0
    headers_match = True
    header_texts = []
    
    # 检查前几行是否匹配
    for i in range(min(len(rows1), len(rows2), max_header_rows)):
        cells1 = rows1[i].find_all(["td", "th"])
        cells2 = rows2[i].find_all(["td", "th"])
        
        # 如果单元格数量相差太大，可能不是同一表格
        if abs(len(cells1) - len(cells2)) > 2:
            break
        
        # 计算行相似度
        row_similarity = calculate_row_similarity(cells1, cells2)
        
        if row_similarity >= 0.7:  # 相似度阈值
            header_rows += 1
            row_texts = [normalize_text(cell.get_text()) for cell in cells1]
            header_texts.append(row_texts)
        else:
            # 如果已经匹配了至少一行，可以停止
            if header_rows > 0:
                break
            else:
                headers_match = False
                break
    
    # 如果没匹配到表头，但表格可能没有明显的表头
    if header_rows == 0:
        # 检查是否可能是没有表头的表格
        headers_match = check_if_headerless_tables(rows1, rows2)
    
    return header_rows, headers_match, header_texts


def calculate_row_similarity(cells1, cells2) -> float:
    """计算两行单元格的相似度"""
    if not cells1 or not cells2:
        return 0.0
    
    # 使用较短的单元格列表作为基准
    min_len = min(len(cells1), len(cells2))
    max_len = max(len(cells1), len(cells2))
    
    similarities = []
    for i in range(min_len):
        text1 = normalize_text(cells1[i].get_text())
        text2 = normalize_text(cells2[i].get_text())
        
        # 计算文本相似度
        if text1 and text2:
            if text1 == text2:
                similarities.append(1.0)
            else:
                # 简单相似度计算：公共字符比例
                common_chars = len(set(text1) & set(text2))
                total_chars = len(set(text1) | set(text2))
                if total_chars > 0:
                    similarities.append(common_chars / total_chars)
                else:
                    similarities.append(0.0)
        elif not text1 and not text2:
            similarities.append(1.0)  # 都为空
        else:
            similarities.append(0.0)
    
    # 考虑长度差异
    length_penalty = min_len / max_len if max_len > 0 else 0
    
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity * length_penalty
    
    return 0.0


def check_if_headerless_tables(rows1, rows2) -> bool:
    """检查是否可能是没有表头的表格"""
    if len(rows1) < 2 or len(rows2) < 2:
        return True
    
    # 检查第一行的内容特征
    cells1 = rows1[0].find_all(["td", "th"])
    cells2 = rows2[0].find_all(["td", "th"])
    
    # 如果没有明显的表头特征（如短文本、关键词等），可能没有表头
    has_header_features1 = any(check_header_cell(cell) for cell in cells1)
    has_header_features2 = any(check_header_cell(cell) for cell in cells2)
    
    return not (has_header_features1 or has_header_features2)


def check_header_cell(cell) -> bool:
    """检查单元格是否具有表头特征"""
    text = normalize_text(cell.get_text())
    
    if not text:
        return False
    
    # 表头特征
    header_keywords = ["序号", "编号", "名称", "单位", "数量", "金额", "日期", 
                     "No", "ID", "Name", "Date", "Time", "Quantity", "Price"]
    
    # 短文本或包含关键词
    return len(text) < 15 or any(keyword in text for keyword in header_keywords)


def check_content_continuity(soup1, soup2, header_rows=0) -> bool:
    """检查表格内容是否连续"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    
    if not rows1 or not rows2:
        return False
    
    # 获取第一个表格的最后几行数据
    data_rows1 = []
    for i in range(len(rows1) - 1, max(-1, len(rows1) - 4), -1):
        cells = rows1[i].find_all(["td", "th"])
        if cells and any(normalize_text(cell.get_text()) for cell in cells):
            data_rows1.insert(0, rows1[i])
            if len(data_rows1) >= 2:
                break
    
    # 获取第二个表格的前几行数据
    data_rows2 = []
    for i in range(header_rows, min(len(rows2), header_rows + 3)):
        cells = rows2[i].find_all(["td", "th"])
        if cells and any(normalize_text(cell.get_text()) for cell in cells):
            data_rows2.append(rows2[i])
            if len(data_rows2) >= 2:
                break
    
    if not data_rows1 or not data_rows2:
        return False
    
    # 检查连续性特征
    continuity_features = 0
    
    # 1. 检查数字序列连续性
    for row1 in data_rows1[-2:]:  # 检查最后两行
        cells1 = row1.find_all(["td", "th"])
        for row2 in data_rows2[:2]:  # 检查前两行
            cells2 = row2.find_all(["td", "th"])
            
            for cell1, cell2 in zip(cells1[:3], cells2[:3]):  # 只检查前三列
                text1 = normalize_text(cell1.get_text())
                text2 = normalize_text(cell2.get_text())
                
                # 尝试解析为数字
                try:
                    num1 = float(text1.replace(',', ''))
                    num2 = float(text2.replace(',', ''))
                    if abs(num2 - num1) <= 2:  # 允许小范围跳跃
                        continuity_features += 1
                except:
                    pass
    
    # 2. 检查文本相似性
    last_cells = data_rows1[-1].find_all(["td", "th"])
    first_cells = data_rows2[0].find_all(["td", "th"])
    
    for i in range(min(len(last_cells), len(first_cells), 3)):
        text1 = normalize_text(last_cells[i].get_text())
        text2 = normalize_text(first_cells[i].get_text())
        
        if text1 and text2:
            if text1 == text2:
                continuity_features += 1
            elif len(set(text1) & set(text2)) / max(len(set(text1) | set(text2)), 1) > 0.5:
                continuity_features += 0.5
    
    return continuity_features >= 1.5


def can_merge_tables(current_table_block, previous_table_block, threshold=0.6) -> Tuple[bool, Optional[BeautifulSoup], Optional[BeautifulSoup], Optional[str], Optional[str]]:
    """判断两个表格是否可以合并"""
    # 检查是否有明确的续接标记
    caption_blocks = [block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION]
    has_continuation_mark = False
    if caption_blocks:
        for block in caption_blocks:
            text = full_to_half(merge_para_with_text(block).strip())
            continuation_keywords = ["(续)", "continued", "cont.", "续表", "接上表"]
            if any(keyword in text.lower() for keyword in continuation_keywords):
                has_continuation_mark = True
                logger.debug(f"检测到续接标记: {text}")
                break
    
    # 如果有续接标记，放宽合并条件
    if has_continuation_mark:
        threshold = 0.4  # 降低阈值
    
    # 获取表格HTML内容
    current_html = ""
    previous_html = ""
    
    for block in current_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            current_html = block["lines"][0]["spans"][0].get("html", "")
    
    for block in previous_table_block["blocks"]:
        if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
            previous_html = block["lines"][0]["spans"][0].get("html", "")
    
    if not current_html or not previous_html:
        return False, None, None, None, None
    
    # 检查表格宽度差异
    x0_t1, y0_t1, x1_t1, y1_t1 = current_table_block["bbox"]
    x0_t2, y0_t2, x1_t2, y1_t2 = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2
    
    width_ratio = abs(table1_width - table2_width) / min(table1_width, table2_width) if min(table1_width, table2_width) > 0 else 1
    if width_ratio > 0.2 and not has_continuation_mark:  # 有续接标记时放宽宽度限制
        logger.debug(f"表格宽度差异太大: {width_ratio:.2f}")
        return False, None, None, None, None
    
    # 解析HTML
    soup1 = BeautifulSoup(previous_html, "html.parser")
    soup2 = BeautifulSoup(current_html, "html.parser")
    
    # 获取表格结构信息
    struct1 = get_table_structure(soup1)
    struct2 = get_table_structure(soup2)
    
    # 计算结构相似度
    structure_sim = calculate_structure_similarity(struct1, struct2)
    
    # 比较表头
    header_rows, headers_match, header_texts = compare_table_headers(soup1, soup2)
    
    # 检查内容连续性
    continuity = check_content_continuity(soup1, soup2, header_rows)
    
    # 综合判断
    merge_score = 0.0
    reasons = []
    
    # 1. 续接标记（权重最高）
    if has_continuation_mark:
        merge_score += 0.4
        reasons.append("有续接标记")
    
    # 2. 结构相似度
    if structure_sim >= 0.7:
        merge_score += 0.3
        reasons.append(f"结构相似度高({structure_sim:.2f})")
    elif structure_sim >= 0.5:
        merge_score += 0.2
        reasons.append(f"结构相似度中等({structure_sim:.2f})")
    
    # 3. 表头匹配
    if headers_match:
        merge_score += 0.2
        reasons.append("表头匹配")
    
    # 4. 内容连续性
    if continuity:
        merge_score += 0.3
        reasons.append("内容连续")
    
    # 5. 列数相近
    if struct1["total_columns"] > 0 and struct2["total_columns"] > 0:
        col_diff = abs(struct1["total_columns"] - struct2["total_columns"])
        if col_diff <= 2:  # 允许列数相差不超过2列
            merge_score += 0.2
            reasons.append(f"列数相近({struct1['total_columns']} vs {struct2['total_columns']})")
    
    can_merge = merge_score >= threshold
    
    logger.debug(f"表格合并评分: {merge_score:.2f} (阈值: {threshold}), 原因: {', '.join(reasons)}")
    
    if can_merge:
        return True, soup1, soup2, current_html, previous_html
    else:
        return False, None, None, None, None


def align_table_columns(soup1, soup2, header_rows=0) -> Tuple[BeautifulSoup, BeautifulSoup]:
    """对齐两个表格的列数"""
    struct1 = get_table_structure(soup1)
    struct2 = get_table_structure(soup2)
    
    cols1 = struct1["total_columns"]
    cols2 = struct2["total_columns"]
    
    if cols1 == cols2:
        return soup1, soup2
    
    logger.debug(f"对齐表格列数: {cols1} -> {cols2}")
    
    # 以列数多的表格为基准
    if cols1 > cols2:
        # 调整表2的列数以匹配表1
        soup2 = adjust_table_to_target_columns(soup2, cols1, header_rows)
    else:
        # 调整表1的列数以匹配表2
        soup1 = adjust_table_to_target_columns(soup1, cols2, header_rows)
    
    return soup1, soup2


def adjust_table_to_target_columns(soup, target_columns: int, header_rows: int) -> BeautifulSoup:
    """调整表格到目标列数"""
    rows = soup.find_all("tr")
    
    for row_idx, row in enumerate(rows):
        # 跳过表头行
        if row_idx < header_rows:
            continue
        
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        
        # 计算当前行的总列数
        current_cols = sum(int(cell.get("colspan", 1)) for cell in cells)
        
        if current_cols == target_columns:
            continue
        
        if current_cols < target_columns:
            # 需要增加列数：扩展最后一个单元格
            diff = target_columns - current_cols
            last_cell = cells[-1]
            current_colspan = int(last_cell.get("colspan", 1))
            last_cell["colspan"] = str(current_colspan + diff)
        
        elif current_cols > target_columns:
            # 需要减少列数：这比较复杂，通常不应该发生
            # 如果有合并单元格，可以尝试拆分
            logger.warning(f"当前行列数({current_cols})超过目标列数({target_columns})")
            # 暂时不处理，保持原样
    
    return soup


def perform_table_merge(soup1, soup2, previous_table_block, wait_merge_table_footnotes) -> str:
    """执行表格合并操作"""
    # 检测表头
    header_rows, headers_match, header_texts = compare_table_headers(soup1, soup2)
    logger.debug(f"表头检测结果: 行数={header_rows}, 匹配={headers_match}")
    
    # 对齐列数
    soup1, soup2 = align_table_columns(soup1, soup2, header_rows)
    
    # 获取表格行
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    
    # 合并表格内容
    tbody1 = soup1.find("tbody") or soup1.find("table")
    tbody2 = soup2.find("tbody") or soup2.find("table")
    
    if tbody1 and tbody2:
        # 将第二个表格的行添加到第一个表格中（跳过表头行）
        for row in rows2[header_rows:]:
            row_copy = BeautifulSoup(str(row), "html.parser").find("tr")
            tbody1.append(row_copy)
    
    # 添加footnote
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)
    
    return str(soup1)


def merge_table(page_info_list):
    """合并跨页表格"""
    for page_idx in range(len(page_info_list) - 1, 0, -1):
        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]
        
        # 检查当前页是否有表格块
        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue
        
        current_table_block = page_info["para_blocks"][0]
        
        # 检查上一页是否有表格块
        if not (previous_page_info["para_blocks"] and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE):
            continue
        
        previous_table_block = previous_page_info["para_blocks"][-1]
        
        # 收集待合并表格的footnote
        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"]
            if block["type"] == BlockType.TABLE_FOOTNOTE
        ]
        
        # 检查两个表格是否可以合并
        can_merge, soup1, soup2, current_html, previous_html = can_merge_tables(
            current_table_block, previous_table_block
        )
        
        if not can_merge:
            continue
        
        # 执行表格合并
        merged_html = perform_table_merge(
            soup1, soup2, previous_table_block, wait_merge_table_footnotes
        )
        
        # 更新previous_table_block的html
        for block in previous_table_block["blocks"]:
            if (block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]):
                block["lines"][0]["spans"][0]["html"] = merged_html
                break
        
        # 标记当前页表格已删除
        for block in current_table_block["blocks"]:
            block['lines'] = []
            block[SplitFlag.LINES_DELETED] = True
        
        logger.info(f"已合并表格: 第{page_idx}页表格合并到第{page_idx-1}页")


# 保持原有的辅助函数
def calculate_row_columns(row):
    """计算行的实际列数（考虑colspan）"""
    cells = row.find_all(["td", "th"])
    return sum(int(cell.get("colspan", 1)) for cell in cells)


def calculate_visual_columns(row):
    """计算行的视觉列数（单元格数量）"""
    cells = row.find_all(["td", "th"])
    return len(cells)


def check_rows_match(soup1, soup2):
    """检查表格行是否匹配（兼容性函数）"""
    rows1 = soup1.find_all("tr")
    rows2 = soup2.find_all("tr")
    
    if not (rows1 and rows2):
        return False
    
    # 获取第一个表的最后一行
    last_row = rows1[-1] if rows1 else None
    
    # 获取第二个表的第一行
    first_row = rows2[0] if rows2 else None
    
    if not (last_row and first_row):
        return False
    
    last_row_cols = calculate_row_columns(last_row)
    first_row_cols = calculate_row_columns(first_row)
    
    return last_row_cols == first_row_cols