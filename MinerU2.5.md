# MinerU
## 一、链接
https://github.com/opendatalab/MinerU
## 二、代码复现
```
conda deactivate
git clone https://github.com/opendatalab/MinerU.git
cd MinerU
conda create -n mineru python=3.12
conda activate mineru
pip install uv
uv pip install -U "mineru[core]" -i https://mirrors.aliyun.com/pypi/simple 
python analyze.py   # config 路径记得修改，这个是师兄直接弄好的情况下的操作，其实也相当于调用了下面那条命令了，只是会根据config文件路径来自动填充命令
mineru -p <input_path> -o <output_path>                         # 这个是自己复现的情况
mineru -p /home/hsr/福大课题数据包/福大课题数据包/基于业务本体的智能文档解 析/带批注的采购需求或采购文件/泉州  -o /home/hsr/MinerU/output
```

## 三、进一步优化思路
1. 这里是否可以考虑将脚注的基于距离判定改成用调用AI判定？
![MinerU](./pictures/M1.png)
这里同理
![MinerU](./pictures/M2.png)
2. 标题分级只有一级，没有二三四级
![MinerU](./pictures/M6.png)
3. 表格识别可以考虑增加一个上下行单元格数量不同时，看几个合并起来会不会和上一行的宽度一样。
表格识别有时会吞掉空单元格啊。
一行表格的右边内容又被分成多行且跨页时合并效果差，但普通有按序号、表格内容没有被中间截断的合并效果还不错。
![MinerU](./pictures/M3.png) 
这个续有点不好说啊
![MinerU](./pictures/M5.png)
1. 这里合并规则？还没看原始代码，先保留疑惑
![MinerU](./pictures/M4.png) 

## 四、学习
### 4.1 涉及的模型学习
#### 4.1.1 pipeline
![minerU](./pictures/M8.png)
##### YOLO学习
1. 相关概念：YOLO（You Only Look Once）是一组实时目标检测机器学习算法。目标检测是一种计算机视觉任务，它使用神经网络来定位和分类图像中的物体。[卷积神经网络(CNN)](https://github.com/luuxiann/RAG-learning/blob/main/CNN.md) 是任何 YOLO 模型的基础。
2. 
   

#### 4.1.2 VLM 端到端
![minerU](./pictures/M9.png)

#### 4.1.3 中间格式转换
##### 后处理阶段
1. 跨页表格合并：BeautifulSoup
2. 阅读顺序排序 (sort_blocks_by_bbox)

    排序方法：
    1. LayoutReader排序（优先）：
        使用LayoutLMv3模型
        支持最多512行
    精度高
    2. XY-Cut排序（备选）：
        当行数 > 200 时使用
        基于递归分割算法
### 4.2 涉及的python库
#### 4.2.1 pypdfium




### 参考学习链接
1. （2025-01-18）   [ YOLO 详解：从 v1 到 v11 ](https://zhuanlan.zhihu.com/p/13491328897)
2.  

## 五、具体代码学习
### 5.2 Pipeline后端详细流程
```
PDF输入
  ↓
【阶段1】PDF分类与预处理
  ├─ PDF类型判断（文本型/图像型）
  ├─ OCR需求判断
  └─ 页面范围裁剪
  ↓
【阶段2】图片提取与批处理
  ├─ 从PDF提取PIL图像
  ├─ 图像预处理
  └─ 批处理组织
  ↓
【阶段3】模型推理流水线
  ├─ Layout检测（DocLayout-YOLO）
  ├─ 公式检测（YOLOv8-MFD）
  ├─ 公式识别（UniMERNet）
  ├─ 表格识别（RapidTable/UnetTable）
  └─ OCR识别（PP-OCR）
  ↓
【阶段4】结果整合与后处理
  ├─ MagicModel处理
  ├─ 坐标转换
  ├─ 置信度过滤
  └─ 重叠处理
  ↓
【阶段5】中间格式转换
  ├─ Span处理
  ├─ 文本提取
  ├─ 图片裁剪
  └─ 区块排序
  ↓
【阶段6】段落合并与排序
  ├─ 段落识别
  ├─ 列表识别
  └─ 跨页合并
  ↓
【阶段7】输出生成
 ```

### 5.3 VLM后端详细流程
```
PDF输入
  ↓
【阶段1】模型初始化与配置
  ├─ 推理引擎选择
  ├─ 模型加载
  └─ 批处理配置
  ↓
【阶段2】PDF图片提取
  ├─ 提取PIL图像
  └─ 图像列表组织
  ↓
【阶段3】两阶段推理           # 代码文件路径：/home/hsr/anaconda3/envs/mineru/lib/python3.12/site-packages/mineru_vl_utils/mineru_client.py
  ├─ 阶段3.1：布局分析
  └─ 阶段3.2：内容识别
  ↓
【阶段4】模型输出解析
  ├─ JSON格式解析
  └─ 区块提取
  ↓
【阶段5】MagicModel处理
  ├─ 坐标转换
  ├─ 类型映射
  ├─ 内容解析
  └─ 区块分类
  ↓
【阶段6】中间格式转换
  ├─ 标题优化
  ├─ 图片裁剪
  └─ 区块排序
  ↓
【阶段7】后处理
  ├─ 跨页表格合并
  └─ LLM标题分级
  ↓
【阶段8】输出生成
```
```
# 推理引擎是执行模型推理的底层库/框架。
# 同一个模型（MinerU2.5-2509-1.2B）可以用不同的引擎运行，差异在于执行方式和性能。

# 基于 Qwen2VLForConditionalGeneration（Qwen2-VL 架构的多模态生成模型）
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,  # "opendatalab/MinerU2.5-2509-1.2B"
    device_map={"": device},
    dtype="auto",
)
```
![minerU](./pictures/M8.png)
 #### 5.3.4 两阶段推理
 ```
batch_two_step_extract(images)
  ↓
根据batching_mode选择模式
  ├─ concurrent模式（异步后端）
  │   └─ 并发执行每个图片的two_step_extract
  │       ├─ layout_detect（布局检测）
  │       ├─ prepare_for_extract（准备提取）
  │       ├─ batch_predict（内容提取）
  │       └─ post_process（后处理）
  │
  └─ stepping模式（同步后端）
      ├─ batch_layout_detect（批量布局检测）
      ├─ batch_prepare_for_extract（批量准备提取）
      ├─ batch_predict（批量内容提取）
      └─ batch_post_process（批量后处理）
```
MinerU的prompt相较之前看的MDocAgent可以说是简洁了很多。
|![minerU](./pictures/M10.png)|![minerU](./pictures/M11.png)|
|-|-|
|![minerU](./pictures/M12.png)|![minerU](./pictures/M13.png)|

