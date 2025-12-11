# MonkeyOCR v1.5 Technical Report: Unlocking Robust Document Parsing for Complex Patterns
## MonkeyOCR v1.5技术报告：解锁复杂模式的稳健文档解析
https://github.com/Yuliang-Liu/MonkeyOCR
## Introduction
1. 文档解析支撑着信息提取、检索增强生成和智能文档分析等下游应用，其将扫描图像、PDF文档等各类文档中包含的文本、表格、图像、公式等复杂多模态内容，系统化转化为结构化表示。
2. 具有高度复杂版式和精细表格结构的文档图像难以解析，如表格可能包含多级嵌套、跨页延伸、合并或拆分单元格，以及图像、公式、混合字体等嵌入元素。
3. 不规则版式、多语言环境和多样化的排版风格，对鲁棒性强且可泛化的解析模型的进一步需求。
4. 现有办法的问题：
   1. 传统的基于管道的方法，将文档解析分解为一系列子任务，这种多阶段过程容易导致错误累积。
   2. 端到端模型，在单次处理中完成整个文档图像的解析，文档图像的高分辨率产生了大量视觉标记，自注意力机制的二次复杂度造成了显著的计算瓶颈。
   3. MonkeyOCR 提出了 SRR 范式，将文档解析解耦为结构检测、内容识别和阅读顺序预测。这种设计简化了传统的多阶段管道，有效缓解了累积错误，同时避免了全页端到端处理带来的巨大计算开销，从而推进了智能多模态文档理解。
   4. Mineru 2.5[18]进一步简化了三阶段框架，采用统一的大规模多模态模型联合预测文档布局和阅读顺序，随后进行内容识别。
   5. PPOCR -VL [5] 采用类似的三阶段方法，利用轻量级模型进行结构分析和阅读顺序预测，随后应用大型多模态模型进行内容识别
5. MonkeyOCR v1.5 两阶段解析策略：第一阶段进行结构检测与关系预测，第二阶段完成内容识别。该设计不仅优化了传统MonkeyOCR流程，还通过融合视觉-语义信息，显著提升了模型在复杂排版布局中的文本序列判定能力。针对复杂表格识别问题，我们创新性地提出基于视觉一致性强化学习算法，通过对比原始图像与渲染版本的视觉一致性来评估识别结果的准确性。
```
conda deactivate
conda create -n MonkeyOCR python=3.10
conda activate MonkeyOCR
export CUDA_VERSION=126 # for CUDA 12.6
# export CUDA_VERSION=118 # for CUDA 11.8

pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/
pip install langchain==0.3.26
pip install "paddlex[base]==3.1.4"

# Install PyTorch. Refer to https://pytorch.org/get-started/previous-versions/ for version compatibility
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
pip install -e .
# CUDA 12.6
pip install lmdeploy==0.9.2
pip install accelerate
```


MonkeyOCR v1.5
