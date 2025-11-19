# MultimodalRAG 多模态检索增强生成系统
通过集成外部文档知识(包括文本和图像)来增强大型语言模型(LLM)的问答能力, 从而提供基于准确、可追溯事实的回答。模块化设计，多个组件通过接口交互实现预期功能。
经典的管道式(Pipeline)架构
### 相关名词概念的学习
### CLIP 模型
Contrastive Language-Image Pre-Training
对比学习，预测文本是否与图像匹配。
学习预测事物是否属于同一类或不属于同一类的策略通常被称为“对比学习” (contrastive Learning)
### 输入与输出
1. 输入：结构化的文档对象列表 
 数据源 (Data Source): 包含原始文档数据(JSON格式的元数据, 如原始ID、文本描述)和相关的图像文件.
 数据加载与关联 (Data Loading & Association): 负责从数据源读取元数据, 并在文件系统中查找并关联对应的图像文件路径. 输出结构化的文档对象列表. 
1. 输出：信息组织成结构化的Prompt, 发送给外部 大型语言模型 (LLM - ZhipuAI API)。由LLM生成最后的自然语言回答。




## 复现
```
conda deactivate    # 退出当前环境
```
```
git clone https://github.com/singularguy/MultimodalRAG.git
cd MultimodalRAG
conda create -n multimodal_rag python=3.12 -y
conda activate multimodal_rag
pip install -r requirement.txt                              #依旧是等待环境配置，好吧这个挺快的
```
根目录下创建一个 .env 文件 ，内容为`ZHIPUAI_API_KEY=your_api_key`
根目录下创建一个 data.json文件，内容示例
```
[
  {
    "name": "Bandgap1",
    "description": "一个基础的带隙基准电路图，展示 BJT 晶体管和电阻，用于生成温度不敏感的参考电压。"
  },
  {
    "name": "PTAT_Current",
    "description": "该原理图展示如何使用两个不匹配的 BJT 生成与绝对温度成正比（PTAT）的电流。"
  }
]
```
图片放在根目录下的images文件
```
export HF_ENDPOINT=https://hf-mirror.com    # 用镜像来调用hugging face上的CLIP模型
python MultimodalRAG.py             # 运行
```

