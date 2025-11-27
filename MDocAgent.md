## MDocAgent: A Multi-Modal Multi-Agent Framework for Document Understanding
## MDocAgent:一种用于文档理解的多模态多智能体框架
### Introduction  
#### 研究背景
1. Answering questions based on reference documents(DocQA)在信息检索和自动化文档分析等方向有重要应用，核心需求是基于文档的文本和视觉信息精准回答问题。
2. LLMs只能处理文本信息，LVLMs则常用于DocQA。LVLMs在处理视觉内容时表现优异，但在处理长文档时表现不佳，缺乏对细节的细致关注，同时在跨模态理解方面也存在局限性。单模态上下文检索可以处理长文档，但仍然面临详细分析或跨模态信息整合的问题。
   ![RAG](./pictures/RAG1.png)
3. 文档中通常存在大量信息。直接处理整个文档会消耗大量计算资源，导致模型难以识别最关键的信息。
4. RAG作为辅助工具被引入来提取长文档中关键字解决以上问题，但其仅在分别处理文本和视觉信息时有效，当同时处理两者时，会有些力不从心。
#### 挑战
DocQA中问题的多样性和涵盖信息的复杂性。
1. LVLMs则常用于DocQA。LVLMs在处理视觉内容时表现优异，但在处理长文档时表现不佳，缺乏对细节的细致关注，同时在跨模态理解方面也存在局限性。
2. 文档包含海量文本和视觉元素，直接处理易导致计算资源浪费，且难以定位关键信息。
3. 现有RAG缺乏跨模态信息整合能力。这种无法有效整合多模态信息的缺陷，制约了当前基于RAG的方法在复杂文档问答任务中的表现。
   
#### 框架概览
MDocAgent该框架融合了检索-生成（RAG）技术与协作式多智能体系统，通过专业智能体协同处理文本与图像信息。系统采用双线程RAG架构：文本RAG负责基础信息提取，图像RAG专注视觉内容分析。整个框架包含五大智能体：通用智能体负责多模态预处理，关键智能体精准识别核心信息，文本智能体进行文本解析，图像智能体执行模态内深度分析，最终由摘要智能体整合生成完整答案。
运作分为五个阶段：(1) 文档预处理：通过OCR提取文本，并将页面保存为图像。(2) 多模态上下文检索：基于文本和图像的RAG工具分别检索出前k个相关文本片段和图像页面。(3) 初始分析与关键信息提取：通用智能体生成初步响应，关键智能体提取核心信息并传递给专业智能体。(4) 专业智能体处理：文本和图像智能体根据关键信息，分别在其专业领域内分析检索到的上下文。(5) 答案综合：摘要智能体整合所有智能体的响应，生成最终答案。
    ![RAG](./pictures/MDocAgent.jpg)

### Related Work
1. **LVLMs in DocQA Tasks** : LVLMs能将图像中的文本转化为视觉表征，同时保留版式和视觉上下文。然而，这类模型仍面临输入规模限制和可能丢失细粒度文本细节等挑战，因此实现文本与视觉信息的有效整合，对DocVQA的精准性能至关重要
2. **Retrieval-Augmented Generation(RAG)** : 现有的针对LVLMs的图像RAG可以用以检索视觉相关，但因为检索的相互独立进行，无法有效整合和分析文本和图像信息。
3. **Multi-Agent Systems** : 多智能体系统可以通过配置专业智能体分工协作，共同完成单一模型难以应对的任务。然而在DocQA应用中，由于需要整合多种模态信息，系统面临独特挑战。单纯叠加文本与图像智能体的输出，往往难以捕捉二者间的微妙交互——这种交互正是准确理解文档的关键。

### MDocAgent(details)
5个阶段
#### 文档预处理
1. 目标：将文档语料库转换为适合文本与视觉分析的格式
2. 技术：采用光学字符识别（OCR）与PDF解析技术相结合的方式提取文本内容。这种双管齐下的方法确保了对不同文档格式和结构的稳健文本提取。
3. 该框架允许通过整合文本与视觉线索实现全面理解。预处理过程生成文档语料库的双重表征：文本表征由提取的文本片段构成，视觉表征则包含原始页面图像。
#### 多模态上下文检索
1. 目标：在同时兼顾文本和图像信息的基础上，从文档语料库中高效检索最相关的信息。
2. 技术：使用ColBERT进行文本检索，使用ColPali进行图像检索，独立获取基于输入问题的前k个相关文本段和图像页面。
![RAG](./pictures/RAG2.png)
#### 初始分析和关键提取
1. 目标：初步解读问题，并在检索到的上下文中精准定位关键信息。
2. 技术：The general agent AG作为多模态整合器，接收检索到的文本上下文Tq和视觉上下文Iq，并整合两种模态中的信息，对这些多模态输入进行处理，然后初步生成答案aG。
   The critical agent AC对接收问题q、检索到的上下文Tq和Iq，以及通用代理生成的初步答案aG这些输入进行细致分析，筛选出对准确回答问题至关重要的关键信息Tc和Ic。

    |![RAG](./pictures/RAG3.png)|![RAG](./pictures/RAG4.png)|
    |-|-|
    |![RAG](./pictures/RAG5.png)|![RAG](./pictures/RAG6.png)|
#### 专门代理处理
1. 目标：通过运用前一阶段提取的关键信息指导的专业智能体进行分析。
2. 技术：The text agentAT接收检索到的文本片段Tq和关键文本信息Tc作为输入，在其模态里对提供的文本片段进行深度分析，最终生成基于文本的详细答案aT。
   The image agentAI接收检索到的图像Iq和关键视觉信息Ic，针对关键视觉信息Ic所标注的区域或特征进行图像处理，从视觉内容中提取有价值的信息，最终生成的视觉化AI答案aI
#### 答案综合
1. 目标：整合前序各阶段的多元输出，将初始的多模态理解与专业智能体分析相结合，生成全面精准的答案。
2. 技术：The summarizing agent AS分别接收前面agent生成的答案aG、aT和aI，分析各智能体的答案，识别共性特征、差异点及互补洞见，并考量各智能体提供的佐证。通过化解智能体间的潜在冲突或分歧，整合各自优势，构建出最终答案aS。
    ![RAG](./pictures/RAG7.png)

### Experiment
实验验证
1. 相较于现有基于RAG的方法，MDocAgent能否有效提升文档理解准确率？
2. 框架中的每个智能体是否发挥着关键作用？
3. 该方法如何增强模型对文档的理解能力？
#### 实验设置
1. 实施细节：
   1.  text agent： Llama-3.1-8B-Instruct 
   2.  other four agents： Qwen2-VL-7B-Instruct 
   3.  the text and image retrievers：ColBERTv2  and ColPali 
   4.  每个示例提取1个或4个最高分片段作为输入上下文
2. 数据集
    这些评估数据集涵盖多种场景，包括开放域与封闭域、文本与视觉、长文档与短文档，确保评估的公平性和完整性。
    MMLongBench
    LongDocUrl
    PaperTab
    PaperText
    FetaTab
3. 评估指标
在所有基准测试中，采用GPT-4o作为评估模型，参照Deng等人和Ma等人的方法，通过比较模型输出与参考答案的一致性，生成二元判定结果（正确/错误）。
#### 主要研究结果
结果：MDocAgent Outperforms All the Comparison Methods
and Other LVLMs.
    ![RAG](./pictures/MDocAgent_result.png)
1. 在top-1检索方式下， MDocAgent 平均提高了12.1%，比当前最新提出的表现最亮眼的 M3DocRAG 还高。表明 MDocAgent 在应对信息过载、细节精准捕捉和跨模态理解等核心挑战方面，比现有方法更具优势。
    1. 在FetaTab数据集上， MDocAgent 获得0.600的分数，超越 M3DocRAG 21.0%。
    2. 在 PaperText 上， MDocAgent 得分为0.399，超越 M3DocRAG 16.7%。
    3. 与最佳 LVLM（Qwen2.5-VL-7B）和基于文本-RAG（ColBERTv2+Llama-3.1-8B）的基准模型相比，MDocAgent 在所有测试中平均提升51.9%和23.7%。
2. 在top-4检索方式下， MDocAgent在所有基准测试中仍表现最佳，并进一步扩大了领先优势。体现其有效整合多个检索项的能力。
    1. 平均超越Qwen2.5-VL-7B 73.5%，展示了其有效利用额外上下文信息的能力。
    2. 平均超越M3DocRAG 10.9%，同时比ColBERTv2+Llama-3.1- 8B提升了6.9%。
    3. 相比之下，M3DocRAG在前4次检索中表现下降，表明在整合多个检索项方面存在局限性。   
> top-1 retrieval:top-1检索方式，指信息检索系统中返回最相关单个结果的检索模式。
> top-4 retrieval:前4项检索方式，指在信息检索过程中返回相关性最高的前4个结果，用于评估检索系统的性能表现。
> Ablation Studies： 消融研究，用于分析模型中各组件对性能的影响
#### 定量分析
1. 消融实验
![RAG](./pictures/RAG8.png)
通过数据比对，可以看出每个智能体的重要性，去除任何一个智能体，都会导致性能的显著下降。
2. 细粒度性能分析

|![RAG](./pictures/RAG9.png)|![RAG](./pictures/RAG11.png)|
|--|--|

主要看基于MMLongBench数据集的结果。通过数据比对，可以看到
   1. MDocAgent在所有证据模态类型中均优于所有 LVLM 基准模型。
   2. 在top-1检索方式下，MDocAgent虽然在Figure类别中表现比M3DocRAG略差，但在Chart、Table和Text类别中展现出强劲实力，这充分体现了其处理文本与视觉信息的卓越能力。
   3. 在top-4检索方式下，MDocAgent在所有类别中均实现性能提升，尤其在Figure类别中表现突出，彰显了其有效处理多样化信息源的强项。
3. 兼容性分析
![RAG](./pictures/RAG10.png)
该框架的鲁棒性在使用不同图像RAG骨干（ColPali与ColQwen2-v1.0）的兼容性测试中进一步得以证明，两者的平均得分几乎相同（0.407），表明MDocAgent的核心优势在于其多智能体架构本身，而非依赖特定检索模型。

#### 案例研究
|![RAG](./pictures/RAG12.png)|![RAG](./pictures/RAG13.png)|
|--|--|

主要看基于Latino populations案例的结果。可以看到：
    1. 虽然ColBERT和ColPali都能成功定位到包含关键信息的页面，但两种基准方法都未能准确生成答案。其中ColBERT+ Llama-3.1-8B 基准模型仅依赖文本分析，错误地得出“外籍拉丁裔人口更多”的结论，暴露出其对文档数值信息的解读能力不足。同样，M3DocRAG模型因捕获错误信息而无法正确解析问题。
    2. 相比之下，MDocAgent通过专业智能体的协同运作，成功从文本和图像中的表格提取关键信息，最终生成正确答案。这充分说明了细粒度多模态分析的重要性，以及在具体语境中精准处理信息的能力。

### Conclusion
论文提出MDocAgent框架 ，通过专用代理和双RAG方法整合文本与视觉信息，解决现有方法局限性。包含文本处理、图像分析和关键信息提取代理，并由合成代理生成最终答案。实验表明，该框架在DocQA任务中显著优于LVLMs和多模态RAG方法，提升 跨模态理解 与答案准确性。未来工作将探索更先进的代理通信和外部知识源集成。




## 复现
```
conda deactivate    # 退出当前环境
```
```
git clone git@github.com:aiming-lab/MDocAgent.git
cd MDocAgent
conda create -n mdocagent python=3.12
conda activate mdocagent
bash install.sh                     # 这一步又要下载好久嘞
mkdir data
cd data
sudo apt install curl -y
curl -X GET \                       # 下载数据集
     "https://datasets-server.huggingface.co/rows?dataset=Lillianwei%2FMdocagent-dataset&config=default&split=train&offset=0&length=100"
# 发现不行，尝试各种其他方法，最后换了个代理，然后用git clone
git clone git@hf.co:datasets/Lillianwei/Mdocagent-dataset
好吧这样下的数据集有问题，还是得重新下

# 换了个代理，hugging face可以注册了     2025.11.17
# 用huggingface-cli也不行，在服务器上用git也不行，直接先git clone到电脑上，再传到服务器上
 git clone https://huggingface.co/datasets/Lillianwei/Mdocagent-dataset
python scripts/extract.py --config-name feta                            # 发现只有FetaTab数据集，先运行这个
python scripts/extract.py --config-name mmlb                        # 重新下了一下数据集，总算齐了
python scripts/extract.py --config-name ldu 
python scripts/extract.py --config-name ptab 
python scripts/extract.py --config-name ptext                       # 出了一点小问题，部分pdf出现损坏好像？不知道这样影不影响，直接继续了
 
 python scripts/retrieve.py --config-name mmlb              # 这里出现了一点报错，可以看2025-11-19.md的学习日志里的记录
 python scripts/retrieve.py --config-name feta
 python scripts/retrieve.py --config-name ldu
 python scripts/retrieve.py --config-name ptab
 python scripts/retrieve.py --config-name ptext
 python scripts/retrieve.py --config-name mmlb 


# 先下载需要的模型，再运行上面的代码
 # 克隆 ColPali 基础模型
git clone https://huggingface.co/vidore/colpaligemma-3b-mix-448-base ./local_models/colpaligemma-3b-mix-448-base
# 克隆 ColPali 适配器和处理器
git clone https://huggingface.co/vidore/colpali ./local_models/colpali
# 克隆 ColBERT 模型
git clone https://huggingface.co/colbert-ir/colbertv2.0 ./local_models/colbertv2.0

python scripts/predict.py --config-name feta run-name=feta-runs
python scripts/predict.py --config-name ldu run-name=ldu-runs
python scripts/predict.py --config-name ptab run-name=ptab-runs
python scripts/predict.py --config-name ptext run-name=ptext-runs
python scripts/predict.py --config-name mmlb run-name=mmlb-runs
# 缺少Llama3.1-8B-Instruct，可以去Llama官网下载申请，申请随便填一下就过了，比hugging face上申请好过
# 好吧，这样下的不能用，与transform不兼容，只能用hugging face上下载的，不过hugging face我重新申请不了了，国家估计还是得填国外的

python scripts/eval.py --config-name feta run-name=feta-runs
python scripts/eval.py --config-name ldu run-name=ldu-runs
python scripts/eval.py --config-name ptab run-name=ptab-runs
python scripts/eval.py --config-name ptext run-name=ptext-runs
# predict和eval的模型都不好搞，都给换成qwen2vl了
```


## 分析代码时的一些笔记
1. MDocAgent输入处理的主要是pdf文档，
2. 一开始extract用pymupdf（pdf解析技术）先提取pdf中每页的文本，然后将pdf渲染保存为图片。
 **（说是用了OCR技术和pdf解析技术结合，但代码似乎只看到了pdf解析技术的使用？）**
 （json）描述每个文档、每页内容、问题
3. 然后 retrieval 分别检索文字和图片信息。
 检索过程本质是“将内容转为向量，建立索引，输入查询后做相似度搜索，返回最相关内容”
    主流程：
    1. 加载配置：确定是文本还是图片检索。
    2. 动态加载检索类：通过配置文件指定检索类路径（如retrieval.text_retrieval.ColbertRetrieval），运行时动态加载类并初始化。
    3. 初始化数据集和检索模型：用 BaseDataset 加载样本数据
    4. 执行检索：调用检索模型的 find_top_k(dataset) 方法，对每个样本进行检索，找出 top_k 最相关的内容。
   
    文本检索：（基于 ColBERT（语义检索模型）的文本检索）
    用 ColBERT 模型将页面文本转为向量，建立索引，然后用问题做查询，返回最相关页面
    1. 准备阶段：用 RAGPretrainedModel 加载 ColBERT 检索模型。
    遍历所有样本，把每页文本转为向量，建立索引（index）。
    2. 检索阶段:对每个样本，输入查询（如问题文本）。
    用 ColBERT 检索模型在索引中搜索，返回 top_k 最相关的页面编号和分数。（用样本里的问题作为查询，检索所有页面，得到相关性分数。）
    支持过滤，只在指定页面范围内检索。
    3. 结果保存:检索结果（相关页面编号和分数）写入样本 json，供后续模型/Agent使用。
    
    图像检索：
    将图片转为向量，批量缓存，输入查询图片后做相似度搜索，返回最相关页面
   1.  准备阶段:用 ColPali 模型批量提取所有文档每页图片的特征（embedding）存入字典，并用 pickle 保存特征到本地。
    2. 检索阶段:对每个样本，输入查询文本（如问题相关图片）。
    将查询文本编码为向量，与图像嵌入计算相似度（通过CustomEvaluator）。
    返回 top_k 最相关页面编号和分数。
    3. 结果保存:检索结果写入样本 json，供后续模型/Agent使用。
4. Multi-Agent Inference：
批量预测（predict_dataset方法）
   1. 加载样本与断点续跑
   从数据集加载样本，从resume_path加载已处理样本，跳过已生成答案的样本。
   可选截断样本数量（truncate_len），用于快速测试。
   2. 单样本预测
   对每个样本执行：
       1. 提取输入数据：通过dataset.load_sample_retrieval_data(sample)获取样本的question（问题）、texts（相关文本列表）、images（相关图像路径列表）。
       2. 通用代理处理：
       接收question、texts、images，生成初步回答（general_response）。
       通过self_reflect方法和critical_prompt生成结构化关键点（critical_info），包含text（文本关键点）和image（图像关键点）。
      1. 文本 / 图像代理细化：
      文本代理接收question+ 文本关键点，仅基于texts生成细化回答（text_response）。
      图像代理接收question+ 图像关键点，仅基于images生成细化回答（image_response）。
       2. 汇总代理生成最终答案：汇总代理整合所有代理的输出（all_messages），生成结构化最终答案（{"Answer": "..."}）。
   3. 结果保存与内存管理
   每处理save_freq个样本，保存一次中间结果到result_dir（路径含时间戳，如2024-05-20-15-30.json）。
   处理完一个样本后，清理代理对话历史（clean_messages）并释放 GPU 内存（torch.cuda.empty_cache()），避免内存累积。
   4. 最终输出
   所有样本处理完成后，最终结果保存为 JSON 文件，包含每个样本的预测答案（ans_key字段）和可选的对话历史（ans_key+"_message"字段）。
   若启用评估（通过Agent.eval_dataset），会生成包含二进制正确率（binary_correctness）的评估报告。
image_agent 、 general_agent 和 sum_agent使用 qwen2vl 模型
text_agent 使用 llama31 模型
eval_agent 使用 openai 

## 一些需补充的知识
1. pymupdf（pdf解析技术）
2. ColBERT 模型：ColBERT（Contextualized Late Interaction over BERT）是一种基于 BERT 的高效文本检索模型，通过上下文感知的后期交互计算查询与文档的相似度，在保持高精度的同时大幅提升检索速度。它将查询和文档编码为向量，通过向量匹配实现快速检索。
3. ColPali 模型：ColPali 是基于 PaLI-Gemma 的跨模态检索模型，支持文本查询与图像文档的匹配（如图像中的文字、布局等信息），适用于无法提取文本的图像类文档（如扫描件、截图）。
4. 


