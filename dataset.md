1. [DocSynth300K数据集](https://link.zhihu.com/?target=https%3A//opendatalab.com/zzy8782180/DocSynth300K)
[在 Huggingface的预训练模型](https://link.zhihu.com/?target=https%3A//huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain)
来源：[DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
DocLayout-YOLO项目用Mesh-candidate Bestfit算法合成的数据集。合成的数据集在风格上多样且视觉真实度高。布局方面，涵盖了单栏、双栏以及多栏混合文档；在文档风格上，包括论文、报纸、杂志等多种类型的页面。
解决：
   1. 当前的布局检测数据集类型较为单一，多数集中于论文文档，例如PubLayNet和DocBank。
   2. 其他类型的文档数据集（如DocLayNet、D4LA、M6Doc）数据量较小，仅适用于下游任务的微调和测试，而不适合用于预训练。

    DocLayout-YOLO使用两大公共数据集 D4LA 和 DocLayNet，并引入了一个复杂且具有挑战性的基准测试集DocStructBench，包含学术、教材、市场分析和财务四个类别的文档，用于验证不同文档类型上的性能。其中，精度指标采用 COCO 风格的 mAP，速度指标为每秒处理的图片数（FPS）。
1. [OmniDocBench](https://opendatalab.com/OpenDataLab/OmniDocBench)
MinerU2.5使用的测评数据集,也是opendatalab的，可用以评估文档解析效果。
https://github.com/opendatalab/OmniDocBench?tab=readme-ov-file
测评运行代码
    ```
    export CUDA_VISIBLE_DEVICES=1
    date "+%Y-%m-%d %H:%M:%S %Z%z"  
    python evaluate_mineru_fused_layout.py     --dataset-dir /home/hsr/OmniDocBench/OmniDocBench     --dolphin-model-path /home/hsr/Dolphin/hf_model
    date "+%Y-%m-%d %H:%M:%S %Z%z"
    ```

1. [M6Doc](https://github.com/HCIILAB/M6Doc)
   * 多格式：包含扫描、拍摄和 PDF 文档。
   * 多类型：涵盖科学文章、教科书、书籍、试卷、杂志、报纸和笔记等七种文档类型。
   * 多布局：包含矩形、曼哈顿、非曼哈顿和多列曼哈顿等四种布局。
   * 多语言：包含中文和英文文档。
   * 多标注类别：包含 74 种标注类别，共 237,116 个标注实例，分布在 9,080 页手动标注的文档中。

2. [LogicsParsingBench]()该测试集未发布
    LogicsParsingBench 测试集：1,078 个真实的PDF页面，覆盖了论文、报纸、书籍、海报、简历、试卷等9大类、超过20个子类

3. [Fox-Page基准测试](https://pan.baidu.com/share/init?surl=t746ULp6iU5bUraVrPlMSw&pwd=fox1)
https://huggingface.co/datasets/ucaslcl/Fox_benchmark_data/tree/main
https://github.com/ucaslcl/Fox

1. [OCRFlux-bench-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-cross)
[OCRFlux-pubtabnet-cross](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-cross)
[OCRFlux-bench-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-bench-single)
[OCRFlux-pubtabnet-single](https://huggingface.co/datasets/ChatDOC/OCRFlux-pubtabnet-single)
**OCRFlux-bench-cross**：包含1000个样本（500个英文样本和500个中文样本），每个样本包含连续两页的Markdown元素列表，以及需要合并的元素索引（通过多轮审核手动标记）。如果没有表格或段落需要合并，注释数据中的索引保持空。
**OCRFlux-pubtabnet-cross**：包含9064对分割表片段及其对应的真实合并版本。
作用：帮助衡量OCR系统在跨页表格/段落检测和合并任务中的表现
**OCRFlux-bench-single** 是一个基于2000个PDF页面及其基于真实的Markdown数据的基准测试，这些数据从我们的私人文档数据集中抽样，这些数据集通过多轮手动标记并进行检查。
**OCRFlux-pubtabnet-single** 是一个基于 9064 张表格图像及其对应的真实 HTML 的基准测试，这些图像基于公开的 PubTabNet 基准测试，经过一些格式转换。
作用：该数据集可用于衡量OCR系统在单页解析中的表现。

1. DocLayout-YOLO项目用来验证的数据集
   [D4LA](https://huggingface.co/datasets/juliozhao/doclayout-yolo-D4LA/tree/main)
   [DocLayNet](https://huggingface.co/datasets/juliozhao/doclayout-yolo-DocLayNet/tree/main)
