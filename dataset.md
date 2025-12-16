## 布局分析
1. [DocSynth300K数据集](https://link.zhihu.com/?target=https%3A//opendatalab.com/zzy8782180/DocSynth300K)
[在 Huggingface的预训练模型](https://link.zhihu.com/?target=https%3A//huggingface.co/juliozhao/DocLayout-YOLO-DocSynth300K-pretrain)
来源：DocLayout-YOLO
DocLayout-YOLO项目用Mesh-candidate Bestfit算法合成的数据集。合成的数据集在风格上多样且视觉真实度高。布局方面，涵盖了单栏、双栏以及多栏混合文档；在文档风格上，包括论文、报纸、杂志等多种类型的页面。
解决：
   1. 当前的布局检测数据集类型较为单一，多数集中于论文文档，例如PubLayNet和DocBank。
   2. 其他类型的文档数据集（如DocLayNet、D4LA、M6Doc）数据量较小，仅适用于下游任务的微调和测试，而不适合用于预训练。

    DocLayout-YOLO使用两大公共数据集 D4LA 和 DocLayNet，并引入了一个复杂且具有挑战性的基准测试集DocStructBench，包含学术、教材、市场分析和财务四个类别的文档，用于验证不同文档类型上的性能。其中，精度指标采用 COCO 风格的 mAP，速度指标为每秒处理的图片数（FPS）。
2. [OmniDocBench](https://opendatalab.com/OpenDataLab/OmniDocBench)
也是opendatalab的，可用以评估文档解析效果。
https://github.com/opendatalab/OmniDocBench?tab=readme-ov-file

3. [M6Doc](https://github.com/HCIILAB/M6Doc)
* 多格式：包含扫描、拍摄和 PDF 文档。
* 多类型：涵盖科学文章、教科书、书籍、试卷、杂志、报纸和笔记等七种文档类型。
* 多布局：包含矩形、曼哈顿、非曼哈顿和多列曼哈顿等四种布局。
* 多语言：包含中文和英文文档。
*多标注类别：包含 74 种标注类别，共 237,116 个标注实例，分布在 9,080 页手动标注的文档中。


