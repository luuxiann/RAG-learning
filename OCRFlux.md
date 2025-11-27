# OCRFlux
https://link.zhihu.com/?target=https%3A//github.com/chatdoc-com/OCRFlux
作用：将 PDF 和图片转换为干净、可读、纯洁的 Markdown 文本
演示地址：https://ocrflux.pdfparser.io/#/
根据所给示例的尝试了一下，可以看到其在提取文字、表格内容上的准确性很高，不论是分栏还是跨页，还有页面边竖着的文字都能较好地处理识别。不过在跨栏且中间有图片解释干扰时识别就会出现一点问题。
|![OCR](./pictures/OCR1.png)|![OCR](./pictures/OCR2.png)|![OCR](./pictures/OCR3.png)|
|--|--|--|
|![OCR](./pictures/OCR4.png)|![OCR](./pictures/OCR5.png)|
|![OCR](./pictures/OCR6.png)|

OCRFlux 的 PDF 解析过程：
PDF 输入 
⬇️
转换为页面图像 
⬇️
构造 LLM 提示词 
⬇️
调用大模型生成 Markdown 
⬇️
错误重试机制 
⬇️
跨页元素合并（段落、表格）
⬇️
输出完整 Markdown 文档
大模型用的OCRFlux-3B
OCRFlux-3B建立在多模态LLM Qwen2.5-VL-3B-Instruct的微调基础上