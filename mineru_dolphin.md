### 完整文档解析时间
速度：18分钟加载模型
|解析时间|页数|平均解析时间（分钟/页）|
|--|--|--|
14 |9  |1.56
5 |2.5  |2
10 |8 |1.25
6 |4 |1.5
10 |5|2
平均速度1.662分钟/页
### 布局分析时间
Dolphin
|解析时间 s|页数|平均解析时间（s/页）|
|--|--|--|
26 |9  |2.88
78 |6  |13
91 |9 |10.1
27 |5 |5.4
161 |14|11.5
平均速度8.556分钟/页

mineru
|解析时间 s|页数|平均解析时间（s/页）|
|--|--|--|
30 |9  |3.33
111 |6  |18.5
202 |9 |22.44
18 |5 |3.6
214 |14|15.2
平均速度12.614分钟/页

融合时间
1s
1
1
1



## 第一次测评结果（未修正图片分类问题）
```
###### Process:  prediction_md_quick_match
【display_formula】
Edit_dist:
---------------  --------
ALL_page_avg     0.249856
edit_whole       0.247847
edit_sample_avg  0.225313
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  --------
equation_language: equation_ch          0.268647
equation_language: equation_en          0.222384
formula_type: handwriting               0.387069
formula_type: print                     0.224699
text_background: white                  0.458824
text_language: text_simplified_chinese  0.458824
text_rotate: normal                     0.458824
--------------------------------------  --------
====================================================================================================
sample_count:
--------------------------------------  ----
equation_language: equation_ch            67
equation_language: equation_en           991
formula_type: handwriting                  4
formula_type: print                     1054
text_background: white                     1
text_language: text_simplified_chinese     1
text_rotate: normal                        1
--------------------------------------  ----
====================================================================================================
Edit_dist:
--------------------------------  --------
ALL                               0.249856
None                              0.238962
colorful_backgroud                0.282502
data_source: PPT2PDF              0.240425
data_source: academic_literature  0.308783
data_source: book                 0.253239
data_source: colorful_textbook    0.233836
data_source: exam_paper           0.244735
data_source: note                 0.333333
fuzzy_scan                        0.424709
language: english                 0.230089
language: simplified_chinese      0.310391
layout: 1andmore_column           0.213292
layout: double_column             0.252879
layout: other_layout              0.374213
layout: single_column             0.241142
layout: three_column              0.375556
watermark                         0.34511
--------------------------------  --------
====================================================================================================
【reading_order】
Edit_dist:
---------------  --------
ALL_page_avg     0.158525
edit_whole       0.356361
edit_sample_avg  0.158525
---------------  --------
====================================================================================================
----Anno Attribute---------------
sample_count:

====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.158525
None                              0.109411
colorful_backgroud                0.134871
data_source: PPT2PDF              0.128893
data_source: academic_literature  0.0310335
data_source: book                 0.0829698
data_source: colorful_textbook    0.123591
data_source: exam_paper           0.150051
data_source: magazine             0.0858845
data_source: newspaper            0.512703
data_source: note                 0.184172
data_source: research_report      0.0628579
fuzzy_scan                        0.124045
language: en_ch_mixed             0.198673
language: english                 0.110071
language: simplified_chinese      0.202434
layout: 1andmore_column           0.082791
layout: double_column             0.0807998
layout: other_layout              0.320212
layout: single_column             0.121595
layout: three_column              0.121861
watermark                         0.185739
--------------------------------  ---------
====================================================================================================
【table】
TEDS:
---  --------
all  0.719741
---  --------
====================================================================================================
TEDS_structure_only:
---  --------
all  0.776487
---  --------
====================================================================================================
Edit_dist:
---------------  --------
ALL_page_avg     0.160995
edit_whole       0.58919
edit_sample_avg  0.247946
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
----------------------------------  --------
include_background: False           0.276691
include_background: True            0.182949
include_equation: False             0.250231
include_equation: True              0.236935
include_photo: False                0.247946
language: table_en                  0.341007
language: table_en_ch_mixed         0.206082
language: table_simplified_chinese  0.189095
line: fewer_line                    0.277298
line: full_line                     0.139244
line: less_line                     0.176485
line: wireless_line                 0.78851
table_layout: horizontal            0.245754
table_layout: vertical              0.406088
with_span: False                    0.286847
with_span: True                     0.161582
with_structured_text: False         0.172863
with_structured_text: True          0.188689
----------------------------------  --------
====================================================================================================
TEDS:
----------------------------------  --------
include_background: False           0.688905
include_background: True            0.789466
include_equation: False             0.715896
include_equation: True              0.738267
include_photo: False                0.719741
language: table_en                  0.619035
language: table_en_ch_mixed         0.806202
language: table_simplified_chinese  0.780496
line: fewer_line                    0.678768
line: full_line                     0.827915
line: less_line                     0.80857
line: wireless_line                 0.199602
table_layout: horizontal            0.722848
table_layout: vertical              0.495603
with_span: False                    0.680205
with_span: True                     0.807517
with_structured_text: False         0.796131
with_structured_text: True          0.87259
----------------------------------  --------
====================================================================================================
TEDS_structure_only:
----------------------------------  --------
include_background: False           0.748967
include_background: True            0.838714
include_equation: False             0.760685
include_equation: True              0.852624
include_photo: False                0.776487
language: table_en                  0.693149
language: table_en_ch_mixed         0.88254
language: table_simplified_chinese  0.824308
line: fewer_line                    0.753584
line: full_line                     0.878898
line: less_line                     0.866275
line: wireless_line                 0.217523
table_layout: horizontal            0.777509
table_layout: vertical              0.702748
with_span: False                    0.730452
with_span: True                     0.878689
with_structured_text: False         0.848818
with_structured_text: True          0.927291
----------------------------------  --------
====================================================================================================
sample_count:
----------------------------------  ---
include_background: False           355
include_background: True            157
include_equation: False             424
include_equation: True               88
include_photo: False                512
language: table_en                  196
language: table_en_ch_mixed          21
language: table_simplified_chinese  295
line: fewer_line                    169
line: full_line                     231
line: less_line                      66
line: wireless_line                  46
table_layout: horizontal            505
table_layout: vertical                7
with_span: False                    353
with_span: True                     159
with_structured_text: False         413
with_structured_text: True           15
----------------------------------  ---
====================================================================================================
Edit_dist:
--------------------------------  ----------
ALL                               0.160995
None                              0.757727
colorful_backgroud                0.117669
data_source: PPT2PDF              0.168144
data_source: academic_literature  0.175661
data_source: book                 0.113337
data_source: colorful_textbook    0.103352
data_source: exam_paper           0.101433
data_source: magazine             0.0957366
data_source: newspaper            0.545105
data_source: note                 0.290483
data_source: research_report      0.136113
fuzzy_scan                        0.00831389
language: en_ch_mixed             0.09787
language: english                 0.209965
language: simplified_chinese      0.141716
layout: 1andmore_column           0.160936
layout: double_column             0.109925
layout: other_layout              0.251416
layout: single_column             0.149245
layout: three_column              0.253687
watermark                         0.0465167
--------------------------------  ----------
====================================================================================================
TEDS:
--------------------------------  --------
ALL                               0.809529
None                              0.213591
colorful_backgroud                0.895879
data_source: PPT2PDF              0.831708
data_source: academic_literature  0.785084
data_source: book                 0.864033
data_source: colorful_textbook    0.936604
data_source: exam_paper           0.869965
data_source: magazine             0.873432
data_source: newspaper            0.366589
data_source: note                 0.603523
data_source: research_report      0.840986
fuzzy_scan                        0.984813
language: en_ch_mixed             0.865024
language: english                 0.75746
language: simplified_chinese      0.83109
layout: 1andmore_column           0.811426
layout: double_column             0.850288
layout: other_layout              0.705228
layout: single_column             0.82647
layout: three_column              0.597688
watermark                         0.951478
--------------------------------  --------
====================================================================================================
TEDS_structure_only:
--------------------------------  --------
ALL                               0.865526
None                              0.25
colorful_backgroud                0.942671
data_source: PPT2PDF              0.888024
data_source: academic_literature  0.889857
data_source: book                 0.915274
data_source: colorful_textbook    0.977387
data_source: exam_paper           0.899006
data_source: magazine             0.968908
data_source: newspaper            0.505921
data_source: note                 0.679893
data_source: research_report      0.864305
fuzzy_scan                        1
language: en_ch_mixed             0.917959
language: english                 0.840336
language: simplified_chinese      0.873621
layout: 1andmore_column           0.887126
layout: double_column             0.894651
layout: other_layout              0.797812
layout: single_column             0.871033
layout: three_column              0.7
watermark                         0.969318
--------------------------------  --------
====================================================================================================
【text_block】
Edit_dist:
---------------  --------
ALL_page_avg     0.258039
edit_whole       0.490917
edit_sample_avg  0.387834
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  ---------
equation_language: equation_ch          0.545455
equation_language: equation_en          0.449474
formula_type: print                     0.450974
text_background: multi_colored          0.24068
text_background: single_colored         0.316186
text_background: white                  0.380355
text_language: other                    0.0387454
text_language: text_en_ch_mixed         0.203174
text_language: text_english             0.258058
text_language: text_simplified_chinese  0.470253
text_rotate: horizontal                 0.685992
text_rotate: normal                     0.365219
text_rotate: rotate270                  0.953735
--------------------------------------  ---------
====================================================================================================
sample_count:
--------------------------------------  -----
equation_language: equation_ch              1
equation_language: equation_en             63
formula_type: print                        64
text_background: multi_colored           1103
text_background: single_colored           887
text_background: white                  13844
text_language: other                        2
text_language: text_en_ch_mixed           363
text_language: text_english              7264
text_language: text_simplified_chinese   8211
text_rotate: horizontal                    46
text_rotate: normal                     15736
text_rotate: rotate270                     31
--------------------------------------  -----
====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.258039
None                              0.207203
colorful_backgroud                0.20815
data_source: PPT2PDF              0.278711
data_source: academic_literature  0.122403
data_source: book                 0.113506
data_source: colorful_textbook    0.18281
data_source: exam_paper           0.247319
data_source: magazine             0.124202
data_source: newspaper            0.668735
data_source: note                 0.371959
data_source: research_report      0.0924481
fuzzy_scan                        0.222791
language: en_ch_mixed             0.330675
language: english                 0.207396
language: simplified_chinese      0.299072
layout: 1andmore_column           0.16301
layout: double_column             0.153713
layout: other_layout              0.433573
layout: single_column             0.227088
layout: three_column              0.212766
watermark                         0.310841
--------------------------------  ---------
====================================================================================================
```


## 第二次测评结果（修正表格分类问题）
01-12 16：12：44 - 16：40：38   1674秒  44页  38.04秒/页
01-11 17:33:33 ~ 01-14 05:24:31   214838秒   1355页   158.55秒/页
01-13 17:14:03 - 18:28:31     4468秒  44页   101.54秒/页
01-14 19:42:22 - 20:06:45    1463秒 44页   33.25秒/页
01-14 20:47:05 - 21:11:10     1445秒 44页 
01-14 21:23:19 - 21:48:02  1483.6秒
```
 "total_processing_time_seconds": 1236.16,  44页
  "start_time": "2026-01-15 15:53:48",
  "end_time": "2026-01-15 16:14:24"
```
```
2026-01-12
1. MinerU VLM布局识别（主要方法） 
   ↓
2. 对疑似不确定表格使用Dolphin进行二次验证
   ↓
3. 跨页表格检测：如果上一页以表格结尾，对当前页进行Dolphin分析
   ↓
4. 重新分类header blocks（结合位置和语义内容）
   ↓
5. 过滤表格周围的文本
   ↓
6. 对表格使用激进扩大策略（不覆盖表格无关内容）
   ↓
7. 内容提取和后处理

2026-01-14
1. MinerU VLM布局识别（主要方法）
   ↓
2. 对疑似不确定表格使用Dolphin进行二次验证（图片先resize_img处理）
   ↓
3. 跨页表格检测
   ↓
4. 重新分类header blocks
   ↓
5. 验证table_caption和table_footnote
   ↓
6. 过滤表格周围的文本
   ↓
7. 对表格使用激进扩大策略（参考dolphin_mineru_layout_comparison.py）
   ↓
8. 使用MinerU的阅读顺序调整策略
   ↓
9. 内容提取和后处理
```
```
【display_formula】
Edit_dist:
---------------  --------
ALL_page_avg     0.254628
edit_whole       0.248847
edit_sample_avg  0.226568
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  --------
equation_language: equation_ch          0.278123
equation_language: equation_en          0.223076
formula_type: handwriting               0.387069
formula_type: print                     0.225967
text_background: white                  0.458824
text_language: text_simplified_chinese  0.458824
text_rotate: normal                     0.458824
--------------------------------------  --------
====================================================================================================
sample_count:
--------------------------------------  ----
equation_language: equation_ch            68
equation_language: equation_en          1004
formula_type: handwriting                  4
formula_type: print                     1068
text_background: white                     1
text_language: text_simplified_chinese     1
text_rotate: normal                        1
--------------------------------------  ----
====================================================================================================
Edit_dist:
--------------------------------  --------
ALL                               0.254628
None                              0.24313
colorful_backgroud                0.298651
data_source: PPT2PDF              0.262738
data_source: academic_literature  0.308783
data_source: book                 0.253239
data_source: colorful_textbook    0.273482
data_source: exam_paper           0.241676
data_source: note                 0.333333
fuzzy_scan                        0.501948
language: english                 0.233737
language: simplified_chinese      0.315251
layout: 1andmore_column           0.206412
layout: double_column             0.260849
layout: other_layout              0.408111
layout: single_column             0.243015
layout: three_column              0.375556
watermark                         0.344415
--------------------------------  --------
====================================================================================================
【reading_order】
Edit_dist:
---------------  --------
ALL_page_avg     0.167682
edit_whole       0.360567
edit_sample_avg  0.167682
---------------  --------
====================================================================================================
----Anno Attribute---------------
sample_count:

====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.167682
None                              0.119908
colorful_backgroud                0.142301
data_source: PPT2PDF              0.140159
data_source: academic_literature  0.0305514
data_source: book                 0.0826472
data_source: colorful_textbook    0.142895
data_source: exam_paper           0.15898
data_source: magazine             0.097022
data_source: newspaper            0.513226
data_source: note                 0.21662
data_source: research_report      0.0608377
fuzzy_scan                        0.143986
language: en_ch_mixed             0.214702
language: english                 0.118036
language: simplified_chinese      0.211738
layout: 1andmore_column           0.0862697
layout: double_column             0.0920533
layout: other_layout              0.324614
layout: single_column             0.134695
layout: three_column              0.122332
watermark                         0.203213
--------------------------------  ---------
====================================================================================================
【table】
TEDS:
---  --------
all  0.718765
---  --------
====================================================================================================
TEDS_structure_only:
---  -------
all  0.77551
---  -------
====================================================================================================
Edit_dist:
---------------  --------
ALL_page_avg     0.161957
edit_whole       0.58973
edit_sample_avg  0.248947
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
----------------------------------  --------
include_background: False           0.278134
include_background: True            0.182949
include_equation: False             0.25144
include_equation: True              0.236935
include_photo: False                0.248947
language: table_en                  0.341007
language: table_en_ch_mixed         0.206082
language: table_simplified_chinese  0.190832
line: fewer_line                    0.277298
line: full_line                     0.141463
line: less_line                     0.176485
line: wireless_line                 0.78851
table_layout: horizontal            0.246768
table_layout: vertical              0.406088
with_span: False                    0.288298
with_span: True                     0.161582
with_structured_text: False         0.172863
with_structured_text: True          0.188689
----------------------------------  --------
====================================================================================================
TEDS:
----------------------------------  --------
include_background: False           0.687497
include_background: True            0.789466
include_equation: False             0.714717
include_equation: True              0.738267
include_photo: False                0.718765
language: table_en                  0.619035
language: table_en_ch_mixed         0.806202
language: table_simplified_chinese  0.778802
line: fewer_line                    0.678768
line: full_line                     0.825751
line: less_line                     0.80857
line: wireless_line                 0.199602
table_layout: horizontal            0.721858
table_layout: vertical              0.495603
with_span: False                    0.678788
with_span: True                     0.807517
with_structured_text: False         0.796131
with_structured_text: True          0.87259
----------------------------------  --------
====================================================================================================
TEDS_structure_only:
----------------------------------  --------
include_background: False           0.747558
include_background: True            0.838714
include_equation: False             0.759505
include_equation: True              0.852624
include_photo: False                0.77551
language: table_en                  0.693149
language: table_en_ch_mixed         0.88254
language: table_simplified_chinese  0.822613
line: fewer_line                    0.753584
line: full_line                     0.876733
line: less_line                     0.866275
line: wireless_line                 0.217523
table_layout: horizontal            0.776519
table_layout: vertical              0.702748
with_span: False                    0.729036
with_span: True                     0.878689
with_structured_text: False         0.848818
with_structured_text: True          0.927291
----------------------------------  --------
====================================================================================================
sample_count:
----------------------------------  ---
include_background: False           355
include_background: True            157
include_equation: False             424
include_equation: True               88
include_photo: False                512
language: table_en                  196
language: table_en_ch_mixed          21
language: table_simplified_chinese  295
line: fewer_line                    169
line: full_line                     231
line: less_line                      66
line: wireless_line                  46
table_layout: horizontal            505
table_layout: vertical                7
with_span: False                    353
with_span: True                     159
with_structured_text: False         413
with_structured_text: True           15
----------------------------------  ---
====================================================================================================
Edit_dist:
--------------------------------  ----------
ALL                               0.161957
None                              0.757727
colorful_backgroud                0.117669
data_source: PPT2PDF              0.168144
data_source: academic_literature  0.175661
data_source: book                 0.113337
data_source: colorful_textbook    0.103352
data_source: exam_paper           0.101433
data_source: magazine             0.0957366
data_source: newspaper            0.569213
data_source: note                 0.290483
data_source: research_report      0.136113
fuzzy_scan                        0.00831389
language: en_ch_mixed             0.09787
language: english                 0.212979
language: simplified_chinese      0.141716
layout: 1andmore_column           0.160936
layout: double_column             0.109925
layout: other_layout              0.258916
layout: single_column             0.149245
layout: three_column              0.253687
watermark                         0.0465167
--------------------------------  ----------
====================================================================================================
TEDS:
--------------------------------  --------
ALL                               0.808817
None                              0.213591
colorful_backgroud                0.895879
data_source: PPT2PDF              0.831708
data_source: academic_literature  0.785084
data_source: book                 0.864033
data_source: colorful_textbook    0.936604
data_source: exam_paper           0.869965
data_source: magazine             0.873432
data_source: newspaper            0.348732
data_source: note                 0.603523
data_source: research_report      0.840986
fuzzy_scan                        0.984813
language: en_ch_mixed             0.865024
language: english                 0.755228
language: simplified_chinese      0.83109
layout: 1andmore_column           0.811426
layout: double_column             0.850288
layout: other_layout              0.699672
layout: single_column             0.82647
layout: three_column              0.597688
watermark                         0.951478
--------------------------------  --------
====================================================================================================
TEDS_structure_only:
--------------------------------  --------
ALL                               0.864814
None                              0.25
colorful_backgroud                0.942671
data_source: PPT2PDF              0.888024
data_source: academic_literature  0.889857
data_source: book                 0.915274
data_source: colorful_textbook    0.977387
data_source: exam_paper           0.899006
data_source: magazine             0.968908
data_source: newspaper            0.488064
data_source: note                 0.679893
data_source: research_report      0.864305
fuzzy_scan                        1
language: en_ch_mixed             0.917959
language: english                 0.838104
language: simplified_chinese      0.873621
layout: 1andmore_column           0.887126
layout: double_column             0.894651
layout: other_layout              0.792256
layout: single_column             0.871033
layout: three_column              0.7
watermark                         0.969318
--------------------------------  --------
====================================================================================================
【text_block】
Edit_dist:
---------------  --------
ALL_page_avg     0.260513
edit_whole       0.494698
edit_sample_avg  0.391614
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  ---------
equation_language: equation_en          0.405038
formula_type: print                     0.405038
text_background: multi_colored          0.248925
text_background: single_colored         0.32369
text_background: white                  0.384113
text_language: other                    0.0387454
text_language: text_en_ch_mixed         0.204629
text_language: text_english             0.263037
text_language: text_simplified_chinese  0.474097
text_rotate: horizontal                 0.685992
text_rotate: normal                     0.369517
text_rotate: rotate270                  0.953735
--------------------------------------  ---------
====================================================================================================
sample_count:
--------------------------------------  -----
equation_language: equation_en             50
formula_type: print                        50
text_background: multi_colored           1110
text_background: single_colored           887
text_background: white                  13859
text_language: other                        2
text_language: text_en_ch_mixed           363
text_language: text_english              7278
text_language: text_simplified_chinese   8220
text_rotate: horizontal                    46
text_rotate: normal                     15758
text_rotate: rotate270                     31
--------------------------------------  -----
====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.260513
None                              0.208136
colorful_backgroud                0.206178
data_source: PPT2PDF              0.274836
data_source: academic_literature  0.122413
data_source: book                 0.112574
data_source: colorful_textbook    0.18296
data_source: exam_paper           0.247813
data_source: magazine             0.135247
data_source: newspaper            0.670823
data_source: note                 0.392302
data_source: research_report      0.0929394
fuzzy_scan                        0.220126
language: en_ch_mixed             0.335095
language: english                 0.20867
language: simplified_chinese      0.302484
layout: 1andmore_column           0.161157
layout: double_column             0.15348
layout: other_layout              0.437419
layout: single_column             0.2309
layout: three_column              0.212968
watermark                         0.31388
--------------------------------  ---------
====================================================================================================
评测完成！结果保存在: /home/hsr/OmniDocBench/result
================================================================================
```
## 单Dolphin(第三次)
```
###### Process:  prediction_md_quick_match
【display_formula】
Edit_dist:
---------------  --------
ALL_page_avg     0.258484
edit_whole       0.250458
edit_sample_avg  0.230082
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  --------
equation_language: equation_ch          0.290227
equation_language: equation_en          0.225967
formula_type: handwriting               0.394013
formula_type: print                     0.229462
text_background: white                  0.458824
text_language: text_simplified_chinese  0.458824
text_rotate: normal                     0.458824
--------------------------------------  --------
====================================================================================================
sample_count:
--------------------------------------  ----
equation_language: equation_ch            68
equation_language: equation_en           994
formula_type: handwriting                  4
formula_type: print                     1058
text_background: white                     1
text_language: text_simplified_chinese     1
text_rotate: normal                        1
--------------------------------------  ----
====================================================================================================
Edit_dist:
--------------------------------  --------
ALL                               0.258484
None                              0.24422
colorful_backgroud                0.31979
data_source: PPT2PDF              0.278849
data_source: academic_literature  0.356163
data_source: book                 0.247672
data_source: colorful_textbook    0.277606
data_source: exam_paper           0.242193
data_source: note                 0.333333
fuzzy_scan                        0.500574
language: english                 0.237317
language: simplified_chinese      0.319908
layout: 1andmore_column           0.215136
layout: double_column             0.265052
layout: other_layout              0.410078
layout: single_column             0.245035
layout: three_column              0.393306
watermark                         0.358296
--------------------------------  --------
====================================================================================================
【reading_order】
Edit_dist:
---------------  --------
ALL_page_avg     0.167735
edit_whole       0.364888
edit_sample_avg  0.167735
---------------  --------
====================================================================================================
----Anno Attribute---------------
sample_count:

====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.167735
None                              0.117868
colorful_backgroud                0.145302
data_source: PPT2PDF              0.137978
data_source: academic_literature  0.0338079
data_source: book                 0.0846041
data_source: colorful_textbook    0.142235
data_source: exam_paper           0.162051
data_source: magazine             0.0950497
data_source: newspaper            0.521056
data_source: note                 0.206591
data_source: research_report      0.0550296
fuzzy_scan                        0.145224
language: en_ch_mixed             0.198634
language: english                 0.118249
language: simplified_chinese      0.21418
layout: 1andmore_column           0.0896499
layout: double_column             0.0839964
layout: other_layout              0.327166
layout: single_column             0.134718
layout: three_column              0.124545
watermark                         0.213751
--------------------------------  ---------
====================================================================================================
【table】
TEDS:
---  --------
all  0.715883
---  --------
====================================================================================================
TEDS_structure_only:
---  --------
all  0.773971
---  --------
====================================================================================================
Edit_dist:
---------------  --------
ALL_page_avg     0.165995
edit_whole       0.601128
edit_sample_avg  0.252719
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
----------------------------------  --------
include_background: False           0.278769
include_background: True            0.193815
include_equation: False             0.257501
include_equation: True              0.229677
include_photo: False                0.252719
language: table_en                  0.34724
language: table_en_ch_mixed         0.211358
language: table_simplified_chinese  0.192863
line: fewer_line                    0.278797
line: full_line                     0.147143
line: less_line                     0.173941
line: wireless_line                 0.80011
table_layout: horizontal            0.249961
table_layout: vertical              0.451663
with_span: False                    0.292936
with_span: True                     0.163431
with_structured_text: False         0.176763
with_structured_text: True          0.210627
----------------------------------  --------
====================================================================================================
TEDS:
----------------------------------  --------
include_background: False           0.690184
include_background: True            0.773991
include_equation: False             0.710252
include_equation: True              0.743013
include_photo: False                0.715883
language: table_en                  0.617575
language: table_en_ch_mixed         0.789958
language: table_simplified_chinese  0.775926
line: fewer_line                    0.685509
line: full_line                     0.818849
line: less_line                     0.800746
line: wireless_line                 0.188641
table_layout: horizontal            0.719608
table_layout: vertical              0.44712
with_span: False                    0.676016
with_span: True                     0.80439
with_structured_text: False         0.792585
with_structured_text: True          0.841385
----------------------------------  --------
====================================================================================================
TEDS_structure_only:
----------------------------------  --------
include_background: False           0.748789
include_background: True            0.830913
include_equation: False             0.756085
include_equation: True              0.860152
include_photo: False                0.773971
language: table_en                  0.690139
language: table_en_ch_mixed         0.870635
language: table_simplified_chinese  0.822789
line: fewer_line                    0.758187
line: full_line                     0.87272
line: less_line                     0.864165
line: wireless_line                 0.206666
table_layout: horizontal            0.775688
table_layout: vertical              0.650125
with_span: False                    0.727159
with_span: True                     0.8779
with_structured_text: False         0.847423
with_structured_text: True          0.896021
----------------------------------  --------
====================================================================================================
sample_count:
----------------------------------  ---
include_background: False           355
include_background: True            157
include_equation: False             424
include_equation: True               88
include_photo: False                512
language: table_en                  196
language: table_en_ch_mixed          21
language: table_simplified_chinese  295
line: fewer_line                    169
line: full_line                     231
line: less_line                      66
line: wireless_line                  46
table_layout: horizontal            505
table_layout: vertical                7
with_span: False                    353
with_span: True                     159
with_structured_text: False         413
with_structured_text: True           15
----------------------------------  ---
====================================================================================================
Edit_dist:
--------------------------------  ----------
ALL                               0.165995
None                              0.757727
colorful_backgroud                0.141556
data_source: PPT2PDF              0.185744
data_source: academic_literature  0.189148
data_source: book                 0.112093
data_source: colorful_textbook    0.114247
data_source: exam_paper           0.102941
data_source: magazine             0.0994531
data_source: newspaper            0.553977
data_source: note                 0.305688
data_source: research_report      0.131898
fuzzy_scan                        0.00831389
language: en_ch_mixed             0.108297
language: english                 0.215979
language: simplified_chinese      0.145702
layout: 1andmore_column           0.177868
layout: double_column             0.110673
layout: other_layout              0.259718
layout: single_column             0.151034
layout: three_column              0.266181
watermark                         0.0483308
--------------------------------  ----------
====================================================================================================
TEDS:
--------------------------------  --------
ALL                               0.804366
None                              0.213591
colorful_backgroud                0.84949
data_source: PPT2PDF              0.80991
data_source: academic_literature  0.786884
data_source: book                 0.869074
data_source: colorful_textbook    0.918084
data_source: exam_paper           0.86633
data_source: magazine             0.796374
data_source: newspaper            0.383424
data_source: note                 0.571148
data_source: research_report      0.843331
fuzzy_scan                        0.984813
language: en_ch_mixed             0.832829
language: english                 0.761765
language: simplified_chinese      0.823553
layout: 1andmore_column           0.808157
layout: double_column             0.84855
layout: other_layout              0.694807
layout: single_column             0.82143
layout: three_column              0.58633
watermark                         0.946561
--------------------------------  --------
====================================================================================================
TEDS_structure_only:
--------------------------------  --------
ALL                               0.86272
None                              0.25
colorful_backgroud                0.914594
data_source: PPT2PDF              0.873347
data_source: academic_literature  0.891498
data_source: book                 0.917108
data_source: colorful_textbook    0.960959
data_source: exam_paper           0.896667
data_source: magazine             0.978859
data_source: newspaper            0.518681
data_source: note                 0.661988
data_source: research_report      0.864668
fuzzy_scan                        1
language: en_ch_mixed             0.886293
language: english                 0.843169
language: simplified_chinese      0.870566
layout: 1andmore_column           0.883537
layout: double_column             0.894651
layout: other_layout              0.798145
layout: single_column             0.867299
layout: three_column              0.7
watermark                         0.963258
--------------------------------  --------
====================================================================================================
【text_block】
Edit_dist:
---------------  --------
ALL_page_avg     0.257091
edit_whole       0.499989
edit_sample_avg  0.391179
---------------  --------
====================================================================================================
----Anno Attribute---------------
Edit_dist:
--------------------------------------  --------
equation_language: equation_en          0.405509
formula_type: print                     0.405509
text_background: multi_colored          0.242088
text_background: single_colored         0.321093
text_background: white                  0.384072
text_language: other                    0.037594
text_language: text_en_ch_mixed         0.20588
text_language: text_english             0.260303
text_language: text_simplified_chinese  0.47505
text_rotate: horizontal                 0.679942
text_rotate: normal                     0.368974
text_rotate: rotate270                  0.953979
--------------------------------------  --------
====================================================================================================
sample_count:
--------------------------------------  -----
equation_language: equation_en             59
formula_type: print                        59
text_background: multi_colored           1108
text_background: single_colored           887
text_background: white                  13837
text_language: other                        2
text_language: text_en_ch_mixed           363
text_language: text_english              7260
text_language: text_simplified_chinese   8213
text_rotate: horizontal                    45
text_rotate: normal                     15731
text_rotate: rotate270                     31
--------------------------------------  -----
====================================================================================================
Edit_dist:
--------------------------------  ---------
ALL                               0.257091
None                              0.200551
colorful_backgroud                0.213206
data_source: PPT2PDF              0.27674
data_source: academic_literature  0.111063
data_source: book                 0.10886
data_source: colorful_textbook    0.179995
data_source: exam_paper           0.244361
data_source: magazine             0.132516
data_source: newspaper            0.666673
data_source: note                 0.383326
data_source: research_report      0.0949716
fuzzy_scan                        0.222425
language: en_ch_mixed             0.3242
language: english                 0.202841
language: simplified_chinese      0.302741
layout: 1andmore_column           0.159595
layout: double_column             0.139165
layout: other_layout              0.434889
layout: single_column             0.23048
layout: three_column              0.198688
watermark                         0.313255
--------------------------------  ---------
====================================================================================================
评测完成！结果保存在: /home/hsr/OmniDocBench/result
================================================================================
处理完成！
评测结果将保存在: /home/hsr/OmniDocBench/result
================================================================================
```
