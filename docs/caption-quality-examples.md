# 附录：真实例子

> **本文件由 `recaption/scripts/extract_examples.py` 生成，请勿手改**（重跑会覆盖）。
>
> 每条都是审查产物里的**原始记录**，逐字摘录、未经改写。唯一的编辑行为是「挑哪几条」，而挑选规则印在每个小节标题下，且是确定性的（按给定顺序取前 N 条，无随机）。
>
> 配套结论见 [`caption-quality-report.md`](caption-quality-report.md)。样本 = 10,000 条 ｜ 每节 3 例。

---

## 一、英文源 caption 的问题

### 1. 与画面不符（源头错）

**选取规则**：`en_score≤2` 且 `sig_zh` 最低的前 N 条 —— 即「VLM 与嵌入模型都认为最差」的记录，两个独立信号交叉确认

| clip | 英文源 caption | 中文译文 | sig_zh | en分 | VLM 标注的问题 |
|---|---|---|---|---|---|
| `sZ1tpezSfgc_00:02:37.200_00:02:39.400.mp4` | the lululemon virtual sexy girl is wearing a blue shirt | lululemon虚拟性感女孩身穿蓝色上衣 | -0.018 | 2 | wrong_subject, hallucinated_detail |
| `xOTVi9EM7bA_00:00:41.375_00:01:39.499.mp4` | russian police use toyota cars to patrol the city | 俄罗斯警察使用丰田汽车在城市巡逻 | -0.006 | 2 | wrong_scene, hallucinated_detail |
| `WW8icbd0XHo_00:00:23.267_00:00:34.433.mp4` | аруител наал авлетс оател | 汽车旅馆 | -0.004 | 0 | wrong_subject, hallucinated_detail |

### 2. 视频有先后事件，字幕只字未提（时序缺失）

**选取规则**：`temporal_order_relevant=true` 且 `temporal_order_correct=null`，取英文最长的前 N 条（长句都不写顺序，最能说明问题）

| clip | 英文源 caption | 中文译文 |
|---|---|---|
| `zYmDM9M9x8M_00:02:42.199_00:02:52.199.mp4` | the first step is to prepare the plants for planting. the seed is planted in a pot and ready to be planted. the first step is to prepare the plants for planting. | 第一步是为种植做准备。种子已种入花盆，准备移栽。第一步是为种植做准备。 |
| `vlpMAtuMDv0_00:03:33.856_00:03:43.856.mp4` | a screenshot of the step to select a number. an image of a keyboard with a text message on the app. a screenshot of a text message from the app. | 选择号码的步骤截图。应用内键盘及短信界面的图片。应用内短信的截图。 |
| `ebMpPFoAEMM_00:03:03.967_00:03:11.100.mp4` | the pattern is repeated with the introduction of a third ball and the instruction the pattern is to be repeated instruction | 引入第三个球后，模式重复，指令要求重复该模式 |

### 2b. 字幕写了顺序、但写反了

**选取规则**：`temporal_order_correct=false` —— **全库仅 8 条**，此处列出前 N 条。n 太小不可外推，仅作定性参考

| clip | 英文源 caption | 中文译文 |
|---|---|---|
| `_dKRitNWn6I_00:05:50.317_00:05:55.722.mp4` | the blue icing is pouring into a pot | 蓝色糖霜正倒入锅中 |
| `motX6kkA-r4_00:00:14.306_00:00:15.974.mp4` | a soldier standing in the dark on top of a roof | 一名士兵站在黑暗的屋顶上 |
| `ra1kQED6S5w_00:01:45.792_00:01:46.833.mp4` | a black limo driving down a busy street | 一辆黑色豪华轿车行驶在繁忙的街道上 |

### 3. 描述笼统（粒度粗）

**选取规则**：英文 caption **最短**的前 N 条 —— 用**长度**这个不依赖模型的客观量挑选，因为 `granularity` 那一维已作废（被质量分闸住，见 caption-quality-report §1.3）

| clip | 英文源 caption | 中文译文 | 英文词数 | VLM 粒度判定 |
|---|---|---|---|---|
| `uPttz67FhK8_00:00:00.000_00:00:07.074.mp4` | air save logo | Air Save 标志 | 3 | coarse |
| `9P6EGIwz4Vk_00:05:32.966_00:05:35.702.mp4` | cz pistol grip | CZ手枪握把 | 3 | coarse |
| `S6SV9GVqPGQ_00:07:11.333_00:07:13.533.mp4` | pea ea ga - 00 | pea ea ga - 00 | 5 | coarse |

### 4a. 错误类型：认错主体（`wrong_subject`）

**选取规则**：`en_issues` 含 `wrong_subject`，按 `sig_zh` 从低到高取前 N 条。⚠️ 该标签出自已塌缩的合并 rubric，**看形态、不看比例**

| clip | 英文源 caption | sig_zh | en分 |
|---|---|---|---|
| `sZ1tpezSfgc_00:02:37.200_00:02:39.400.mp4` | the lululemon virtual sexy girl is wearing a blue shirt | -0.018 | 2 |
| `WW8icbd0XHo_00:00:23.267_00:00:34.433.mp4` | аруител наал авлетс оател | -0.004 | 0 |
| `kopHHwN8-Uo_00:00:20.920_00:00:22.680.mp4` | an image of a black background with a small bird | 0.007 | 0 |

### 4b. 错误类型：认错动作（`wrong_action`）

**选取规则**：`en_issues` 含 `wrong_action`，按 `sig_zh` 从低到高取前 N 条。⚠️ 该标签出自已塌缩的合并 rubric，**看形态、不看比例**

| clip | 英文源 caption | sig_zh | en分 |
|---|---|---|---|
| `VZKGM_AuPUY_00:02:19.667_00:02:53.267.mp4` | galaxy note10 rcpt | 0.025 | 3 |
| `fst9tjBKC2Y_00:06:15.775_00:06:23.850.mp4` | a man is cutting a piece of plastic that looks like a lamp | 0.029 | 2 |
| `ymDhLSV0CeI_00:08:49.240_00:09:02.920.mp4` | tania morgan and amy osborn show you how to get the sexy glow | 0.033 | 2 |

### 4c. 错误类型：编造细节（`hallucinated_detail`）

**选取规则**：`en_issues` 含 `hallucinated_detail`，按 `sig_zh` 从低到高取前 N 条。⚠️ 该标签出自已塌缩的合并 rubric，**看形态、不看比例**

| clip | 英文源 caption | sig_zh | en分 |
|---|---|---|---|
| `sZ1tpezSfgc_00:02:37.200_00:02:39.400.mp4` | the lululemon virtual sexy girl is wearing a blue shirt | -0.018 | 2 |
| `xOTVi9EM7bA_00:00:41.375_00:01:39.499.mp4` | russian police use toyota cars to patrol the city | -0.006 | 2 |
| `WW8icbd0XHo_00:00:23.267_00:00:34.433.mp4` | аруител наал авлетс оател | -0.004 | 0 |

### 4d. 错误类型：把水印/字幕当画面内容（`ocr_or_watermark_read`）

**选取规则**：`en_issues` 含 `ocr_or_watermark_read`，按 `sig_zh` 从低到高取前 N 条。⚠️ 该标签出自已塌缩的合并 rubric，**看形态、不看比例**

| clip | 英文源 caption | sig_zh | en分 |
|---|---|---|---|
| `ad1-l4_WslE_00:00:02.520_00:00:08.640.mp4` | video card balance - mai 2020 | 0.050 | 2 |
| `0eISQGAM5D0_00:01:11.367_00:01:37.367.mp4` | i f t possible t d ve l heure e de développement de d eve | 0.072 | 2 |
| `_MXI3Dm475w_00:00:03.637_00:00:09.309.mp4` | a person in the dark with the quote about | 0.079 | 3 |

---

## 二、中文译文 caption 的问题

### 5. 真实翻译错误（3.6% 的那批）

**选取规则**：**文本-only 重判**的 `trans_score≤2`，按分数升序取前 N 条。这一版判定不看画面、只比中英两条文本，是对翻译环节的专项测量

| clip | 英文源 | 中文译文 | 翻译分 | 问题标签 |
|---|---|---|---|---|
| `fXUqy-x8Q-g_00:03:08.720_00:03:27.240.mp4` | an image of someone on a laptop with the words je sais pas à l'oise | 一张有人在笔记本电脑上的图片，上面写着“je sais pas à l'oise” | 0 | untranslated_fragment |
| `TQaJ3IiJAUc_00:03:39.967_00:03:43.933.mp4` | kak susu dana jumlah nisa khan | 卡克苏苏达纳朱马特尼萨汗 | 0 | mistranslation, artifact_text |
| `nsNVm7esIys_00:00:13.080_00:00:15.840.mp4` | и тем олени на он влтс | 以及驯鹿 | 0 | mistranslation, artifact_text |

### 6. 真·未翻译（0.3%）

**选取规则**：文本-only 重判 `untranslated=true` —— **全库仅 33 条**

| clip | 英文源 | 中文栏内容 |
|---|---|---|
| `fXUqy-x8Q-g_00:03:08.720_00:03:27.240.mp4` | an image of someone on a laptop with the words je sais pas à l'oise | 一张有人在笔记本电脑上的图片，上面写着“je sais pas à l'oise” |
| `3OU6wtbZsdQ_00:11:39.433_00:11:43.633.mp4` | korean language, korean hw esl student loan calculation | 韩语，韩国hw esl学生贷款计算 |
| `OuPEacoosis_00:00:00.000_00:00:09.409.mp4` | an image of a french language letter with the words apprend a cire les chiffres, 1, 2 and | 一张法语信件图片，上面写着“apprend a cire les chiffres”（学习书写数字），以及1、2和 |

### 7. 语言污染的**误报**（3.3% → ~0.3% 的由来）

**选取规则**：**修复后**仍命中比例旗标的记录。修复前有 327 条，其中 15 条人工抽检 15/15 是正确译文（保留了品牌名/型号）

| clip | 中文译文 | 英文源 | 命中旗标 |
|---|---|---|---|
| `QkLf4dqrWrE_00:09:50.920_00:09:59.920.mp4` | Essence 提供上门服务 | есеен доли сервис ире удет | zh_mostly_english |
| `7ipoOwru4XA_00:11:56.600_00:11:57.360.mp4` | Roblox Ota Oti Oti | roblox ора оти оти | zh_mostly_english |
| `1Mfr5pGSmCM_00:01:58.367_00:02:05.067.mp4` | Hobby Consolas 的标志 | the logo for hobby consollas | zh_low_cjk_ratio, zh_mostly_english |

> 📌 **修复前被误报的典型形态**（人工抽检的 15 条，逐字摘自 `review_conflicts.tsv`）：
>
> | 中文译文 | 英文源 | 为什么被误报 |
> |---|---|---|
> | `华硕 P8Z77-V BM 主板` | `asus p8z77-v bm motherboard` | CJK 占比 4/15=0.27 < 0.4 —— 而它连 ASUS 都译成了「华硕」 |
> | `Windows 10 开始菜单` | `windows 10 start menu` | 型号名不该翻，却拉低了 CJK 比 |
> | `Sun Cool 官方网站` | `sun cool official site` | 同上 |

---

## 三、工程类问题

### 8. 画面本身不可用（黑屏/花屏/纯色，0.49%）

**选取规则**：`video_readable=false` —— 全库 49 条。注意 `better_zh` 应为空（模型看不到画面就写不出字幕），可作自洽性检查

| clip | 英文源 caption | better_zh | VLM 建议动作 |
|---|---|---|---|
| `ybhN89tKcbI_00:05:22.923_00:05:23.824.mp4` | a black background with an airplane flying in the air | 空 | drop |
| `XqACXtN5twQ_00:13:37.520_00:13:40.040.mp4` | the black background with an animal in it | 空 | drop |
| `ldqKRrNQCvg_00:00:05.800_00:00:06.680.mp4` | a purple image with plants in the background | 空 | drop |

---

## 四、v0.2 修复素材：`better_zh` 长什么样

### 9. RECAPTION 桶的候选字幕（原文 vs 重写）

**选取规则**：`zh_score≤2` 且 `sig_zh` 最低的前 N 条 —— 即最该被重写的那批。⚠️ **这些候选的质量从未被评估**（caption-quality-report §5 的 P0 硬门），此处仅供**肉眼预览形态**，不构成质量证据

| clip | 原中文字幕 | better_zh 候选（≤30 字） |
|---|---|---|
| `sZ1tpezSfgc_00:02:37.200_00:02:39.400.mp4` | lululemon虚拟性感女孩身穿蓝色上衣 | 短发人士身穿蓝色衬衫，低头后抬头说话 ／ 屏幕上显示搜索框，输入' Samuel Bendahan' ／ 室内场景，背景有绿植和家具 |
| `xOTVi9EM7bA_00:00:41.375_00:01:39.499.mp4` | 俄罗斯警察使用丰田汽车在城市巡逻 | 画廊内男子演讲，背景有警察与汽车的影像叠加 ／ 男子在艺术展现场发言，画面中出现警察巡逻场景 ／ 演讲者与摄影师在画廊，背景影像显示警察与丰田车 |
| `WW8icbd0XHo_00:00:23.267_00:00:34.433.mp4` | 汽车旅馆 | 男子在黑板前讲解经济数据图表 ／ 讲解者手势指向上升曲线，讨论油价与出口关系 ／ 黑板上有手写数字和俄文经济分析文字 |

---

> **怎么核对**：`clip` 列是 `youtube_id_起始时间_结束时间.mp4`，在 §6 的 Gradio 工具里搜这个名字即可对着画面看。
