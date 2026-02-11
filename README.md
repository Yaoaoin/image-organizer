# 图片整理器 v4.0（CLI + GUI）

一个按图片内容自动分类的小工具，会将图片整理到：

- `people`（人物）
- `scenery`（风景）
- `animals`（动物）
- `vehicles`（交通工具）
- `food`（食物）
- `buildings`（建筑）
- `objects`（其他物品）
- `unknown`（低置信度或低匹配分）

## v4.0 更新

- 识别策略升级：从“只看 Top-1 标签”升级为“融合 Top-K 预测 + 语义匹配打分”，降低误分类。
- 分类逻辑升级：增加标签标准化、词级重叠匹配、精确匹配加权，提升类别命中率。
- 置信机制升级：新增 `--min-category-score`，与 `--min-confidence` 双阈值联合判定 `unknown`。
- GUI 增强：可直接配置 `Min category score`，更方便按数据集调参。

## 最简部署（推荐）

```bash
python run.py
```

> 首次运行会下载模型权重，耗时取决于网络环境。

## 手动安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## GUI 模式

```bash
python organize_images.py --gui
```

## CLI 模式

```bash
python organize_images.py <图片目录> -o <输出目录>
```

示例：

```bash
python organize_images.py ./photos -o ./sorted_photos --min-confidence 0.25 --min-category-score 0.12
```

可选参数：

- `--move`：移动文件（默认复制）
- `--min-confidence 0.2`：模型 Top-1 最低置信度（0~1），低于阈值放入 `unknown`
- `--min-category-score 0.12`：类别语义匹配最低分（0~1），低于阈值放入 `unknown`

## 分类原理（v4.0）

- 使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 对每张图片提取 Top-K ImageNet 标签及其置信度。
- 对标签做标准化，并与类别关键词进行：
  - 精确匹配
  - 子串匹配
  - 词级重叠匹配
- 按置信度加权累计类别分数，选取得分最高类别。
- 同时满足 `min-confidence` 和 `min-category-score` 才进入目标类别，否则归入 `unknown`。
