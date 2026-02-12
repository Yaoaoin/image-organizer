# 图片整理器 v5.0（CLI + GUI）

一个按图片内容自动分类的小工具，会将图片整理到：

- `people`（人物）
- `scenery`（风景）
- `animals`（动物）
- `vehicles`（交通工具）
- `food`（食物）
- `buildings`（建筑）
- `objects`（其他物品）

> 默认不再生成 `unknown`，尽量将图片归入可用类别；如需严格模式可开启 `--allow-unknown`。

## v5.0 更新

- 识别策略升级：从单模型升级为**双模型融合**（MobileNetV3 + ResNet50），提升对人物、建筑、常见物品的区分能力。
- 类别词库扩展：补充建筑与物品关键词，减少“物品/建筑”互相混淆。
- 默认策略调整：默认关闭 `unknown` 路由，低分样本会回落到最佳类别；可通过 `--allow-unknown` 开启严格模式。
- 阈值优化：`--min-confidence` 默认 0.16、`--min-category-score` 默认 0.08，更适配复杂场景。
- GUI 增强：新增 `Allow unknown folder (strict mode)` 选项，便于快速切换策略。

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

示例（默认不使用 unknown）：

```bash
python organize_images.py ./photos -o ./sorted_photos
```

示例（严格模式，低分样本进 unknown）：

```bash
python organize_images.py ./photos -o ./sorted_photos --allow-unknown --min-confidence 0.16 --min-category-score 0.08
```

可选参数：

- `--move`：移动文件（默认复制）
- `--min-confidence 0.16`：模型 Top-1 最低置信度（0~1）
- `--min-category-score 0.08`：类别语义匹配最低分（0~1）
- `--allow-unknown`：启用严格模式，低分样本进入 `unknown`

## 分类原理（v5.0）

- 使用 `torchvision` 预训练 `MobileNetV3-Small + ResNet50`。
- 对每张图片分别提取两个模型的 Top-K ImageNet 标签并加权融合。
- 对融合后的标签做标准化，并与类别关键词进行：
  - 精确匹配
  - 子串匹配
  - 词级重叠匹配
- 按置信度加权累计类别分数，选取得分最高类别。
- 默认将图片归类到最佳类别；当开启 `--allow-unknown` 时，只有同时满足阈值才归入目标类别，否则进入 `unknown`。
