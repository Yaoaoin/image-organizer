# 图片整理器 v3.0（CLI + GUI）

一个按图片内容自动分类的小工具，会将图片整理到：

- `people`（人物）
- `scenery`（风景）
- `animals`（动物）
- `vehicles`（交通工具）
- `food`（食物）
- `buildings`（建筑）
- `objects`（其他物品）
- `unknown`（低置信度）

## v3.0 更新

- 扩展识别类别：在原有人物/风景/物品基础上，新增动物、交通工具、食物、建筑。
- 提升类别命中逻辑：支持更灵活的关键词匹配，减少“识别不出来”的情况。
- 重做桌面 GUI 视觉风格：深色卡片式布局，更清晰的层级和状态展示。

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
python organize_images.py ./photos -o ./sorted_photos
```

可选参数：

- `--move`：移动文件（默认复制）
- `--min-confidence 0.2`：最低置信度（0~1），低于阈值放入 `unknown`

## 分类原理

- 使用 `torchvision` 预训练 `MobileNetV3-Small`。
- 对每张图片做 ImageNet 标签预测。
- 通过关键词映射到 `people/scenery/animals/vehicles/food/buildings/objects`。
- 若预测置信度低于 `--min-confidence`，则归入 `unknown`。
