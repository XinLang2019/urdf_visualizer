# URDF Viewer

**中文** | [English](README.en.md)

浏览器中的机器人 URDF 工作台，集模型预览、关节调节、三维测量、代码编辑与静力学分析于一体，无需安装前端依赖。

![URDF Viewer：左侧关节控制、中间三维视图、右侧代码编辑](docs/images/workbench.png)

## 快速开始

在项目目录启动静态网页服务：

```bash
python3 -m http.server 8080
```

访问 [localhost:8080](http://localhost:8080)，选择内置示例，或加载包含 URDF、网格与纹理的完整机器人文件夹。请勿双击 `index.html` 以 `file://` 打开，否则浏览器会阻止内置模型的资源请求。

推荐 Chrome / Edge。保存源文件需要在 localhost 或 HTTPS 下授权文件写入；内置示例首次保存时需选择磁盘上对应的原文件。

GitHub Pages 与普通静态服务可直接加载 URDF / MJCF / 文本 USDA 并互转；全部在浏览器本地处理，无需安装 Python，不上传模型。二进制 USD 为可选本地服务功能，见[格式支持范围](docs/formats.md)。

## 主要功能

- **模型与结构**：URDF、MJCF、USD 文件 / 文件夹导入、拖拽加载、结构拓扑；内置机器人及 MJCF / USD 示例。
- **格式转换**：URDF ↔ MJCF ↔ USD，打包网格下载，附转换差异说明；在线版 USD 导出为文本 `.usda`。[使用方法](docs/formats.md)。
- **STEP 装配导出**：URDF 导出 `.step`，保留连杆层级与零位装配位置；后台分块生成，支持进度与取消，网格保留为三角面几何。[使用方法](docs/step-export.md)。
- **Mesh Simplifier**：选择已加载的 STL / OBJ，按保留比例或目标面数简化，对比预览后覆盖源文件或下载。[使用方法](docs/mesh-simplifier.md)。
- **关节控制**：拖动连杆或滑块调姿，支持限位和角度 / 弧度切换；双击连杆定位关节和代码。
- **辅助显示**：视觉、碰撞、质心、等效惯量长方体、坐标轴与关节轴；支持明暗主题和中英文界面。
- **三维测量**：两点或两连杆原点之间的距离及 XYZ 分量，提供最多 5 个几何吸附候选点。
- **多模型**：左侧模型列表切换或同时显示，自动排列；“位置”设置相对 XYZ 距离（m），点击三维模型切换编辑对象，分别保留草稿与关节姿态。
- **编辑与保存**：显示 URDF、MJCF、文本 USDA 原格式代码，校验、重新加载，修改后直接保存到对应源文件。
- **静力学分析**：实时关节力矩、支撑反力和力臂；设置坡度与摩擦系数，一键贴合轮子 / 足端并计算机身姿态。

| 三维距离与 XYZ 分量 | 斜坡接触与关节力矩 |
| --- | --- |
| ![两连杆原点的三维测量示例](docs/images/measurement.png) | ![Go2 四足贴合斜坡与静力学参数](docs/images/slope-statics.png) |

## 常用操作

| 操作 | 功能 |
| --- | --- |
| 左键拖动空白 / 右键拖动 / 滚轮 | 旋转 / 平移 / 缩放视图 |
| 拖动连杆 / 点击轴旁数值 | 调节关节 / 输入目标角度或位移 |
| 双击连杆 | 定位关节卡片与对应 link 代码 |
| Shift + 点击两点 | 表面点测量与几何吸附；Alt 暂停吸附 |
| Ctrl + 点击两个连杆或结构节点 | 测量连杆原点间距离；松开恢复关节拖动 |
| Ctrl + Enter / Ctrl + S | 应用代码到模型 / 保存源文件 |
| F / Esc | 适应模型视图 / 清除测量与选择 |

## 静力学分析

调好站姿后开启“受力分析”。斜坡分析需设置坡度、坡向和静摩擦系数，再点击“一键贴合斜坡”；保持关节值不变，根据接触几何求解整机姿态。点击力矩标签可查看力与力臂，再次点击恢复全部标签。

结果是重力与地面接触作用下的静态估算，多接触载荷采用最小平方分配；不包含动态惯性、传动效率或滚阻。轮轴结果表示保持静止所需的轴端力矩。无法同时接地、支撑不稳或摩擦不足时会提示，不输出有效站立力矩。

## 验证

修改前端后运行 `python3 tools/version_assets.py`，更新静态资源版本，避免 GitHub Pages 混用旧缓存。浏览器测试会检查版本是否匹配。

安装 Python 3 和 Chrome / Chromium 后运行浏览器回归检查：

```bash
python3 tests/browser_smoke.py
```

可通过 `CHROME_BIN` 指定浏览器路径。

## 参考与贡献

- [**urdf-loader**](https://github.com/gkjohnson/urdf-loaders) — Robust URDF loading for Three.js；本项目复用的 URDF 加载与关节交互基础。
- [**robot_viewer**](https://github.com/fan-ziqi/robot_viewer) — 机器人可视化工作台与交互设计参考。
- [**meshoptimizer**](https://github.com/zeux/meshoptimizer) — 浏览器网格简化算法，MIT 许可。

感谢以上项目的作者及开源社区贡献者。欢迎通过 [Issues](https://github.com/XinLang2019/urdf_visualizer/issues) 反馈问题，或提交 Pull Request 改进功能、文档和示例；修改功能时请附复现步骤与验证结果，并保留上游项目及模型资源的署名和许可说明。
