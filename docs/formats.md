# 模型格式 / Model formats

**中文** · [English](#english)

支持加载 URDF、MJCF（`.xml` / `.mjcf`）、USD（`.usd` / `.usda` / `.usdc` / `.usdz`），以及 **URDF ↔ MJCF ↔ USD** 六个方向的转换。MJCF 使用 MuJoCo 编译器解析，USD 使用 OpenUSD SDK；不会把二进制 USD 当成文本 XML。

## 启动

格式服务与网页使用同一个地址。首次安装依赖，后续只运行最后一行：

```bash
python3 -m venv .venv-formats
.venv-formats/bin/python -m pip install -r requirements-formats.txt
python3 model_server.py --port 8080
```

需要 Python 3.10+。默认自动使用 `.venv-formats/bin/python` 处理模型，也可通过 `--python /path/to/python` 指定环境。打开 [localhost:8080](http://localhost:8080)。原有的 `python3 -m http.server 8080` 仍可浏览、编辑 URDF，但没有 MJCF / USD 的解析转换接口。

## 使用

1. 文件栏中加载完整模型文件夹（包括 MJCF include、USD 引用层、网格与贴图），或选择内置的 MJCF / USD 双连杆示例。
2. MJCF / USD 自动生成 **URDF 工作副本**，可调关节、测量、编辑并查看静力学。原始文件保持不变；质量缺失时静力学会提示，转换器不会凭空补质量。
3. 文件栏的“格式转换”选择 URDF、MJCF 或 USD，点击“转换并下载”。使用当前编辑器中的代码，无须先重新加载；拖动滑块的当前关节角度不作为模型零位写入。
4. 解压 ZIP 后打开 `model.urdf`、`model.xml` 或二进制 `model.usd`，保持 `meshes/` 相邻。包中包含 `conversion-report.json`，记录单位与转换差异。

MJCF / USD 的编辑器按钮为“导出 URDF 包”，会一起打包工作副本及其网格；不会用 URDF 文本覆盖原始 MJCF 或 USD。原生 URDF 的“保存”仍覆盖已授权的源文件。

![MJCF / USD 加载、关节控制与格式转换](images/format-conversion.png)

## 支持范围与差异

| 数据 | 行为 |
| --- | --- |
| 刚体与关节 | 树形结构，固定、转动、连续转动、移动关节；自由根保留为浮动连接。保留关节轴、原点、限位、已知力矩与速度限值。 |
| 坐标与惯量 | 内部统一为米、千克、弧度、Z 向上；处理 USD 的单位、Y/Z 向上、关节两侧坐标、主惯量轴。 |
| 几何 | 盒、球、圆柱、三角网格；MJCF 胶囊与椭球、USD 基础几何转为三角网格。网格导出为 OBJ；材质保留常量 RGBA。 |
| 文件资源 | MJCF include / default、USD 引用层与 USDZ；引用文件必须位于上传文件夹中。共享网格按内容去重。 |
| 控制与扩展 | 控制器、传动、传感器、肌腱、动画和仿真器扩展不会等价迁移。URDF mimic 在 MJCF 中使用线性 equality，在 USD 中仅保存为 `urdf:*` 元数据。 |

转换不等于仿真器之间的动力学等价。未知的惯量、速度或电机参数会在说明中标记，不会伪造。MJCF 导出要求运动刚体具有正质量和有效惯量；多关节 MJCF 可展开为串联帧浏览，零质量运动帧不能直接回写为 MJCF。

球铰、通用 D6、闭环或多父约束、MJCF 插件与 asset-model attach、带缩放的 USD 刚体帧、含孔或超过四顶点面的 USD 网格，需要先在源工具中展开、烘焙或三角化。无法表示的结构会报错，保留上一个已加载模型。USD 细分面显示控制网格；贴图与复杂材质不会迁移。包含 URI / 网络或文件夹外绝对路径的依赖需要先收集到模型文件夹中。

单次上传上限 384 MiB / 1500 个文件；原生解析在独立进程中运行，超时会提示。相同服务支持两项并行转换。

## 验证

```bash
.venv-formats/bin/python -m unittest discover -s tests -p test_model_formats.py
.venv-formats/bin/python tests/model_formats_browser.py
python3 tests/browser_smoke.py
```

可设置 `CHROME_BIN=/usr/bin/microsoft-edge` 使用 Edge。浏览器测试实际上传 MJCF / 二进制 USD、下载六方向转换包并重载，检查关节限位、网格与副本导出。数值测试对比多个角度下的质心、惯量与几何位置。

参考：[MuJoCo XML](https://mujoco.readthedocs.io/en/stable/XMLreference.html)、[OpenUSD Physics](https://openusd.org/release/api/usd_physics_page_front.html)。

---

## English

Load URDF, MJCF (`.xml` / `.mjcf`), and USD (`.usd` / `.usda` / `.usdc` / `.usdz`), and convert in all six directions. The native readers are **MuJoCo** and **OpenUSD**, including binary USD support.

Create the `.venv-formats` environment, install `requirements-formats.txt`, and start `python3 model_server.py --port 8080` using the commands above. Python 3.10+ is required. The server automatically selects the environment; `--python` overrides it. Open [localhost:8080](http://localhost:8080). Ordinary static hosting still supports URDF without native conversion.

Load the complete model folder, including referenced layers / includes and meshes. MJCF / USD become explicit **URDF working copies** for the existing joint, measurement, editing and statics tools. The original native files are retained. **Export URDF ZIP** includes the working XML and its meshes; it does not overwrite the native source. Original URDF files retain their normal save-in-place behavior.

Under **Convert format**, choose a target and download the ZIP. Conversions use the current editor text, not slider positions as a new zero pose. Extract the package and keep the `meshes/` folder beside the native entry file. Read `conversion-report.json` for differences.

Supported interchange covers rigid-body trees, fixed / revolute / continuous / prismatic joints, floating roots, frames, limits, explicit masses, centers of mass and inertia. Units normalize to metres, kilograms, radians and Z-up. USD joint frames and principal inertia axes are transformed. Primitive shapes and meshes are supported; some shapes are tessellated, OBJ meshes are packed, and materials reduce to constant RGBA.

Controller, sensor, tendon, transmission, animation and simulator-specific semantics are not transferred. Mimic is a linear equality in MJCF and `urdf:*` metadata in USD. Conversion does **not** guarantee equivalent simulation dynamics. Missing physical parameters are reported. MJCF export requires positive mass / inertia on moving bodies, including any expanded joint frames.

Unsupported spherical / D6 / closed-loop constraints, MJCF plugins and asset-model attachments, scaled USD rigid frames, mesh holes and USD polygons with more than four vertices require flattening / baking / triangulation first. Unsupported inputs fail explicitly and retain the previously loaded viewer model. Subdivision meshes use their polygon cage. Texture maps / complex shaders are not transferred. All external resources must be included in the uploaded folder.

Uploads are limited to 384 MiB / 1500 files. Native parsing uses isolated worker processes with a timeout. Run the validation commands above; set `CHROME_BIN` for a specific browser.
