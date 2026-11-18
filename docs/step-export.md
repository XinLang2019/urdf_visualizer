# STEP 装配导出 / STEP assembly export

**中文** · [English](#english)

加载包含 URDF 与网格资源的完整文件夹，在文件栏的“格式转换”中选择 **STEP (.step)**，点击“转换并下载”。解压 ZIP 后，用 CAD 软件打开 `model.step`；几何已内嵌，无需附带网格文件。GitHub Pages 可直接使用，无需 Python，不上传模型。

- **装配关系**：机器人 → 根连杆 → 各级子连杆；连杆内每个 visual 是独立几何件，重复网格复用零件定义。无视觉几何时使用该连杆的 collision，避免重复导出两套几何。
- **位置与单位**：使用编辑器中 URDF 的关节零位、关节 origin、visual / collision origin 与 mesh scale。保留原始 Z 向上坐标，米换算为毫米；滑块姿态、显示开关、视图居中和贴地不改变导出装配。
- **几何**：STL / OBJ / DAE 与基础几何导出为三角平面。可确认封闭、流形且无重叠壳的组件导出为实体；开放、非流形、重叠或可能含内腔的壳保留为曲面，不自动补洞。退化三角面跳过并记录。
- **范围**：输出 STEP AP214 静态装配，不恢复原始参数化 CAD 曲面，不生成可驱动的 CAD 关节约束，不迁移惯量、贴图和材料。关节名称、类型与轴向附在装配说明中。暂不支持 STEP 导入或 STEP 转回 URDF。

`conversion-report.json` 记录单位、零位约定、独立几何定义数、三角面数量与转换说明。大网格的 STEP 文件可能较大，可先使用 Mesh Simplifier；不会在导出时自动降低精度。

## 开发验证

仅测试需要 CAD 内核；网页用户无需安装：

```bash
.venv-formats/bin/python -m pip install cadquery-ocp-novtk
URDF_TEST_BROWSER_FORMATS=1 CHROME_BIN=/usr/bin/microsoft-edge .venv-formats/bin/python tests/step_export_browser.py
```

真实浏览器下载 ZIP，再使用 Open CASCADE 读取并核对：装配层级和名称、每个几何实例的世界坐标、毫米单位、URDF 零位、共享零件、实体有效性及开放网格。`URDF_STEP_TEST_GO2=1` 追加内置 Go2 大模型检查。

参考：[Open CASCADE STEP 装配映射](https://github.com/Open-Cascade-SAS/OCCT/wiki/step)、[STEP faceted B-rep](https://www.steptools.com/docs/stp_aim/html/t_faceted_brep_shape_representation.html)。

## English

Load a complete URDF folder including its meshes. In **Files → Convert format**, choose **STEP (.step)** and download. Extract the ZIP and open `model.step` in a CAD application. Geometry is embedded; no external mesh files are needed. This works locally in your browser on GitHub Pages without Python or uploads.

The AP214 assembly retains the robot / link hierarchy, each visual's placement, joint origins and mesh scales. Repeated meshes share a part definition. Collision geometry is used only for links without visuals. Original Z-up coordinates are preserved and metres become millimetres. Export uses the editor's URDF **zero pose**, independently of sliders, visibility, centering or ground placement.

Meshes and primitives become triangular planar geometry. Confirmed closed manifold components with nonoverlapping shells become solids. Open, nonmanifold, overlapping or potentially nested shells remain surfaces; holes are not filled. Degenerate triangles are skipped and reported. Analytic CAD surfaces, live CAD joint constraints, inertia, textures and materials are not reconstructed. Joint names, types and axes are included in assembly descriptions. STEP import and reverse conversion are not supported.

The ZIP includes `conversion-report.json` with units, pose convention, unique geometry count, triangle count and limitations. Large mesh-derived STEP files may be large; Mesh Simplifier can reduce them explicitly before export. Export never silently simplifies geometry.

The development test above reads real browser downloads with Open CASCADE and independently checks named hierarchy, each instance's world coordinates, shared geometry, units, valid solids and open surfaces. Set `URDF_STEP_TEST_GO2=1` to include the bundled Go2 model. The CAD kernel is only a development test dependency.
