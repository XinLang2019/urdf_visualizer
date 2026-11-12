# 关键帧动作编排 / Keyframe motion authoring

[中文说明](#中文) · [English guide](#english)

![Keyframe motion editor / 关键帧动作编辑器](images/motion-editor.png)

## 中文

点击顶部“轨迹”或左侧“轨迹”标签。编辑器记录**独立关节 + 机身六自由度姿态**，不修改 URDF 文件。

1. 调整站姿，点击“记录姿态”，首帧时间为 `0 s`。
2. 在关节标签中拖动连杆、滑块或输入关节值；在轨迹标签中设置机身 XYZ、Roll / Pitch / Yaw，也可单击三维机身，在旁边的浮动编辑框中直接修改，并记录或更新关键帧。
3. 设置“下次记录时间”，再次记录。选择已有帧后，“更新帧”将当前三维姿态写回该帧；单纯拖动模型不会自动覆盖关键帧。
4. 点击时间轴菱形选择关键帧；拖动菱形或修改帧时间调整节奏。首帧固定在 0 秒。时间滑块用于预览，支持逐采样帧、变速、循环、撤销和重做。时间轴的 ⌖ 按钮适应整段轨迹；轨迹栏内 Ctrl + S 保存动作工程。
5. 用“保存工程”导出可再次编辑的 `.motion.json`；用“导出参考”按指定 Hz 输出 JSON，或 CSV + 元数据 JSON。

进入轨迹模式后，地面的位置和朝向保持固定，播放、调姿和适应视图不会使地面跟随机器人。进入模式时不会自动重新取景；可使用时间轴的 ⌖ 按钮查看整段动作。

### 插值与空翻

| 方式 | 行为 |
| --- | --- |
| 线性 | 每段匀速；关键帧处速度可能跳变 |
| 三次缓入缓出 | 每个关键帧速度为零，适合停顿动作 |
| 五次缓入缓出 | 每个关键帧速度和加速度为零，适合停稳过渡 |
| 保形三次 | 非均匀时间 PCHIP，连续通过关键帧，避免关节过冲；加速度不保证连续 |

每帧的插值设置作用于**该帧到下一帧**。连续通过模式应应用于相邻各段；混用插值可能造成端点导数不连续。

默认姿态为连续 RPY，支持 `0° → 180° → 360°`，或 `0° → −360°`，保留空翻方向和圈数。连续关节也保留累积角度。SLERP 使用四元数最短路径，`0° → 360°` 不会产生整圈旋转；选择 SLERP 时，保形三次的姿态通道使用三次缓入缓出的 SLERP，关节和位置仍使用 PCHIP。

+Z 向上的模型可生成前 / 后空翻**运动学示意**，然后分别调整起跳、腾空收腿、落地等关键帧。模板只编排机身位姿，保留当前关节姿态，不是动力学求解出的可执行空翻。

### 导出约定

- `schema: urdf-viewer.motion.v1`：可编辑工程，包含模型签名、关键帧、插值和采样设置。导入需匹配当前模型名称、根连杆、独立关节名称、类型及顺序，并按当前 URDF 限位校验。
- `schema: urdf-viewer.reference.v1`：采样参考。`frames` 每行包含 `time`、`base_position`、`base_rpy`、`base_quaternion`、`joint_positions`，以及机身线 / 角速度、线 / 角加速度、关节速度与加速度。
- 时间为秒，平移为米，旋转为弧度；四元数顺序为 **xyzw**，相邻采样的四元数保持符号连续。
- 机身坐标位于 **URDF 根连杆的父坐标系**，未乘以浏览器的向上轴显示旋转。RPY 对应 `Rz(yaw) Ry(pitch) Rx(roll)`。角速度为父坐标系表达的空间角速度，**不是 RPY 的直接差分**。
- `metadata.independent_joints` 明确数组关节顺序；移动关节单位为米，其余为弧度。mimic 关节不独立导出，可由 URDF 约束还原。内部 floating / planar 关节暂不支持记录。
- `metadata.ground_plane` 给出 URDF 父坐标中的地面法向量与偏移，满足 `normal · position = offset`。根坐标高度不自动归零；接入仿真器时需对齐坐标轴、地面高度及关节名称。
- 采样包含最后一个关键帧时刻；总时长不整除采样间隔时，最后一段间隔更短。导数由插值函数解析计算，在线性拐点等不可微位置采用所在段的单侧值。
- `diagnostics` 给出采样得到的关节峰值速度及 URDF 速度超限关节。位置遵守 URDF 限位，速度超限不会自动修改动作节奏。
- 循环只是回到起点播放，**不自动修复首尾位置 / 速度接缝**。有限角度关节的相同姿态、连续旋转的圈数和机身四元数需分别判断。

输出是用于轨迹跟踪、模仿学习的**运动学参考**，未计算接触时序、逆动力学、碰撞、执行器力矩或落地可行性。播放和预览会暂停原静力学标签，避免把腾空动作解释为静止支撑。

## English

Open **Motion** from the toolbar or left workspace. Each keyframe stores independent joint positions and the base's six-degree-of-freedom pose without changing the URDF.

1. Set a pose and record the first key at `0 s`.
2. Adjust links or joint sliders in Joints; edit base XYZ and Roll / Pitch / Yaw in Motion, or click the 3D chassis to edit them beside the model and record or update a key.
3. Set the next recording time and record again. Select a key and use **Update key** to replace its pose; posing the robot alone does not overwrite a key.
4. Select or drag timeline diamonds, or edit key times. The first key remains at zero. Scrub, step by one sample, adjust playback speed, loop, undo, or redo. Use ⌖ to fit the whole trajectory; Ctrl + S inside Motion saves the project.
5. Save `.motion.json` for further editing. Export sampled references as JSON or CSV with a metadata JSON sidecar at your chosen Hz.

Motion mode holds the ground position and orientation fixed during posing, playback, and fitting the view. Entering the mode does not automatically refit the camera; use the timeline’s ⌖ button to fit the entire motion.

### Interpolation and flips

| Mode | Behavior |
| --- | --- |
| Linear | Constant segment speed; velocity may jump at keys |
| Cubic easing | Zero velocity at every key |
| Quintic easing | Zero velocity and acceleration at every key |
| Shape-preserving cubic | Nonuniform PCHIP, continuous passage and no joint overshoot; acceleration may jump |

A key's mode controls its **outgoing segment**. Use consistent modes across neighboring segments for derivative continuity.

Unwrapped RPY retains turns, e.g. `0° → 180° → 360°` or `0° → −360°`. Continuous joints retain winding too. Quaternion SLERP uses the shortest path, so `0° → 360°` does not create a flip. In SLERP mode, cubic orientation uses eased SLERP with zero endpoint angular velocity, while positions and joints retain PCHIP.

For +Z-up models, the front/back flip sketches animate base pose and preserve the current joints. Author takeoff, tuck, and landing joint poses afterward. These sketches are kinematic examples, not dynamically solved flips.

### Reference data

- Editable projects use `urdf-viewer.motion.v1`. Imports must match the current model, root, independent joint names, types, and order; joint limits come from the current URDF.
- Sampled references use `urdf-viewer.reference.v1`. Each frame includes time, base position / RPY / quaternion, joint positions, base linear / angular velocity and acceleration, and joint velocity / acceleration.
- Units are seconds, meters, and radians. Quaternion order is **xyzw**, with continuous signs between samples.
- Base coordinates use the **URDF root's parent frame**, before the viewer's up-axis rotation. RPY means `Rz(yaw) Ry(pitch) Rx(roll)`. Angular velocity is spatial, expressed in the parent frame, rather than differentiated Euler angles.
- `metadata.independent_joints` defines array order; prismatic positions are meters. Mimic joints are excluded and reconstructed from URDF constraints. Internal floating / planar joints are not supported for recording.
- `metadata.ground_plane` defines `normal · position = offset` in the URDF parent frame. Root height is not automatically ground-normalized. Map axes, ground height, and joint names to your simulator.
- Sampling includes the exact final key; the final interval may be shorter than `1 / fps`. Derivatives are analytical and piecewise, with one-sided values at nonsmooth knots.
- `diagnostics` reports sampled peak joint speeds and URDF velocity-limit violations. Joint position limits are enforced; speed violations do not retime the motion automatically.
- Loop playback restarts at zero; it does **not** repair pose or velocity discontinuities at the seam.

Exports are **kinematic reference motions** for tracking and imitation learning. Contact schedules, inverse dynamics, collisions, actuator torques, and landing feasibility are not solved. Motion preview pauses quasi-static overlays.

## 方法参考 / Method references

- [Modern Robotics: polynomial time scaling](https://modernrobotics.northwestern.edu/nu-gm-book-resource/9-1-and-9-2-point-to-point-trajectories-part-2-of-2/)
- [SciPy: SLERP and shortest-path rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Slerp.html)
