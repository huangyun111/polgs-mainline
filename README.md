# PolGS Mainline

偏振感知 Gaussian / surfel 三维重建研究代码，基于 [PolGS++](https://github.com/PRIS-CV/PolGS_plus)。本仓库保存 **PolGS++ 主干 + G 几何统计 + Image-score 有限预算增密分配** 的源码快照（2026-11-17）。

## 方法分工

- **参数优化**：使用原有 common losses，包括偏振监督；Image-score 分支不把 G 加到 Adam 更新梯度。
- **结构资格与需求**：使用 common + G 的 joint 统计。
- **父点排序**：将 DSSIM 像素误差通过原生 alpha 贡献回投到 Gaussian，在结构操作窗口中逐点取最大值，再分配 clone / split 名额。
- **结构执行**：保留原生出生、剪枝、opacity reset 和 Adam 行为。39,000 是这组研究对照的容量设置。

## 代码结构

| 目录 | 内容 |
| --- | --- |
| `backbone/` | 实际实验使用的 PolGS++ 主干，包含训练、渲染、场景读取、损失和原生 CUDA 子模块 |
| `mainline/image_score/` | Image-score 主分支、G 统计路由、容量控制及提取/验收代码；同一实现也保留 Joint-update 对照 |
| `mainline/shared_warmup/` | 共同预热、深度共视邻接和梯度/统计分离的必要源码 |
| `mainline/independent_capacity/` | 独立容量控制消融及协议 |
| `extensions/geometry_depth/` | G 所需的几何深度反传 CUDA 源码及补丁 |
| `evaluation/` | 完整三角表面距离、采样、法线和统一评价源码 |
| `runtime_reference/` | 实验运行时 rasterizer 的 Python 接口原样快照 |
| `docs/` | 已完成 Image-score 对照的协议与结果记录 |
| `SOURCE_MANIFEST.json` | 每份原始源码的来源、字节数与 SHA-256 |

主干环境安装与数据格式见 [上游说明](backbone/README.md)。源码包含三个 CUDA 子模块和 GLM 头文件/内联实现；编译产物需要在目标 Linux/CUDA 环境重新构建。

## 研究脚本的运行边界

这是实际运行代码的保存版本。`mainline/` 中的脚本仍保留原实验目录、配置、共同预热状态和扩展加载路径；不能直接在新机器执行便得到完整复现。迁移时须配置数据路径、输出目录、共同预热与共视文件，并编译、连接匹配的原生 rasterizer 和几何深度扩展。`evaluation/` 同样保留原实验目录布局，需要另行提供 GT 和冻结的配准文件。

本次上传验证了源码一致性、Python 语法和敏感凭据扫描，没有重新训练或在新环境验证数值复现。数据集、GT、模型权重、Mesh、大数组、编译缓存、连接凭据和调度/迁移工具均不包含在仓库中。

## 来源与许可

上游作者、论文引用和项目说明原样保存在 [backbone/README.md](backbone/README.md)。保留现有源码中的版权声明；Inria Gaussian Splatting 相关代码的完整许可见 [licenses/Gaussian-Splatting-LICENSE.md](licenses/Gaussian-Splatting-LICENSE.md)，相关 CUDA 模块目录也附有该许可。GLM 的原始许可见 [copying.txt](backbone/submodules/diff-gaussian-rasterization/third_party/glm/copying.txt)。各第三方组件继续遵循其自身许可，本仓库没有为它们重新授予许可。
