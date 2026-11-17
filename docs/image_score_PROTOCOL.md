# 第二阶段：固定S-cap的两条强对照

只新增StandingRabbit官方20视角、seed0的Image-score和Joint-update，复用第一阶段S-cap/N-cap及旧强N/S/SB。两臂从同一个官方WARMUP1000完整状态（43308点）继续到固定15000原生输出，新增各13999、总14999 Adam。本轮不迁场景、不扫描、不扩seed，不预设双统计限制增长更好，不自动进入第三阶段。

| 版本 | 参数更新 | 结构资格/需求 | 父点排序 |
| --- | --- | --- | --- |
| Image-score | common | S-cap的joint | 像素误差回投 |
| Joint-update | common+G | S-cap的joint | joint平均XY |

G、common、输入/分辨率/偏振权重、冻结top1、原生clone/split/Adam/reset和提取评价保持与第一阶段一致。G已含0.03权重，不重复乘权重。两臂独立原生prune，然后必要时按activated opacity降序/索引升序裁至39000；1100操作后启用上限，初态保留。joint原生资格XY>=1e-4、clone signed scaling<=1e-7/raw opacity<=2及大小尺度规则不变；需求为joint合格数，剩余容量按Kc=floor(Q*c/(c+s)), Ks=Q-Kc分配（不超限时全取）。不读另一模型数量或旧quota表。独立轨迹资格和实际容量可以随后分叉，不宣称逐操作同数量。

## Image-score冻结定义

正文依据：[RDGS §3.2](https://arxiv.org/html/2404.06109v1#S3.SS2)，实际读取全文的公式E^v_i=sum_x e_v(x)w_iv(x)、E_i=max_v E^v_i及辅助标量反传实现（2026-11-11）。本轮为RDGS式排序对照，不是完整RDGS复现；不引入它的opacity修正/衰减、增长百分比或误差资格阈值。

像素误差预先固定为e_v(x)=max(0,(1-mean_RGB SSIM_map(S0_pred,S0_obs)(x))/2)。SSIM使用原生11x11 Gaussian窗口、sigma1.5、C1=0.01²/C2=0.03²、相同zero padding。S0_pred为本次训练实际使用的rendered_image_with_ref（含漫反射与环境反射），S0_obs为本次训练相同mask/随机背景下get_gtStokes的S0；不混用diffuse-only颜色。全训练图像，和原photometric目标一样处理输入mask/背景，不读取几何GT。像素误差detach。

每个正式reference帧都评分；在两次100步结构操作之间，对每次观测的E^v_i逐元素max，不按可见数或总像素归一化，不平均视角。每窗包含原相机采样序列，不另做全视角扫描。共同1000初态处原生统计已reset，误差bank从0开始。bank与本模型原生/额外prune同步，原生生成后的统计reset时归零，快照包含bank和已见视角。

原生override_color设置为独立的零值辅助标量a_i复制至三通道，以黑背景渲染，aux=sum_x stopgrad(e_v(x))*render_aux[0,x]；autograd.grad(aux,a_i)正好得到sum_x e_v w_iv。几何、opacity、rotation和camera均经只读detached代理传入，辅助相机不update、不抽patch、不消耗RNG；只对辅助标量求导。native initial_stage=True仅用于读取线性颜色合成贡献；误差源仍是实际完整S0预测。辅助计算不加入训练loss，不改变主梯度、深度缓存或随机流，不改CUDA。评分只在joint合格池排序，永不进入1e-4资格阈值。并列按当前父索引升序。

## 验收与交付

正式首步验证common梯度原生oracle、joint统计公式、选定Adam梯度指针及G路由；Image-score同时验证DSSIM均值与原生SSIM一致、辅助评分不变更模型/Adam/统计/梯度/深度/RNG，三重叠Gaussian的VJP对独立one-hot渲染积分一致。实际前两次结构操作保存候选池/排序分数/父ID，独立CPU重排与额度复算；不重复已验证训练前缀smoke。

保存每步容量、完整Gaussian×forward-step/Adam暴露、出生/预算触发和耗时/显存；Image-score额外逐步记录像素误差与回投GPU/墙钟成本、额外前反传次数。每1000完整快照及15000原生输出前后状态保留；15000输出后的未使用增密另计。只在必要恢复/实际变更/异常做针对性检查，无整树大资产rehash。

后台有限队列为两训练→两份原生depth8/4线程Mesh→相同旧GT变换与100000面积样本×20261020/21/22完整评价→六距离数组复算/父点排序/容量账本审计→报告与小包。CD、A/C、法线、覆盖/远尾、实际容量和成本同时比较；单位GT坐标×1000非标定mm。单训练seed，三面积重复不是训练重复。质量接近时结合容量/成本/复杂度判断，不凭微小均值强行升级。

所有训练worker独立process session，控制器退出不会连带训练退出；PID/命令/收据和进度共同用于存活检查，失败立即记录并隔离，禁止把旧progress当运行证据。训练后任务不依赖本地连接。GPU0异常排除，实时连续采样后放置在健康低利用且显存充裕卡，初始每臂8192MiB预算加约2458MiB余量；Image-score新增辅助渲染的实际峰值在正式首批核验。原S-cap/N-cap不重跑，PISR及其他用户任务不触及。
