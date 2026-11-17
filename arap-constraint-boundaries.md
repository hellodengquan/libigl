# ARAP 形变的约束配置与预计算复用边界

> 分析对象：本仓库 `include/igl/arap.{h,cpp}`、`arap_dof.{h,cpp}`、`min_quad_with_fixed*.{h,impl.h}`、`kkt_inverse.cpp`、`fit_rotations.cpp`、`polar_svd3x3.cpp`/`svd3x3.h`、`cotmatrix_entries.cpp`、`project_isometrically_to_plane.cpp`。
> 所有"实测"结论都在仓库外（`/tmp`）用独立小程序复现：直接 `#include <igl/arap.h>` 等头文件做 header-only 编译，Eigen 取自同级 `0045-pmp-library-2-aafab641/external/eigen-5.0.1`。测试网格若无特别说明为两个互不连通的 4 顶点薄片（两块 `2` 个三角形，`F = {0,1,2},{0,2,3},{4,5,6},{4,6,7}`）。
> 未在源码中实现、也没有测试覆盖的推断，一律标注 **【未实现】** 或 **【推断】**。

---

## 0. 两套实现，先分清

| | `igl::arap_*` | `igl::arap_dof_*` |
|---|---|---|
| 未知量 | 顶点位置 `U`（`#V×dim`） | 每个权重函数的仿射变换 `L`（`#handles×dim×(dim+1)`） |
| 约束 | 顶点索引 `b` + 位置 `bc`，作为**硬约束**进入线性系统 | 线性等式 `A_eq * L = B_eq`，可外加 `fixed_dim` 部分固定 |
| 每帧求解 | 局部-全局迭代（`max_iter` 次），每次一个稀疏分解的回代 | 固定的若干次局部-全局迭代，全部是**稠密小矩阵乘**（`M_FullSolve`、`CSolveBlock1`） |
| 预计算内容 | `CSM`、`K`、`Q=-L`（或加质量项）、`min_quad_with_fixed` 的分解 | `CSM_M`、`M_KG`、`Q`、`Mass_tilde`，以及 `recomputation` 产出的 `M_FullSolve`/`CSolveBlock1` |
| 适用 | 交互式自由形变、参数化 | 骨骼/手柄驱动、需要亚毫秒级更新的场景 |

`arap.cpp:163` 把问题交给 `min_quad_with_fixed_precompute(Q, b, /*Aeq=*/空, /*pd=*/true, ...)`：**ARAP 主路径永远走"固定值约束 + LLT"分支**（`min_quad_with_fixed.impl.h:222`），不涉及 QR/等式约束分支。

---

## 1. 多连通部件的两类"解不出来"

必须区分这两件事，它们的现象、判据、处置完全不同。

### 1.1 类型 A：约束不足 → 自由刚性运动（解集是一个流形）

ARAP 能量对每个连通部件的整体刚体运动不变。因此若某部件没有足够的约束把刚体自由度钉死，最优解**不唯一**：

- 3D 形变：刚体群 `SE(3)` 6 维。实测（单块薄片，`max_iter=200`，同一条 `bc`，仅换初值 `U0`）：

| 约束点数 | 两个初值得到的解之差 `max|U_A−U_B|` | 结论 |
|---|---|---|
| 1 个点（且目标点移动） | `1.26` | 剩余 5 维 |
| 2 个点 | `0.84` | 剩余绕连线旋转 1 维 |
| 3 个**不共线**点 | `4.5e−8` | 刚体框架唯一 |
| 3 个**共线**点 | `0.84` | 仍剩绕该直线旋转 1 维 |

- 2D 参数化（`dim=2`，`V.cols()=3`）：`SE(2)` 3 维。实测：

| 约束点数 | 两初值之差 |
|---|---|
| 1 个点 | `0.88` |
| 2 个点 | `0.0` |

  换句话说：2D 参数化**每个部件至少要 2 个不同的固定点**（或等价地，边界上两处被固定）才能消除整体旋转；3D 形变需要 3 个不共线点。注意"点"是硬约束，**共线/重合不算**。

- 这些自由度对**整体可解性**的影响不是"多解但都合法"这么温和。因为实现把约束当硬约束、把其余部分交给一个**可能奇异的稀疏 Laplacian 系统**，实测有三种行为，且都只由数值主元是否恰好为 0 决定：

| 约束分布（两个部件） | `arap_precomputation` | 实际结果 |
|---|---|---|
| `b={}` | **true**（QUu 有 2 个零特征值，最小 5.3e−17） | 求解返回，两部件按初值整体乱走；`max|U−V|=2.24` |
| `b={0}`（只钉部件 0） | **true**（nullity 1） | 部件 1 自由，被拖走且随初值变化 |
| `b={0,1,2}`（钉满部件 0） | **false** | 若调用方忽略返回值继续 `arap_solve`：`solve` 仍返回 true，部件 1 全部输出 `0` |
| `b={0,1,2,3}`（钉满部件 0） | **true**（nullity 1） | 部件 0 正确；部件 1 由自由运动决定（实测随约束点/初值漂移） |
| `b={0,4}`（每部件 1 点） | true | 两部件各自只剩旋转，结果依赖 `U0` |
| `b={4,5,6}`（钉满部件 1，4 顶点中留 1） | **false** | 未固定顶点为 0 |

  零特征值在 LLT 里被舍入成 `+5e−17` 就"成功"，被算成精确 0 就失败（Eigen `SimplicialCholesky_impl.h:374-377`：`d <= 0` 才置 `NumericalIssue`）。所以 **`arap_precomputation` 的返回值不是"约束是否足够"的判据**，只是"这次数值分解有没有刚好踩到非正主元"。

- 结论：约束不足**不是**报错条件。要让实现按预期工作，调用方必须自己保证每个部件都有独立的刚体锚定；否则输出可能是零、可能是任意平移旋转，而且**bc 依然被精确满足**（见 §3）。

### 1.2 类型 B：几何退化 → 求解数值失效

这类问题与约束个数无关，来自输入几何或初值：

| 退化 | 源码位置 | 实测现象 |
|---|---|---|
| 零面积三角形 | `cotmatrix_entries.cpp:48-56` 用 `dblA` 作除数 | `precompute=true`，`solve=true`，`U` 含 **NaN**（余切权重 inf/nan 进入 Q） |
| 近平板/近退化（`eps=1e−9`） | 同上 | `U` 含 NaN |
| 初值使协方差全零（如 `U0=0` 且无 `bc`） | `arap.cpp:206` `S /= S.array().abs().maxCoeff()` | **0/0 → NaN**；`solve` 仍返回 true |
| 初值为均匀平移且无 `bc` | 同上（S 仍为 0） | 实测不 NaN，但结果无意义（退化后的方向保留原值） |
| `V` 含 NaN | `cotmatrix` 传播 | `precompute=true`，`solve` 后 `U` 含 NaN |
| `bc` 含 NaN | `arap.cpp:195-198` 直接写入 U | `U` 全 NaN，`solve=true` |
| 2D 参数化、镜像/反向的边界映射 | `fit_rotations.cpp:95-101` 只对**拟合出的旋转**强制 `det>0`，不约束顶点位置 | 8/8 三角形面积变负，**不报错** |
| 2D 参数化、三角形退化 | `project_isometrically_to_plane.cpp:36-43` 逐三角形展开 | 展开失败则 NaN 进入 `CSM`/`K` |
| `dim=2` 但 `F` 是四面体 | `project_isometrically_to_plane.cpp:26` 仅 `assert(F.cols()==3)` | Debug 断言；**Release 静默算出一组无意义结果**（实测 `CSM.cols=8, K.cols=4`，solve=true） |

所以"几何退化"的典型信号是 **NaN/Inf、符号翻转、输出塌缩为 0**，而不是返回值。

### 1.3 部件之间会互相影响，而且不止一种方式

1. **约束不跨部件传递**：Laplacian 是块对角，钉部件 0 对部件 1 没有任何锚定作用（上表）。部件 1 的"解"完全由奇异系统的任意分量决定。
2. **分组旋转会跨部件耦合**：`data.G` 非空时，`arap.cpp:229-244` 用 `eff_R.block(...) = R.block(0, Rdim*data.G(r), ...)` 把**组**的旋转分发给每个顶点/单元。若两个部件落在同一组，移动其中一个会带动另一个。实测（两部件、全部 `G=0`）：部件 0 被强行移动 `0.129`，部件 1 被拖走 `0.6`；`G` 每部件一组时部件 0 位移回到 `8.9e−7`。**默认 `G` 为空时不耦合**，一旦设置分组就应至少按部件分组。
3. **全局协方差归一化会跨部件耦合（比例敏感）**：`arap.cpp:206` 用全体 `S` 的最大绝对值归一化，而 `arap.cpp:215` 调用 `fit_rotations(S, /*single_precision=*/true, R)`。`single_precision=true` 走的是 `polar_svd3x3`→`svd3x3`，而 `svd3x3.h:28` 明写 *"this will not work correctly for double precision"*（内部 epsilon 是按 float 调的）。
   实测（部件 1 相对部件 0 放大 `s` 倍，静止位姿，理想旋转应为 `I`）：

| 缩放比 `s` | 归一化后部件 0 的 `|S|` | 拟合出的旋转与 `I` 的偏差 |
|---|---|---|
| 1 | 8.2e−1 | 2.1e−7 |
| 10 | 8.2e−3 | 5e−8 |
| 100 | 8.2e−5 | 1.5e−3 |
| 1000 | 8.2e−7（≈6× float eps） | **0.120** |
| 1e5 | 8.2e−11 | 旋转完全错（`R0` 变成 90° 换轴） |

   端到端后果（两块网格，每块 1600 顶点，均是 `max_iter=20`）：

| 缩放比 | 全局一个 `ARAPData`：小部件的刚度误差 | 每部件一个 `ARAPData` |
|---|---|---|
| 1 | 3.9e−7 | 3.7e−7 |
| 10 | 1.6e−6 | 3.7e−7 |
| 100 | 2.9e−3 | 3.7e−7 |
| 1000 | **0.98**（完全不等距） | 3.7e−7 |

   这条是"多部件网格"最容易被忽略的坑：**不是约束问题，是尺度差 + 单精度快速 SVD**。`arap_dof` 在 `__SSE__` 下走 `fit_rotations_SSE`（`arap_dof.cpp:766-767`）会转成 float，同等风险；在无 SSE 时走 `fit_rotations(S,false,R)` 是双精度 JacobiSVD，实测相同尺度差下无此问题。**【未实现】**：没有"按部件归一化"或"自动退回双精度"的保护。

---

## 2. 二维参数化与三维形变：前提差异

源码把二者统一在 `dim` 参数里，但前提并不相同。

| 前提 | 三维形变 `dim=3, V.cols()=3` | 二维参数化 `dim=2, V.cols()=3` |
|---|---|---|
| 进入的代码路径 | `flat=(3-3)==1`→false，直接用 `V,F` | `flat==true`→`project_isometrically_to_plane`（`arap.cpp:57-67`） |
| 参考几何 | 原始三维形状 | 逐三角形**等距展开**到平面得到的 `plane_V`；`CSM`、`K` 由展开网格构造后再左乘/右乘 `ref_map`（`arap.cpp:98-102,140-144`） |
| 旋转拟合 | `fit_rotations`（3×3，要求 `dim==3`，`fit_rotations.cpp:27`） | `fit_rotations_planar`（2×2，第三行列补单位，`fit_rotations.cpp:95-101`） |
| 能量默认值 | 三角形 → `SPOKES_AND_RIMS`（`arap.cpp:81`） | 三角形 → `ELEMENTS`（`arap.cpp:84`） |
| 支持的单元 | 三角形/四面体；`SPOKES_AND_RIMS` 对四面体 `assert(false)`（`arap_linear_block.cpp:146`） | **仅三角形**（`project_isometrically_to_plane.cpp:26`）；四面体只在 Release 下"能跑"，结果无意义 |
| 刚体自由度 | 6 | 3 |
| 最少锚定点 | 3 个不共线点 | 2 个不同点 |
| "约束满足"能否代表结果正确 | 不能：不可等距的硬约束会以拉伸形式消化（实测把一条边从 1 拉到 2，`bcres=0`、边长误差 **1.0**） | 更不能：边界形状与内在度量冲突时一律以大幅扭曲消化（把 4×4 网格边界钉到半径 0.5 / 1 / 2 的圆，`bcres=0`，边长误差 **0.65 / 0.29 / 1.0**） |
| 三角翻转 | 无检测 | 无检测；边界映射反向时 8/8 三角形负面积（实测） |
| 尺度 | 由 `bc` 决定 | 由 `bc` 决定；仅给形状不给 `bc`（教程 503 的做法）时靠动力学项软锚定，整体平移旋转仍自由 |
| 不连通部件 | 每部件独立 6 自由度 | 每部件独立 3 自由度；教程 503 的"无 bc + 动力学"下，两块互不连通的部分会**各自**停在不同的刚体位置（实测两初值误差 `1.07`） |

要点：

- **2D 参数化的"约束满足"是构造性成立的，不传递任何质量信息。** 因为 `bc` 在每次迭代开头被逐行写入（`arap.cpp:195-198`），`min_quad_with_fixed_solve` 也把 `Z(known)=Y` 精确赋值（`min_quad_with_fixed.impl.h:444-446`）。所以哪怕结果是镜像的、扭成麻花的，`bcres` 仍是 0。
- **2D 参数化对"固定点不足"更敏感**：少一个点就整体转；而且因为 2D 旋转拟合只在旋转上强制 `det>0`，位置层面的翻转是自由的。
- **3D 形变对约束的"几何兼容性"更敏感**：约束点若摆成与等距形变不相容的位姿，ARAP 只能拉伸；此时没有"不可行"错误，只有刚度指标变差。
- 2D 下 `fit_rotations_planar` 的 `assert(dim==2)` 被注释掉了（`fit_rotations.cpp:72`），意味着传错维度不会在 Debug 下报错，只会读越界/算错。**【未实现】** 维度校验。

---

## 3. 预计算复用判定

`ARAPData` 里各部分依赖的输入不同，必须按字段拆开看。

### 3.1 复用矩阵（`arap_*`）

| 变化项 | 可复用？ | 依据 |
|---|---|---|
| `bc`（每帧拖动） | **可复用**，这是设计用途 | `arap_solve` 只读 `data.b/CSM/K/solver_data`；实测 20 帧复用正常 |
| `max_iter` | **可复用**，每次 `arap_solve` 现读 | `arap.cpp:191`；实测 `max_iter=1` 循环 10 次 == `max_iter=10` 一次（末次差 0） |
| `U` 初值 | 可复用，只是热启动 | 非动力学时只当初始猜测 |
| `f_ext`（动力学外力） | **必须在 precompute 之后设置** | `arap.cpp:159` precompute 会把 `f_ext` 重置为 0；实测 precompute 前设的外力被清零 |
| `h`（时间步） | 实测**无影响** | `Dw=(1/ym)h²`、`Q += M/ym`、`Dl=(1/ym)(M(-U0-h·vel) - h²f_ext)`，`h·vel = U−U0`，`h` 整体约掉；实测 5 帧下 `h=1/0.1/3` 差 0 |
| `ym`（刚度） | **不可复用** | `Q` 在 precompute 里含 `M/ym`（`arap.cpp:155-157`），而 `Dl` 每次按当前 `ym` 计算（`arap.cpp:258-259`）；改 `ym` 后两者不一致，实测差 `2.39` |
| `with_dynamics` | **不可复用**（UB） | 切换后 `data.M` 为空，`arap.cpp:249-250` 仅 Debug 断言；Release 实测段错误 |
| `b` 的**索引值** | **不可复用** | 只比较长度：`arap.cpp:175` `assert(data.b.size()==bc.rows())`。换一组同长度索引而不重算，`U` 会在旧约束位置上满足 `bc`，新位置的约束被完全忽略（实测 `U(0)=V(0)`、`U(4)=V(4)`，`U(1)`/`U(5)` 的残差 0.99） |
| `b` 的**长度** | 不可复用 | 同上，长度不一致 Debug 断言，Release **堆越界**（ASAN 复现） |
| `b` 含**重复索引** | **不可复用**（静默错误） | `data.known` 直接保存（`min_quad_with_fixed.impl.h:70`），而 `unknown_mask` 会去重（同文件 74-88），导致 `preY`/`Y` 的行数与未知数不匹配；实测 `b={0,4}` vs `{0,4,4}` 结果差 `3.74`，且某个顶点被压到 0 |
| `dim` | 不可复用 | 无任何检查；实测 3D 数据上按 2D 求解 → ASAN 堆越界 |
| `V` 的尺度/形状 | **不可复用** | `CSM`、`K`、`Q` 全部按 `V` 编译进去。两倍缩放后的新静止形状仍会在旧尺度下求解：`bc` 用新坐标给，结果对旧 `V` 的边长误差 1.4e−7、对新 `V` 的误差 0.5（即输出是旧尺寸） |
| `V,F` 的**顶点数/拓扑** | 不可复用 | 无检查；新增顶点后 ASAN 堆越界，`U.rows` 与 `data.n` 不一致时直接越界写 |
| `F` 的三角形连接（同顶点数） | 不可复用 | `CSM/K/Q` 内嵌邻接 |
| `energy` | 不可复用 | `eff_energy` 是 **precompute 内的局部变量**（`arap.cpp:73-93`），`data.energy` 仍是 `DEFAULT`；事后改 `data.energy` 实测差 0（完全被忽略），不会报错 |
| `G` | 不可复用 | 见下节，尤其是 `ELEMENTS` 下会被**就地改写** |

### 3.2 `G` 与重复 precompute 的陷阱

`arap.cpp:117-133`：`ELEMENTS` 能量 + 非空 `G` 时，会把**逐顶点的 `G` 就地改写成逐面**（`data.G = GG`，长度 `#F`）。于是：

- 同一个 `ARAPData` 上第二次调用 `arap_precomputation` 时，`data.G` 已是逐面数组，却被当作逐顶点使用（`GF.col(j) = data.G(F.col(j))` 取到越界的面编号）。实测（8 顶点 / 4 面，`G` 从 `0 0 0 1 2 2 2 3` 变成 `0 2 0 0`）：Debug 断言 `IndexedView` 越界；Release **静默得到另一组分组**。
- 因此"更换几何后重算预计算"这条路径，在用过 `ELEMENTS + G` 之后是**不安全**的，除非调用方自己保存并恢复原始逐顶点 `G`。**【未实现】**：`data.G` 的语义没有版本标记，也没有 `G_dim` 之类的伴随字段。

### 3.3 可以复用的边界总结

安全复用（"同一套 `(V,F,dim,energy,G,b)`，只改 `bc`、`U0`、`max_iter`"）：

```
arap_precomputation(V,F,dim,b,data)   // 一次
for each frame:
    enforce user's bc
    arap_solve(bc,data,U)             // data.b / CSM / K / solver_data 全部原样
```

一旦触碰 `V`、`F`、`dim`、`energy`、`G`、`b`、`with_dynamics`、`ym` 中任意一项，必须重新 `arap_precomputation`，而且**建议换一个新的 `ARAPData` 对象**（`ARAPData` 因内含 `SimplicialLLT` 等分解对象**不可拷贝**，实测编译失败；要放进容器请用 `std::unique_ptr<ARAPData>` 或 `std::deque`/`std::list`，`vector` 的移动插入也不满足要求）。

### 3.4 `arap_dof` 的复用边界

- `arap_dof_precomputation(V,F,M,G,data)` 依赖 `V,F,M,G,energy`；`LbsMatrixType`（稀疏/稠密）也是编译期固定的。
- **约束集/固定维度变化时**用 `arap_dof_recomputation(fixed_dim, A_eq, data)`：它重做 `kkt_inverse` 与本征 `M_FullSolve`，实测 900 顶点 / 10 手柄时约 0.5 ms，是设计好的"每帧可调约束"入口。
- `data.with_dynamics` 切到 true 后**必然失败**：`arap_dof.cpp:828-829` 引用了 `data.L0`/`data.Lvel0`，但全仓库从未给它们赋值（`arap_dof.h:207,212` 只有声明）。Debug 断言 `lhs.cols()==rhs.rows()`；Release 段错误（实测）。
- `data.effective_dim` 在**真正的 3D 网格上是未初始化的**：`arap_dof.cpp:78-88` 只在 `dim==3 且 z 全零` 或 `dim!=3` 时写入。构造/默认构造都不赋值（`arap_dof.h:231-238`）。它同时控制 `use_lu`（`arap_dof.cpp:583`）和用 2D/3D 旋转拟合（`arap_dof.cpp:761-770`）。实测把哨兵值 12345 留在里面会原样保留，读它就是 UB。**【未实现】**：这一点应视为使用 `arap_dof` 前必须手动设置的字段。
- `L0` 参数同时承担"初始猜测"和"固定变换取值"两个角色（`arap_dof.cpp:702` `B_eq_fix_SSCALAR = L0SSCALAR(data.fixed_dim)`）。`fixed_dim` 里的条目**必须同时出现在 `A_eq` 的行里**（头文件 `arap_dof.h:99-103` 的 NOTE）。实测：`fixed_dim={0}` 但 `A_eq` 只有 3 行位置约束 → `Lout(0)=0.99` 而不是指定的 `2`；补一行把 `Lxx` 约束住后精确变成 2。

---

## 4. 失败/未收敛时，调用方能否判断"约束已满足"

**结论：不能。现有 API 的返回值既不代表收敛，也不代表约束有意义。**

源码证据：

1. `arap_solve` **无条件返回 true**（`arap.cpp:293`），且完全不检查 `min_quad_with_fixed_solve` 的返回值（`arap.cpp:278-281`）。
2. `min_quad_with_fixed_solve` 本身在回代后也不检查 `llt.info()`（`min_quad_with_fixed.impl.h:476-493`；`info()` 只在 **precompute** 阶段检查，同文件 228-241）。
3. `arap_precomputation` 只把 `min_quad_with_fixed_precompute` 的 bool 透传（`arap.cpp:163-164`）。该 bool 仅在 ①有未知量断言（78-112，Release 下不生效）、②`Aeq` 的 QR 失败、③分解出现非正/零主元 时为 false。**它不检查约束是否足以固定刚体自由度**。
4. `SolverStatus`/`SOLVER_STATUS_*` 只被 `active_set` 使用（`active_set.cpp:55,177,304`）；`arap` / `min_quad_with_fixed` 完全不用。

实测对照：

| 场景 | `precompute` | `solve` | 实际 |
|---|---|---|---|
| 峰值约束不足，但 LLT 侥幸成功 | true | true | `bc` 精确满足，结果随初值任意变化 |
| 同样峰值约束，LLT 踩到 0 主元 | **false** | true（如果忽略返回值） | 未约束顶点输出 **0** |
| `max_iter=0` | true | **true** | 内层 while 一次都没执行，`bc` **根本没写进 U**（实测 `U(4)=(0,0,0)`，期望 `(4,0,0)`） |
| 零面积单元 | true | true | `U` 含 **NaN** |
| `bc` 含 NaN | true | true | `U` 全 NaN |
| `A_eq` 秩亏（`arap_dof`） | true | true | 3D 路径（`use_lu`）输出 **NaN**；2D 路径输出残差 `2.5` 的"最小二乘"结果 |
| `arap_dof` 不可行约束（同一条边被要求长度 4） | true | true | 两个端点各自满足位置约束，边长误差 3 |
| `arap_dof` `with_dynamics` | true | true→崩溃 | Debug 断言 / Release 段错误 |

因此调用方能得到的唯一"硬"信息是：

- `precompute` 的 bool（但见 §1.1，**假阴性/假阳性都有**）；
- `arap_dof_update` 的 tol **完全无效**：`arap_dof.cpp:36` 定义了 `IGL_ARAP_DOF_FIXED_ITERATIONS_COUNT`，把 `tol` 参数在签名里注释掉（637-643），并把收敛判断整体编译掉（685-695、859-862）。实测 `tol=0` 与 `tol=1e9` 输出逐位相同。
- `arap_solve` 没有迭代次数、残差、能量出参。

### 4.1 调用方必须自己做的检查（推荐实现）

```cpp
// 1) 预计算是否"够用"（模型侧一次性检查，不要只看返回值）
//    每个连通部件至少要有独立的刚体锚定：
igl::vertex_components(F, C);                    // 每顶点部件号
for (int c = 0; c < C.maxCoeff()+1; ++c) {
  std::vector<int> bc_in_c;                      // b 中属于部件 c 的索引
  // 3D：要求 >=3 且不共线（面积 > 0）；2D：要求 >=2 且不重合
  // 不满足就不要把它当成"可形变"配置
}
if (!arap_precomputation(V,F,dim,b,data)) { /* 必须中断，不能继续调用 arap_solve */ }

// 2) 每帧检查（廉价）
bool ok = true;
ok &= U.allFinite();                             // 退化几何/零协方差都会以 NaN 出现
double bcres = 0;                                // 硬约束残差；正常应为 0
for (int i = 0; i < b.size(); ++i)
  bcres = std::max(bcres, (U.row(b[i]) - bc.row(i)).cwiseAbs().maxCoeff());
//   注意：bcres==0 是构造性成立的，只能用来抓 NaN/维度错配，不能当"形变合理"

// 3) 几何质量（真正的判据，见 §5）
```

如果只需要"约束被写进去了吗"这个弱保证：注意 `max_iter==0` 时它也不成立，所以 `max_iter` 至少为 1。

---

## 5. 选择条件、判据与收敛依据

### 5.1 "多部件"的配置选择（推荐）

| 场景 | 推荐 | 理由 |
|---|---|---|
| 部件之间独立交互（各自拖动、各自受力） | **每部件一个 `ARAPData`**，用 `vertex_components(F)` 切出 `(V_c, F_c)` 后各自 precompute | 每部件独立锚定，天然消除"自由部件"；实测与全局求解在尺度一致时结果一致（刚度误差同为 3.9e−7），但**不受全局归一化影响**（`s=1e3` 时 3.7e−7 vs 0.98） |
| 部件之间刚性联动（同一物体拆分、同一手柄驱动） | 全局一个 `ARAPData`，但 `G` 必须**按部件分组**（或留空） | 共享 `G` 组会把旋转耦合过去（实测位移 0.129 泄露） |
| 部件尺度差异 > ~100× | **强制拆分**，或改用双精度旋转拟合 | 见 §1.3-3，单精度快速 SVD 的误差上界 |
| 需要亚毫秒级每帧更新、且形变可用少量仿射变换表达 | `arap_dof`（每部件至少一个手柄；手柄权重不要只覆盖部分部件） | 实测 900 顶点/10 手柄：precompute 21 ms、recompute 0.5 ms、update 0.04 ms |
| 2D 参数化 | `dim=2`，每部件至少 2 个固定点；边界最好给完整一圈 | 少一个点就整体转；`dim=2` 不要喂四面体 |

补充的实测开销（22 500 顶点、44 402 三角形、桌面 CPU）：

| 操作 | 耗时 |
|---|---|
| `arap_precomputation`（LLT 分解） | ~62–97 ms |
| `arap_solve`，1 次迭代 | ~9 ms |
| `arap_solve`，10 次迭代 | ~87 ms |
| 复用预计算的 20 帧（10 iter/帧） | 87 ms/帧 |
| 每帧重新 precompute 的 20 帧 | 151 ms/帧 |

即：**预计算 ≈ 10 次迭代的成本**，所以"每帧重建"在交互里不可接受；而"每帧 1–3 次迭代 + 热启动"是可行的。

### 5.2 交互开销 vs 几何正确性的选择条件

设每帧允许预算为 `B`，单次迭代成本 `c`（≈ 顶点数 × 常数，稀疏 LLT 回代），预计算成本 `P ≈ 10c`：

- 可交互：`max_iter · c ≤ B`。实测 `max_iter=1` 时刚度误差 ~0.09（大幅拖动）、`max_iter=3` ~1.5e−3、`max_iter=10` ~2e−7（`bc` 移动 `1.5` 单位）。因此**拖动过程中用 `max_iter∈[1,3]` 热启动**，**松手/导出时再用 `max_iter` 20–100 冷启动一次**。
- 判定"热启动是否够好"：连续调用两次同样 `bc` 的 `arap_solve`，比较 `|U_new − U_old|`（实测收敛后残差 2.5e−6；10 次外层迭代后 1.0e−5）。这个量不需要新 API。
- 约束集变化（用户增删固定点）：**必须重算 precompute**（§3.1）。只改 `bc` 值：复用。
- 拓扑/几何变化（换模型、重网格）：重建 `ARAPData`（不要复用，Release 下是越界写）。

### 5.3 判据（可直接用的量）

对每个连通部件 `c`（用 `vertex_components(F)` 划分），定义：

1. **约束残差**（必须为 0，否则是数值失败/维度错配）
   `E_bc(c) = max_{i∈b∩c} ‖U_i − bc_i‖∞`
   注意：只要求解路径正常执行，这个量恒为 0，**不能单独用来判断形变质量**。

2. **刚性保持**（ARAP 的核心指标，推荐主判据）
   对原始网格的每条边（或每个三角形的三个边长）：
   `E_rigid(c) = max_e | ‖U_i − U_j‖ / ‖V_i − V_j‖ − 1 |`
   实测分级：`max_iter=1` → 0.09（大幅拖动）；`3` → 1.5e−3；`5` → 2.3e−5；`10` → 2e−7；`≥20` → 2e−7（进入双精度噪声地板）。
   建议阈值：交互中 `E_rigid ≤ 1e−2`，交付 `E_rigid ≤ 1e−5`。

3. **尺度/翻转检查**（2D 参数化必查，3D 也建议）
   - 每个三角形有符号面积 `A_f`；若 `A_f ≤ 0` 说明局部翻面（实测边界反向映射时 8/8 为负，`solve` 仍返回 true）。
   - 总面积 `Σ A_f` 与目标的比值：ARAP 不控制全局尺度（`bc` 才控制），用于发现"整体缩放"。
   - 2D 参数化额外看面积畸变 `max_f |A_f/A_f^rest − 1|`；把边界钉到不同半径的圆时，该量从 0.29 到 1.0 变化，而 `E_bc` 恒为 0。

4. **部件解耦检查**（多部件）
   正交性：若两部件理应互不影响，比较"只移动部件 A 的约束"与"同时移动 A、B"两次求解中部件 B 的位移，应 ≤ 数值噪声。不满足说明 `G` 分组或全局归一化把它们耦合住了。

### 5.4 收敛判断的依据

- 源码里 **`arap_solve` 没有任何收敛判据**：`while(iter < data.max_iter)`（`arap.cpp:191`），固定次数退出，没有残差/能量比较。**唯一正确的收敛依据只能由调用方在外部构造**：
  1. 固定 `bc`，把上一步结果作为 `U0` 再调用一次 `arap_solve`（例如 `max_iter=1`），测 `ΔU = ‖U_new − U_old‖∞`；`ΔU` 单调下降到双精度地板即收敛（实测 10 次外层迭代后 1.0e−5，之后平台在 ~1e−6）。
  2. 同时观察 `E_rigid`：它比 `ΔU` 更贴近期望语义（ARAP 的局部-全局迭代在能量上单调，但实现没有断言这一点）。
- `arap_dof_update` 的 `tol` **不可用**（§4）。若需要收敛判断，只能在外部循环调用 `arap_dof_update(..., max_iters=1, ...)` 并测 `ΔL`。这意味着 **arap_dof 目前只有"固定次数"模式**。**【未实现】**
- `min_quad_with_fixed` 的线性求解阶段没有"迭代收敛"概念（直接法）；Eigen 的 `SimplicialLLT` 只在主元非正时报 `NumericalIssue`（`SimplicialCholesky_impl.h:374-377`），而 ARAP 的 `Q` 在约束不足时是半正定的，零主元可能被舍入成 `+5e−17` 而不报错（实测多处）。所以 **precompute 成功 ≠ 系统非奇异**。

---

## 6. 未实现的保证（清单）

以下都不能从现有 API 得到，文档/命名也不能暗示它们存在：

1. **约束齐备性检查**：没有任何代码检查每个连通部件是否被独立锚定；`arap_precomputation` 的返回值只反映数值分解是否恰好失败。
2. **收敛判断与迭代信息**：`arap_solve` 固定 `max_iter` 次后无条件 `return true`（`arap.cpp:191,293`），不返回残差/能量/迭代数。
3. **约束满足的真实性**：`bc` 是硬写入的，不检查是否与等距形变相容；不可行时静默拉伸（3D）或扭曲/翻面（2D）。
4. **`arap_dof` 的 `tol`**：被 `IGL_ARAP_DOF_FIXED_ITERATIONS_COUNT`（`arap_dof.cpp:36`）在编译期去掉，收敛分支被 `#ifndef` 排除（685-695、859-862）。
5. **`arap_dof` 的动力学**：`data.L0`/`data.Lvel0` 从未赋值（`arap_dof.cpp:828-829` 只读），`with_dynamics=true` 不可用。
6. **`ArapDOFData::effective_dim` 在真 3D 网格上未初始化**（`arap_dof.cpp:78-88`、`arap_dof.h:231-238`）。
7. **`kkt_inverse` 无失败返回**（返回 `void`，`kkt_inverse.h:52-54`）：LU 路径直接 `LM.inverse()`（`kkt_inverse.cpp:42-45`），奇异时返回 inf/nan；SVD 路径把小于 `FLOAT_EPS` 的奇异值置零（26、59-68 行），**静默丢弃约束**，仅在 stdout 打印一行计数。
8. **重复/冲突约束的去重或可行性判定**：`data.known` 原样保存（`min_quad_with_fixed.impl.h:70`），重复索引导致 RHS 被重复累加（实测结果差 3.74）；`A_eq` 秩亏在 `arap_dof` 中不报错。
9. **三角翻转 / 自交 / 局部单射检测**：全仓库的 ARAP 路径都没有。
10. **预计算失效检测**：`b` 只比较长度（`arap.cpp:175`），`V/F/dim/energy/G` 变化无任何版本或尺寸校验；`U.rows() != data.n`、`bc.rows() != b.size()` 在 Release 下是越界读写（ASAN 已复现）。
11. **`2D` 参数化对四面体的支持**：`project_isometrically_to_plane` 仅 Debug 断言（`project_isometrically_to_plane.cpp:26`），Release 静默产生无意义结果。
12. **`SolverStatus` 在 ARAP 链路中的使用**：`SolverStatus.h` 的枚举只被 `active_set` 使用；`arap`/`min_quad_with_fixed` 不暴露任何状态码。
13. **`ARAPData` 的可拷贝/可移动**：内含 `SimplicialLLT` 等分解对象，拷贝构造被删除（实测编译失败），存入 `std::vector` 亦不满足 Cpp17MoveInsertable。
14. **`ELEMENTS` + 逐顶点 `G` 的重复 precompute 安全性**：`arap.cpp:129` 就地改写 `data.G` 为逐面，二次调用即越界（Debug 断言 / Release 静默错分组）。
15. **按部件归一化 / 自动双精度回退**：`arap.cpp:206,215` 的全局归一化 + 单精度 `svd3x3`（`svd3x3.h:28` 自述不支持 double）没有保护；部件尺度差 ~1e3 即可让小的那个部件旋转错乱（实测刚度误差 0.98 vs 3.7e−7）。

---

## 7. 附：验证代码的位置与复现方式

验证程序位于仓库外、不进入版本库的 `/tmp/arapcheck/`（`t1.cpp` … `t62.cpp`），编译方式：

```bash
clang++ -std=c++17 -O1 -w -DNDEBUG \
  -I<repo>/include \
  -I<...>/0045-pmp-library-2-aafab641/external/eigen-5.0.1 \
  tXX.cpp -o tXX
```

- 加 `-DNDEBUG` 观察 Release 行为；去掉 `-DNDEBUG` 观察断言位置。
- 加 `-fsanitize=address` 可复现"尺寸/索引失配"的堆越界（`arap_solve` 中 `U.row(...)` 写入、`bc` 行数与 `b` 不符、拓扑变化后继续复用）。
- 文中所有"实测"数值均来自这些程序在同一台机器上的输出；数值随网格与平台略有变化，但结论（例如"`bc` 恒精确满足"、"部件尺度差导致旋转拟合失效"、"`arap_solve` 无条件返回 true"）由源码逻辑决定，不随平台改变。
