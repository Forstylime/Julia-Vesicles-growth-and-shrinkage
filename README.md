# Multiple_SAV

本项目用于 [简要描述你的项目目的，例如：求解多重谱分析的数值模拟]。

## 项目结构
- `src/`: 核心源代码，包含各种求解器、类型定义和工具函数。
- `test/`: 单元测试文件。
- `scripts/`: 运行入口脚本（如 `Main.jl`）。
- `results/`: 存放模拟输出的图片或数据文件。

## 如何使用

### 1. 环境准备
本项目使用了 Julia 的包管理系统。在终端中进入项目根目录，运行：

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate()'