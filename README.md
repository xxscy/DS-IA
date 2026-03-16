# DS-IA: Device State Intent Analysis Framework

本项目实现了三种智能家居控制框架在HomeBench数据集上的应用和对比实验。

## 📁 项目结构

```
.
├── code/                           # IA框架和基准框架实现
│   ├── ablation_study.py          # IA框架消融实验（核心）
│   ├── model_test.py              # 基准框架测试
│   ├── eval_basic.py              # 基础评估脚本
│   ├── eval_with_type.py          # 按类型分类评估
│   ├── system.txt                 # 系统提示词
│   └── example.txt                # Few-shot示例
│
├── dataset/                        # HomeBench数据集
│   ├── test_data.jsonl            # 测试数据
│   ├── train_data_part1.jsonl     # 训练数据（第1部分）
│   ├── train_data_part2.jsonl     # 训练数据（第2部分）
│   ├── valid_data.jsonl           # 验证数据
│   └── home_status_method.jsonl   # 家庭环境配置
│
└── SAGE-HomeBench-GitHub/          # SAGE框架实现
    ├── sage_brain/                 # SAGE核心模块
    │   ├── coordinator.py          # 协调器（ReAct Agent）
    │   ├── homebench_tool.py       # 工具实现
    │   └── prompts.py              # Prompt模板
    ├── run_experiment_qwen.py      # SAGE实验运行脚本
    ├── evaluate_by_category.py     # SAGE评估脚本
    └── requirements.txt            # 依赖包
```

## 🎯 三种框架对比

### 1. IA框架（Intent Analysis Framework）
**位置**: `code/ablation_study.py`

**核心思想**: 多阶段处理流程
- **Stage 1**: 意图分析（Intent Analysis）- 验证操作可行性
- **Stage 2**: 代码生成（CodeGen）- 生成控制代码
- **Stage 3**: Grounding验证 - 细粒度验证和修正

**支持的实验配置**:
```bash
# 完整框架（Model IA + CodeGen + Grounding）
python code/ablation_study.py --experiment full_framework

# Oracle IA（使用真实标签，理论上限）
python code/ablation_study.py --experiment oracle_ia

# 无意图分析（CodeGen + Grounding）
python code/ablation_study.py --experiment no_ia

# 无Grounding验证（Model IA + CodeGen）
python code/ablation_study.py --experiment no_grounding

# 基线（仅CodeGen）
python code/ablation_study.py --experiment baseline
```

**特点**:
- 支持选择性Grounding（基于置信度）
- 支持Few-shot和Zero-shot
- 详细的性能统计（tokens、延迟等）
- 支持自定义测试数据

### 2. SAGE框架（Tool-based Agent Framework）
**位置**: `SAGE-HomeBench-GitHub/`

**核心思想**: 基于工具调用的ReAct Agent
- 使用LangChain实现
- 工具：`query_devices`, `execute_command`, `get_device_state`
- 动态查询设备信息，不预先提供环境描述

**运行方式**:
```bash
cd SAGE-HomeBench-GitHub
python run_experiment_qwen.py
```

**特点**:
- 基于ReAct模式的推理
- 动态环境查询
- 支持多轮交互

### 3. 基准框架（Baseline Framework）
**位置**: `code/model_test.py`

**核心思想**: 直接LLM生成
- 最简单的prompt工程方法
- 支持多种模型（Qwen, LLaMA, Mistral, Gemma）
- 支持Few-shot和RAG

**运行方式**:
```bash
# Few-shot测试
python code/model_test.py --model qwen --few_shot --test_type few_shot_5k

# Zero-shot测试
python code/model_test.py --model qwen --test_type zero_shot
```

## 🚀 快速开始

### 环境配置

```bash
# 安装依赖
pip install torch transformers langchain langchain-community tqdm

# 下载模型（示例：Qwen2.5-7B-Instruct）
# 将模型放置在 models/Qwen2.5-7B-Instruct/ 目录下
```

### 运行IA框架实验

```bash
# 完整框架测试（500样本）
python code/ablation_study.py --experiment full_framework --sample_size 500

# Oracle IA测试（理论上限）
python code/ablation_study.py --experiment oracle_ia --sample_size 500

# 无IA测试
python code/ablation_study.py --experiment no_ia --sample_size 500
```

### 评估结果

```bash
# 基础评估
python code/eval_basic.py code/results/ablation_full_framework.json

# 按类型评估（VS, IS, VM, IM, MM）
python code/eval_with_type.py code/results/ablation_full_framework.json
```

## 📊 评估指标

- **EM (Exact Match)**: 完全匹配率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1**: F1分数

### 测试数据类型分类

- **VS (Valid Single)**: 单一设备有效指令
- **IS (Invalid Single)**: 单一设备无效指令
- **VM (Valid Multi)**: 多设备有效指令
- **IM (Invalid Multi)**: 多设备无效指令
- **MM (Mixed Multi)**: 多设备混合指令（有效+无效）

## 📝 数据集格式

### test_data.jsonl
```json
{
  "id": "test_001",
  "home_id": "home_123",
  "input": "Turn on the light in living room",
  "output": "'''living_room.light.turn_on()'''",
  "type": "normal"
}
```

### home_status_method.jsonl
```json
{
  "home_id": "home_123",
  "home_status": {
    "living_room": {
      "light": {
        "state": "off",
        "attributes": {"brightness": {"value": 0}}
      }
    }
  },
  "method": [
    {
      "room_name": "living_room",
      "device_name": "light",
      "operation": "turn_on",
      "parameters": []
    }
  ]
}
```

## 🔬 实验结果

结果文件保存在 `code/results/` 目录下：
- `ablation_*.json` - 实验结果（JSONL格式）
- `ablation_*_perf.json` - 性能统计
- 日志文件保存在 `code/logs/` 目录

## 📄 引用

如果使用本项目，请引用：

```bibtex
@article{homebench2024,
  title={HomeBench: A Benchmark for Smart Home Control},
  author={...},
  journal={...},
  year={2024}
}
```

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

## 📜 许可证

本项目遵循MIT许可证。
