# GUI-Owl Data Pipeline Implementation

基于Wuying AgentBay SDK实现的GUI-Owl自演化轨迹数据生产工作流。

## 架构概述

该实现包含以下核心模块：

### 1. 环境基础设施 (environment.py)
- 管理云端虚拟环境池
- 支持Mobile、PC、Web三大平台
- 动态环境创建和管理
- 并行会话支持

### 2. Query生成器 (query_generator.py)
- **移动端**: DAG图结构建模、路径采样、指令自然化
- **PC端**: 原子操作技能链生成
- 自动难度评估

### 3. 轨迹收集器 (trajectory_collector.py)
- 异步轨迹收集
- 步骤级动作执行
- 状态捕获（截图、UI层次结构）
- 失败重试机制

### 4. 轨迹评估器 (trajectory_evaluator.py)
- **Step-Level Critic**: 步级评判（GOOD/NEUTRAL/HARMFUL）
- **Trajectory-Level Critic**: 双通道评估（文本+多模态）
- 共识机制确保质量

### 5. Query指导生成器 (guidance_generator.py)
- 为困难Query生成特定指导
- 动作描述和效果验证
- 关键步骤提取
- 错误模式识别

### 6. 数据管道 (data_pipeline.py)
- 自演化循环控制
- 批量处理优化
- 下游数据生成（Grounding、规划、语义）
- 结果导出和统计

## 使用方法

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 设置API密钥
export AGENTBAY_API_KEY="your_api_key"
```

### 移动端数据生产

```python
from gui_owl_pipeline import DataPipeline
from gui_owl_pipeline.config import PipelineConfig, Platform, OperatingSystem

# 配置
config = PipelineConfig(
    api_key="your_api_key",
    platform=Platform.MOBILE,
    operating_system=OperatingSystem.ANDROID,
    parallel_sessions=5,
    max_trajectory_length=30
)

# 初始化
pipeline = DataPipeline(config)
await pipeline.initialize()

# 运行
result = await pipeline.run_mobile_pipeline(
    app_name="MyApp",
    pages=pages,
    transitions=transitions,
    query_count=10
)

# 导出
pipeline.export_pipeline_data("./output")
```

### PC端数据生产

```python
# 运行PC端pipeline
result = await pipeline.run_pc_pipeline(
    software_name="TextEditor",
    skills=skill_list,
    query_count=10
)
```

## 运行示例

```bash
# 运行移动端示例
python gui_owl_pipeline/example_mobile.py

# 运行PC端示例
python gui_owl_pipeline/example_pc.py
```

## 输出结构

```
output/
├── queries/              # 生成的Query
├── trajectories/         # 收集的轨迹
│   ├── traj_xxx/
│   │   ├── trajectory.json
│   │   └── screenshots/
├── evaluations.json      # 轨迹评估结果
├── guidance/            # Query指导
└── downstream/          # 下游数据
    ├── grounding/
    ├── planning/
    └── action_semantics/
```

## 配置参数

- `parallel_sessions`: 并行会话数
- `max_trajectory_length`: 最大轨迹长度
- `critic_threshold`: 质量评判阈值
- `max_retries`: 失败重试次数
- `screenshot_interval`: 截图间隔

## 自演化机制

1. **初始轨迹生成**: GUI-Owl模型基于Query生成轨迹
2. **质量评估**: 双层评估确保数据质量
3. **困难识别**: 识别失败或低质量轨迹
4. **指导生成**: 为困难Query生成特定指导
5. **迭代改进**: 基于指导重新生成轨迹
6. **循环优化**: 持续迭代直到质量达标

## 注意事项

- 需要有效的AgentBay API密钥
- 大规模数据生产建议使用多个并行会话
- 可根据需要自定义评估标准和阈值
- 支持断点续传和增量数据生产