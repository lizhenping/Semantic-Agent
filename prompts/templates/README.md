# 提示词模板说明

本目录包含SemanticSQL Agent使用的所有Jinja2提示词模板。

## 目录结构

```
templates/
├── analysis/           # 数据库分析相关提示词
│   ├── database_analysis.j2         # 数据库分析任务提示词
│   ├── initial_domain_analysis.j2   # 初始领域分析
│   ├── field_classification.j2      # 字段分类
│   ├── column_description.j2        # 列描述生成
│   ├── table_description.j2         # 表描述生成
│   ├── er_analysis_logical.j2       # 逻辑关系分析
│   └── er_analysis_conceptual.j2    # 概念关系分析
├── generation/         # 训练数据生成相关提示词
│   ├── training_data_generation.j2  # 训练数据生成任务
│   └── question_generation.j2       # 问题生成
├── tools/             # 工具相关提示词
│   ├── sql_generation.j2           # SQL生成
│   └── tool_system.j2              # 工具系统提示词
├── reflection/        # 反思相关提示词
│   ├── analyze_empty_result.j2     # 分析空结果原因
│   └── evaluate_result_quality.j2  # 评估结果质量
├── thinking/          # 思考相关提示词
│   └── sequential_thinking.j2      # 顺序思考
└── system/           # 系统提示词
    └── main.j2                     # 主系统提示词
```

## 模板使用说明

### 1. 分析工具链（analysis/）

按执行顺序：

1. **initial_domain_analysis.j2** - 领域分析
   - 输入：database_name, database_ddl, type_distribution, field_patterns
   - 输出：domain_type, domain_description, key_entities, business_rules

2. **field_classification.j2** - 字段分类
   - 输入：fields[], domain_type, domain_description
   - 输出：字段类别(identifier/measure/dimension等)、类型、重要性

3. **column_description.j2** - 列描述生成
   - 输入：table_name, table_ddl, columns[], domain_type
   - 输出：每列的业务含义描述

4. **table_description.j2** - 表描述生成
   - 输入：tables[], domain_type, domain_description
   - 输出：每个表的业务职责描述

5. **er_analysis_logical.j2** - 逻辑关系分析
   - 输入：formatted_schema, fk_info, database_name
   - 输出：逻辑关系列表

6. **er_analysis_conceptual.j2** - 概念关系分析
   - 输入：formatted_schema, physical_relations, logical_relations
   - 输出：概念层面的实体关系

### 2. Memory数据流

工具执行顺序和Memory依赖：

```
schema_extraction → memory["schema_info"]
                    ↓
domain_analysis   → memory["domain_info"] 
                    ↓
field_classification → memory["field_classification"]
                    ↓
column_meaning    → memory["column_meanings"]
                    ↓
table_meaning     → memory["table_meanings"]
                    ↓
er_analysis       → memory["er_relations"]
```

每个工具都可以访问之前工具保存的memory数据。

### 3. 模板变量规范

- 使用下划线分隔的小写字母命名
- 列表变量使用复数形式
- 布尔变量使用is_或has_前缀
- 避免使用保留字

### 4. 输出格式要求

所有分析类模板都要求返回JSON格式，确保LLM返回可解析的结构化数据。

## 使用示例

```python
from prompts.manager import PromptManager

prompt_manager = PromptManager()

# 获取分析提示词
prompt = prompt_manager.get_analysis_prompt(
    "field_classification",
    fields=fields_data,
    domain_type="电商",
    domain_description="..."
)

# 获取工具提示词
prompt = prompt_manager.get_tool_prompt(
    "sql_generation",
    context=context,
    question=question,
    dialect="mysql"
)
```