"""候选样本契约（app/contracts/candidate.py）

论文核心候选三元组 (q, s, r)（概念模型 §3.6/§3.7）：
- question q   自然语言问题，反思阶段固定
- sql s        目标 SQL，可被修正（执行反馈在 SQLExecutionResult）
- rationale r  结构化生成依据，可被修正并保留 correction_history

CandidateSample 在旧 Triple 之上补充 provenance（数据库/知识快照/模型/
Prompt 来源）；Triple 保留为兼容别名，旧 JSONL 产物可直接解析。
q ≠ r：问题是用户意图，rationale 是生成依据（概念模型 §5）。
"""

from typing import Optional

from pydantic import BaseModel, Field

from app.contracts.diagnosis import (
    CorrectionRecord,
    Diagnosis,
)
from app.contracts.execution import SQLExecutionResult
from app.contracts.scenario import GenerationMetadata


class TableSelection(BaseModel):
    """r.table_selection"""

    tables_used: list[str] = Field(default_factory=list)
    reasoning: str = ""


class ColumnOperation(BaseModel):
    """r.column_selection 单项"""

    name: str
    type: str = ""                 # 数据类型
    operation: str = ""            # SELECT/WHERE/GROUP BY/ORDER BY/AGG
    purpose: str = ""


class SQLStrategy(BaseModel):
    """r.sql_strategy：SQL 生成策略选择（论文结构化 think 的核心组成，
    也是诊断阶段追溯 SQL 设计意图的依据）。"""

    operations: list[str] = Field(default_factory=list)   # ["SELECT","ORDER BY DESC","LIMIT"]
    approach: str = ""
    no_need: list[str] = Field(default_factory=list)      # 明确不需要的操作


class Rationale(BaseModel):
    """结构化推理链 r（论文 Fig.training_data_format）。

    生成阶段填充 focus/metadata/table_selection/column_selection/
    sql_strategy/expected_output；反思阶段追加 errors 与
    correction_history（每次修正追加，不覆盖）。
    """

    focus: str = ""
    metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    table_selection: TableSelection = Field(default_factory=TableSelection)
    column_selection: list[ColumnOperation] = Field(default_factory=list)
    sql_strategy: SQLStrategy = Field(default_factory=SQLStrategy)
    expected_output: str = ""

    # 反思阶段填充
    errors: list[Diagnosis] = Field(default_factory=list)
    correction_history: list[CorrectionRecord] = Field(default_factory=list)
    admission_decision: str = "unresolved"
    admission_reason: str = "not_diagnosed"


class Question(BaseModel):
    """q：自然语言问题（反思阶段固定不可变）。"""

    question_id: str
    text: str = ""
    question_focus: str = ""       # 问题关注点（生成阶段产出）
    business_rules: list[dict] = Field(default_factory=list)
    metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    # 生成该问题时看到的表子集（表格子集裁剪：q 与 s 的提示词必须同子集；
    # 空 = 未裁剪，SQL 生成回退全库上下文）
    context_tables: list[str] = Field(default_factory=list)


class CandidateProvenance(BaseModel):
    """候选来源溯源：任何 accepted 样本都必须能回溯到
    数据库、知识快照、Prompt 与模型版本。"""

    database_id: str = ""
    dialect: str = "sqlite"
    knowledge_snapshot_id: str = ""
    knowledge_content_digest: str = ""
    model_id: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    created_at: str = ""


class CandidateSample(BaseModel):
    """论文核心候选 (q, s, r) 及其来源。

    生命周期：生成（q,r 完整 + s 已生成）→ 反思（修正 s/r，追加历史）
    → 准入（accepted/rejected/unresolved）。
    """

    question: Question
    sql_result: SQLExecutionResult = Field(default_factory=SQLExecutionResult)
    rationale: Rationale = Field(default_factory=Rationale)
    provenance: CandidateProvenance = Field(default_factory=CandidateProvenance)

    @property
    def question_id(self) -> str:
        return self.question.question_id

    def to_training_record(self) -> dict:
        """转为论文 Fig.training_data_format 的 JSON 记录。

        column_selection 输出为 {columns_used: [...]}，与既有产物逐字对齐。
        在论文四元组之外补充：
        - correction_history：反思阶段的逐次矫正记录（sql_before→sql_after）
        - execution：SQL 实际执行结果（成功/行数 + 受控行采样）
        """
        r = self.sql_result
        return {
            "question_id": self.question_id,
            "question": self.question.text,
            "think": {
                "focus": self.rationale.focus,
                "metadata": self.rationale.metadata.model_dump(),
                "table_selection": self.rationale.table_selection.model_dump(),
                "column_selection": {
                    "columns_used": [c.model_dump() for c in self.rationale.column_selection],
                },
                "sql_strategy": self.rationale.sql_strategy.model_dump(),
                "expected_output": self.rationale.expected_output,
            },
            "answer": self.sql_result.sql,
            "correction_history": [
                c.model_dump() for c in self.rationale.correction_history],
            "execution": {
                "executed": r.executed,
                "execution_success": r.execution_success,
                "result_count": r.result_count,
                "execution_error": r.execution_error,
                "result_columns": r.result_columns,
                "result_rows": r.result_rows,
                "result_rows_truncated": r.result_rows_truncated,
                "rows_capped": r.rows_capped,
            },
        }


# 兼容别名：旧管线/评测脚本消费的 Triple 即 CandidateSample。
# 概念模型 §5：不得再引入"通用语义三元组"含义。
Triple = CandidateSample
