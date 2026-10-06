"""诊断契约（app/contracts/diagnosis.py）

论文 §III.E：Diagnose / Retrieve / Correct 三个算子的输入输出。
- Diagnosis（旧名 Error）对齐论文 e_j=(t_j,u_j,λ_j,v_j)，
  即（类别，受影响产物，位置，违反条件）。
- EvidenceBundle（旧名 Evidence）对齐 Φ（从 K 取出的修正证据）。
- CorrectionRecord（旧名 Correction）对齐"append to r 保留精修历史"。
- ReflectionTrace（旧名 DiagnosisTrace）记录候选的完整裁决轨迹。
- ReviewedDiagnosisSet 对齐 Retrieve 的复核算子输出（保留/驳回/不确定）。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.contracts.execution import AdmissionDecision
from app.contracts.knowledge import (
    ColumnSemantics,
    CrossTableRelation,
    DomainKnowledge,
    FieldCategory,
    TableSemantics,
)


class ErrorType(str, Enum):
    """错误类型。两类来源对应论文"程序化校验 + LLM 语义审查"的混合设计：

    - 结构性错误（程序化校验产出）：COLUMN_*/JOIN_*/AGGREGATION_*/TABLE_*/SYNTAX/...
    - 语义性错误（LLM 审查产出）：SEMANTIC_INCONSISTENCY（带七类 SemanticClass）
    - SEMANTIC_REVIEW_FAILED 是审查执行失败，不是空错误列表，
      不得误标为 accepted。
    """

    # 结构性错误（程序化校验）
    COLUMN_NOT_FOUND = "column_not_found"
    COLUMN_SEMANTIC_MISMATCH = "column_semantic_mismatch"
    JOIN_INVALID = "join_invalid"
    # 实测画像判定：该表对在数据上 0 命中（连接不成立）
    JOIN_UNJOINABLE = "join_unjoinable"
    # 实测画像判定：键值含前导/尾随空格，需 TRIM 归一化
    JOIN_UNNORMALIZED_KEY = "join_unnormalized_key"
    AGGREGATION_TYPE_MISMATCH = "aggregation_type_mismatch"
    TABLE_NOT_FOUND = "table_not_found"
    SYNTAX_ERROR = "syntax_error"
    TYPE_ERROR = "type_error"
    EXECUTION_FAILED = "execution_failed"
    EMPTY_RESULT = "empty_result"
    VALUE_NOT_IN_DOMAIN = "value_not_in_domain"   # 谓词字面值不在实测值域（C 检查）

    # 语义性错误（LLM 审查）
    SEMANTIC_INCONSISTENCY = "semantic_inconsistency"
    SEMANTIC_REVIEW_FAILED = "semantic_review_failed"


class SemanticClass(str, Enum):
    """论文 Table semantic-taxonomy 的七类 SQL 语义错误：

    Group I（意图错配，仅凭 (q,s) 判定，不取知识条目）
    Group II（元素误用：II-1 证据 K3,K4；II-2 证据 K6）
    Group III（域规则违反：证据 K2,K5,K6）
    """

    I_1_TASK_MISMATCH = "I-1"
    I_2_CONDITION_MISMATCH = "I-2"
    I_3_AGGREGATION_GRANULARITY_MISMATCH = "I-3"
    I_4_ORDERING_RESULT_FORM_MISMATCH = "I-4"
    II_1_FIELD_ROLE_MISUSE = "II-1"
    II_2_JOIN_MISUSE = "II-2"
    III_1_DOMAIN_RULE_VIOLATION = "III-1"


# 七类 → 取证所需 K 层（论文 taxonomy 第三列）
SEMANTIC_CLASS_EVIDENCE: dict["SemanticClass", set[str]] = {
    SemanticClass.I_1_TASK_MISMATCH: set(),
    SemanticClass.I_2_CONDITION_MISMATCH: set(),
    SemanticClass.I_3_AGGREGATION_GRANULARITY_MISMATCH: set(),
    SemanticClass.I_4_ORDERING_RESULT_FORM_MISMATCH: set(),
    SemanticClass.II_1_FIELD_ROLE_MISUSE: {"K3", "K4"},
    SemanticClass.II_2_JOIN_MISUSE: {"K6"},
    SemanticClass.III_1_DOMAIN_RULE_VIOLATION: {"K2", "K5", "K6"},
}


class ErrorCategory(str, Enum):
    """错误的验证边界：可执行性与语义正确性不得混为一谈。"""

    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


class DetectorType(str, Enum):
    """错误发现者：分别报告确定性检查和 LLM 审查的效果。"""

    DETERMINISTIC = "deterministic"
    LLM = "llm"


class ErrorLocation(BaseModel):
    """错误定位（论文要求 positions in r）"""

    clause: str = ""           # SELECT / WHERE / JOIN / GROUP BY / ORDER BY 等
    column: Optional[str] = None
    table: Optional[str] = None


class Error(BaseModel):
    """单个诊断（论文 E 的元素，e_j=(t_j,u_j,λ_j,v_j)）。

    artifact="q" 表示问题本身无效：q 在反思中固定，候选直接拒绝。
    """

    type: ErrorType
    location: ErrorLocation = Field(default_factory=ErrorLocation)
    detail: str = ""
    evidence_ref: Optional[str] = None   # 指向 EvidenceBundle 的 key，供 Retrieve 路由
    category: Optional[ErrorCategory] = None
    detector: Optional[DetectorType] = None
    semantic_class: Optional[SemanticClass] = None  # 仅语义性错误填写（t_j）
    artifact: Optional[str] = None                  # 受影响产物 q/s/r（u_j）
    violated_condition: str = ""                    # 违反的条件（v_j）

    def model_post_init(self, __context) -> None:
        """为程序化检查产生的 Error 补全可审计的来源和边界。"""
        if self.category is None:
            self.category = (
                ErrorCategory.SEMANTIC
                if self.type in {
                    ErrorType.SEMANTIC_INCONSISTENCY,
                    ErrorType.SEMANTIC_REVIEW_FAILED,
                    ErrorType.COLUMN_SEMANTIC_MISMATCH,
                }
                else ErrorCategory.STRUCTURAL
            )
        if self.detector is None:
            self.detector = (
                DetectorType.LLM
                if self.type == ErrorType.SEMANTIC_INCONSISTENCY
                else DetectorType.DETERMINISTIC
            )

    def with_location(self, clause: str = "", column: Optional[str] = None,
                      table: Optional[str] = None) -> "Error":
        """链式设置 location"""
        self.location = ErrorLocation(clause=clause, column=column, table=table)
        return self


class EvidenceBundle(BaseModel):
    """修正证据 Φ（Retrieve 产物）：带来源与用途，不是无含义的 score。

    - columns（K4）：正确列语义，供替换错误列引用（II-1）
    - field_types（K3）：字段类型，供聚合错误找正确 measure 列（II-1）
    - relations（K6）：关系/外键，供 JOIN 错误找正确连接键（II-2/III-1）
    - domain_rules（K2）：域规则（III-1）
    - table_constraints（K5）：表约束（III-1）
    """

    columns: dict[str, ColumnSemantics] = Field(default_factory=dict)
    field_types: dict[str, FieldCategory] = Field(default_factory=dict)
    relations: list[CrossTableRelation] = Field(default_factory=list)
    domain_rules: Optional[DomainKnowledge] = None
    table_constraints: list[TableSemantics] = Field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.columns
            and not self.field_types
            and not self.relations
            and self.domain_rules is None
            and not self.table_constraints
        )


class CorrectionRecord(BaseModel):
    """单次修正记录（append to r，保留精修历史，不覆盖原始候选）。"""

    iteration: int
    errors_addressed: list[ErrorType] = Field(default_factory=list)
    sql_before: str = ""
    sql_after: str = ""
    summary: str = ""


class DiagnosisIteration(BaseModel):
    """Eq.4 的一次完整可复现检查（Diagnose/Retrieve/Correct 证据齐全）。"""

    iteration: int
    question: str = ""
    sql_before: str = ""
    rationale_focus: str = ""
    errors: list[Error] = Field(default_factory=list)
    evidence: Optional[EvidenceBundle] = None
    sql_after: Optional[str] = None
    execution_success_after: Optional[bool] = None
    execution_error_after: Optional[str] = None
    result_count_after: Optional[int] = None
    action: str = "verified"  # verified / corrected / rejected_question_invalid /
    #                           semantic_review_failed / max_iterations_reached


class ReflectionTrace(BaseModel):
    """一个候选从初始生成到最终裁决的完整审计轨迹（候选局部状态，
    不写回共享知识库）。"""

    question_id: str
    original_question: str = ""
    original_sql: str = ""
    max_correction_iterations: int = 0
    iterations: list[DiagnosisIteration] = Field(default_factory=list)
    final_question: str = ""
    final_sql: str = ""
    final_execution_success: Optional[bool] = None
    final_execution_error: Optional[str] = None
    final_result_count: Optional[int] = None
    decision: AdmissionDecision = AdmissionDecision.UNRESOLVED
    decision_reason: str = ""


# ============================================================
# Retrieve 复核算子输出（论文 Eq.(retrieve)：(E~, H) = Retrieve(...)）
# ============================================================

class ReviewStatus(str, Enum):
    """诊断复核结论。uncertain 不等同审查通过；复核服务调用失败是技术失败，
    不得当成 dismissed。"""

    RETAINED = "retained"
    DISMISSED = "dismissed"
    UNCERTAIN = "uncertain"


class ReviewedDiagnosis(BaseModel):
    """单条诊断的复核记录：结论 + 理由 + 复核时参考的证据。"""

    diagnosis: Error
    status: ReviewStatus = ReviewStatus.RETAINED
    reason: str = ""
    evidence_refs: list[str] = Field(default_factory=list)


class ReviewedDiagnosisSet(BaseModel):
    """ReviewedDiagnosisSet + EvidenceBundle 中的前者：
    只有 retained 的诊断进入 Correct。"""

    reviewed: list[ReviewedDiagnosis] = Field(default_factory=list)
    review_strategy: str = "retain_all_supported"  # 首版固定策略（行为基线）
    review_failed: bool = False                    # 复核服务本身失败（技术失败）
    review_error: str = ""

    @property
    def retained(self) -> list[Error]:
        return [r.diagnosis for r in self.reviewed
                if r.status == ReviewStatus.RETAINED]

    @property
    def has_uncertain(self) -> bool:
        return any(r.status == ReviewStatus.UNCERTAIN for r in self.reviewed)


# ============================================================
# 兼容别名（概念模型 §5：旧名不得在新代码中继续扩散，仅用于读旧产物）
# ============================================================

Diagnosis = Error                 # 论文算子输出命名
Evidence = EvidenceBundle         # 旧名
Correction = CorrectionRecord     # 旧名
DiagnosisTrace = ReflectionTrace  # 旧名
SampleDecision = AdmissionDecision  # 旧名（准入裁决三态）
