"""证据检索与复核（app/domain/evidence.py）

论文 §III.E Retrieve 算子，两步：
1. 取证路由：按错误类型取最小相关证据（论文 Table semantic-taxonomy）；
2. 诊断复核：结合候选、数据库反馈与证据复核每条诊断，
   输出 ReviewedDiagnosisSet（retained / dismissed / uncertain）。

首版复核策略固定为 retain_all_supported（行为冻结基线：
保留全部诊断，等价旧实现的 Retrieve 路由）；uncertain 不等同
审查通过，进入 unresolved。复核服务失败是技术失败，不得当 dismissed。
"""

import logging

from app.contracts.database import DatabaseContext
from app.contracts.diagnosis import (
    ErrorType,
    EvidenceBundle,
    ReviewStatus,
    ReviewedDiagnosis,
    ReviewedDiagnosisSet,
    SEMANTIC_CLASS_EVIDENCE,
    Diagnosis,
)
from app.contracts.execution import SQLExecutionResult
from app.contracts.knowledge import KnowledgeSnapshot

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """按错误类型从知识快照取证（Φ），并输出复核后的诊断集合。"""

    def __init__(self, review_strategy: str = "retain_all_supported"):
        self.review_strategy = review_strategy

    # ---------- 第一步：取证路由 ----------

    def retrieve(self, diagnoses: list[Diagnosis],
                 snapshot: KnowledgeSnapshot) -> EvidenceBundle:
        """按论文七类语义错误 + 结构性错误路由到对应 K 层取证。

        - I-1..I-4（意图错配）：仅凭 (q, s, r) 判定，不取知识条目
        - II-1 字段角色误用：K3 + K4
        - II-2 JOIN 误用：K6
        - III-1 域规则违反：K2 + K5 + K6
        - 结构性错误按修复需要取对应层
        """
        evidence = EvidenceBundle()
        for err in diagnoses:
            sc = err.semantic_class
            if sc is not None:
                needed = SEMANTIC_CLASS_EVIDENCE.get(sc, set())
                table = err.location.table
                if "K3" in needed and table:
                    for ft in snapshot.k3_field_types:
                        if ft.table_name == table:
                            evidence.field_types[ft.field_key] = ft.category
                if "K4" in needed:
                    cols = ([c for c in snapshot.k4_column_semantics
                             if c.table_name == table]
                            if table else snapshot.k4_column_semantics)
                    for col in cols:
                        evidence.columns[col.field_key] = col
                if "K5" in needed:
                    if table:
                        ts = snapshot.get_table_semantic(table)
                        if ts:
                            evidence.table_constraints.append(ts)
                    else:
                        evidence.table_constraints.extend(snapshot.k5_table_semantics)
                if "K6" in needed:
                    evidence.relations.extend(snapshot.foreign_keys())
                if "K2" in needed and evidence.domain_rules is None:
                    evidence.domain_rules = snapshot.k2_domain
                continue

            # ---- 结构性错误：按修复需要取证 ----
            if err.type in (ErrorType.COLUMN_NOT_FOUND,
                            ErrorType.COLUMN_SEMANTIC_MISMATCH):
                if err.location.table:
                    for col in snapshot.k4_column_semantics:
                        if col.table_name == err.location.table:
                            evidence.columns[col.field_key] = col

            elif err.type == ErrorType.JOIN_INVALID:
                evidence.relations.extend(snapshot.foreign_keys())

            elif err.type == ErrorType.AGGREGATION_TYPE_MISMATCH:
                if err.location.table:
                    for ft in snapshot.k3_field_types:
                        if ft.table_name == err.location.table:
                            evidence.field_types[ft.field_key] = ft.category

        return evidence

    # ---------- 第二步：诊断复核 ----------

    def review(self, diagnoses: list[Diagnosis],
               feedback: SQLExecutionResult,
               evidence: EvidenceBundle) -> ReviewedDiagnosisSet:
        """复核每条诊断：retained / dismissed / uncertain。

        首版策略 retain_all_supported：不丢弃任何诊断（与旧行为等价，
        不引入基于 LLM 自评的驳回）。后续引入 LLM 复核属算法实验，
        须保持本接口不变。
        """
        reviewed = [
            ReviewedDiagnosis(
                diagnosis=err,
                status=ReviewStatus.RETAINED,
                reason="retain_all_supported 策略默认保留",
            )
            for err in diagnoses
        ]
        return ReviewedDiagnosisSet(reviewed=reviewed,
                                    review_strategy=self.review_strategy)
