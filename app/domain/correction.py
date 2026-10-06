"""语义修正（app/domain/correction.py）

论文 §III.E Correct 算子（Eq.4 第三步）：
    (s^(k+1), r^(k+1)) = Correct(τ^(k), B^(k), Ẽ^(k), H^(k), K)

论文 L311 硬约束（移植自旧 tools/diagnosis/correct.py）：
1. 只做局部、以诊断为条件的编辑，绝不重新生成整条三元组
2. 问题 q 永不编辑（artifact="q" 的诊断不可修复，由准入拒绝）
3. 每次编辑追加 CorrectionRecord，不覆盖历史
4. 保留生成阶段的 expected_output（记录设计意图），不用证据猜测覆盖
"""

import logging
import re
from typing import Optional

from app.contracts.candidate import CandidateSample
from app.contracts.database import DatabaseContext
from app.contracts.diagnosis import (
    CorrectionRecord,
    Diagnosis,
    EvidenceBundle,
)
from app.ports.database import DatabaseGateway
from app.ports.llm import ChatGateway
from app.ports.prompting import PromptRenderer

from app.contracts.execution import (
    DEFAULT_RESULT_ROWS_LIMIT,
    apply_result_rows,
)

logger = logging.getLogger(__name__)


class Corrector:
    """基于 retained 诊断 + 证据 Φ 修正 s/r（q 固定）。"""

    def __init__(self, chat: Optional[ChatGateway], prompts: PromptRenderer,
                 gateway: Optional[DatabaseGateway] = None,
                 context: Optional[DatabaseContext] = None,
                 dialect: str = "sqlite",
                 rows_limit: int = DEFAULT_RESULT_ROWS_LIMIT,
                 row_cap: Optional[int] = None):
        self.chat = chat
        self.prompts = prompts
        self.gateway = gateway
        self.context = context
        self.dialect = dialect
        # 修正后重新执行的结果行采样上限（进 result_rows）
        self.rows_limit = rows_limit
        # 答案执行的取回上限（不改写 SQL；命中即 rows_capped=True）
        self.row_cap = row_cap or None

    def correct(self, sample: CandidateSample,
                diagnoses: list[Diagnosis],
                evidence: EvidenceBundle,
                iteration: int) -> CandidateSample:
        """修正候选 → 新 CandidateSample（原候选不变，历史追加）。"""
        # 问题无效（artifact="q"）不可修复，由编排层拒绝；这里只修 s/r
        repairable = [e for e in diagnoses if (e.artifact or "s") != "q"]

        sql_before = sample.sql_result.sql
        sql_after = sql_before

        if self.chat is not None and repairable:
            sql_after = self._llm_correct(sample, repairable, evidence)
            sql_after = self._postprocess_sql(sql_after)
        elif not repairable:
            logger.debug("%s: 仅问题无效类诊断，无 s/r 可修正", sample.question_id)
        else:
            logger.warning("corrector 未注入 chat，跳过 LLM 修正")

        # 执行新 SQL（修正后的候选必须重新执行，不是只改文本）
        new_sql_result = self._execute(sql_after, sample.sql_result.dialect)

        correction = CorrectionRecord(
            iteration=iteration,
            errors_addressed=[e.type for e in repairable],
            sql_before=sql_before,
            sql_after=sql_after,
            summary=self._summarize_errors(repairable),
        )

        # q 原样（The question is never edited）；r 深拷贝后追加历史
        new_rationale = sample.rationale.model_copy(deep=True)
        new_rationale.errors = list(diagnoses)
        new_rationale.correction_history.append(correction)

        new_sample = CandidateSample(
            question=sample.question,
            sql_result=new_sql_result,
            rationale=new_rationale,
            provenance=sample.provenance.model_copy(deep=True),
        )
        return new_sample

    # ---------- LLM 修正 ----------

    def _llm_correct(self, sample: CandidateSample,
                     diagnoses: list[Diagnosis],
                     evidence: EvidenceBundle) -> str:
        prompt = self.prompts.render(
            "diagnosis/correct.j2",
            question=sample.question.text,
            sql=sample.sql_result.sql,
            rationale=sample.rationale.focus,
            errors=self._render_errors(diagnoses),
            evidence=self._render_evidence(evidence),
            dialect=self.dialect,
        )
        result = self.chat.generate_json(prompt)
        return result.get("sql") or result.get("corrected_sql") or sample.sql_result.sql

    # ---------- 执行 ----------

    def _execute(self, sql: str, dialect: str):
        from app.contracts.execution import SQLExecutionResult
        result = SQLExecutionResult(sql=sql, dialect=dialect,
                                    corrected_by_reflection=True)
        if self.gateway is None or self.context is None:
            return result
        try:
            ret = self.gateway.execute_readonly(self.context, sql, limit=None,
                                                row_cap=self.row_cap,
                                                keep_rows=self.rows_limit)
            if ret.get("success"):
                result.executed = True
                result.execution_success = True
                result.result_count = ret.get("row_count", 0)
                apply_result_rows(result, ret, self.rows_limit)
            else:
                result.executed = True
                result.execution_success = False
                result.execution_error = str(ret.get("error", "执行失败"))
        except Exception as e:  # noqa: BLE001
            result.executed = True
            result.execution_success = False
            result.execution_error = str(e)
        return result

    # ---------- 渲染辅助 ----------

    @staticmethod
    def _postprocess_sql(sql: str) -> str:
        sql = re.sub(r"```sql\s*", "", sql, flags=re.IGNORECASE)
        sql = re.sub(r"```\s*", "", sql)
        sql = re.sub(r"\s+", " ", sql).strip()
        if sql and not sql.endswith(";"):
            sql += ";"
        return sql

    @staticmethod
    def _render_errors(diagnoses: list[Diagnosis]) -> str:
        if not diagnoses:
            return "(无错误)"
        lines = []
        for i, e in enumerate(diagnoses, 1):
            loc = e.location
            loc_str = loc.clause or ""
            if loc.table:
                loc_str += f" {loc.table}"
            if loc.column:
                loc_str += f".{loc.column}"
            lines.append(f"{i}. [{e.type.value}] {loc_str}: {e.detail}")
        return "\n".join(lines)

    @staticmethod
    def _render_evidence(evidence: Optional[EvidenceBundle]) -> str:
        if evidence is None:
            return "(无证据)"
        parts: list[str] = []
        if evidence.columns:
            parts.append("可用列（K4 列语义）：")
            for key, col in evidence.columns.items():
                parts.append(f"  - {key}: {col.description}")
        if evidence.field_types:
            parts.append("字段类型（K3）：")
            for key, cat in evidence.field_types.items():
                parts.append(f"  - {key}: {cat.value}")
        if evidence.relations:
            parts.append("可用关系（K6）：")
            for rel in evidence.relations:
                src = (f"{rel.source_table}.{rel.source_column}"
                       if rel.source_column else rel.source_table)
                tgt = (f"{rel.target_table}.{rel.target_column}"
                       if rel.target_column else rel.target_table)
                parts.append(f"  - {src} → {tgt} ({rel.relationship_type})")
        if evidence.domain_rules is not None and evidence.domain_rules.description:
            parts.append(f"域规则（K2）：{evidence.domain_rules.description}")
        for ts in evidence.table_constraints:
            parts.append(f"表约束（K5）{ts.table_name}: {ts.description}")
        return "\n".join(parts) if parts else "(无具体证据)"

    @staticmethod
    def _summarize_errors(diagnoses: list[Diagnosis]) -> str:
        if not diagnoses:
            return "无错误"
        types = sorted({e.type.value for e in diagnoses})
        return f"修正 {len(diagnoses)} 个错误: {', '.join(types)}"
