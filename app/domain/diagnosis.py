"""LLM 语义审查（app/domain/diagnosis.py）

论文 §III.E Diagnose 的语义部分：
    E^(k) 含 Diagnose(τ, K, D) 的语义诊断

移植自旧 tools/diagnosis/diagnose.py 的 _llm_semantic_review + _kb_digest：
- 审查顺序对齐 Definition 1：问题有效性 → 意图匹配 → 使用一致性
- K 以"所涉元素的相关知识条目"注入（不是整库 K）
- 每个违例解析为 e_j=(t_j, u_j, λ_j, v_j)
- 审查失败仅重试序列化契约（FORMAT RETRY），不重新解读候选；
  重试仍失败 → SEMANTIC_REVIEW_FAILED（技术失败，不得当无错误）
"""

import logging
import re
from typing import Optional

from app.contracts.candidate import CandidateSample
from app.contracts.diagnosis import (
    Diagnosis,
    DetectorType,
    ErrorLocation,
    ErrorType,
    SemanticClass,
)
from app.contracts.knowledge import KnowledgeSnapshot
from app.contracts.parsing import ParsedSql
from app.ports.llm import ChatGateway
from app.ports.prompting import PromptRenderer

logger = logging.getLogger(__name__)


class SemanticReviewer:
    """LLM 语义审查器（程序化检查之外的第二类诊断来源）。"""

    def __init__(self, chat: ChatGateway, prompts: PromptRenderer,
                 enabled: bool = True):
        self.chat = chat
        self.prompts = prompts
        self.enabled = enabled

    def review(self, sample: CandidateSample, snapshot: KnowledgeSnapshot,
               parsed: Optional[ParsedSql]) -> list[Diagnosis]:
        """对候选做 LLM 语义审查 → list[Diagnosis]。"""
        if not self.enabled:
            return []
        knowledge_context = self._kb_digest(sample, snapshot, parsed)
        prompt = self.prompts.render(
            "diagnosis/semantic_review.j2",
            question=sample.question.text,
            sql=sample.sql_result.sql,
            rationale=sample.rationale.focus,
            knowledge_context=knowledge_context,
        )
        try:
            result = self.chat.generate_json(prompt)
        except Exception as first_error:
            # 只重试序列化契约；不重新解读 (q, s)
            retry_prompt = (
                f"{prompt}\n\n"
                "FORMAT RETRY: Return exactly one valid JSON object matching the "
                "required schema. Do not use markdown, comments, or explanatory text."
            )
            try:
                result = self.chat.generate_json(retry_prompt)
            except Exception as second_error:
                logger.warning(
                    "LLM 语义审查失败 %s: first=%s; retry=%s",
                    sample.question_id, first_error, second_error)
                return [Diagnosis(
                    type=ErrorType.SEMANTIC_REVIEW_FAILED,
                    detail=f"LLM semantic review failed after retry: {second_error}",
                    artifact="s",
                    detector=DetectorType.LLM,
                )]

        errors: list[Diagnosis] = []
        issues = result.get("issues", [])
        if not isinstance(issues, list):
            return errors
        for issue in issues:
            if not isinstance(issue, dict):
                continue
            detail = (issue.get("detail") or issue.get("description") or "").strip()
            if not detail:
                continue
            # 论文七类 t_j（非法类名兜底为通用语义不一致）
            try:
                sc = SemanticClass(str(issue.get("class", "")).strip())
            except ValueError:
                sc = None
            artifact = (issue.get("artifact") or "s").strip().lower()
            if artifact not in {"q", "s", "r"}:
                artifact = "s"
            errors.append(Diagnosis(
                type=ErrorType.SEMANTIC_INCONSISTENCY,
                location=ErrorLocation(
                    clause=issue.get("clause", ""),
                    column=issue.get("column") or None,
                    table=issue.get("table") or None,
                ),
                detail=detail,
                semantic_class=sc,
                artifact=artifact,
                violated_condition=(issue.get("condition") or "").strip(),
            ))
        return errors

    # ---------- K 摘要（τ, K, D 的 K 注入） ----------

    def _kb_digest(self, sample: CandidateSample, snapshot: KnowledgeSnapshot,
                   parsed: Optional[ParsedSql]) -> str:
        """只取 SQL 涉及表/列的相关条目，避免整库 K 撑爆提示。"""
        parts: list[str] = []

        domain = snapshot.k2_domain
        if domain.description:
            parts.append(f"[K2 域规则] {domain.description}")

        sql = sample.sql_result.sql
        # 涉及表
        touched_tables: set[str] = set()
        if parsed is not None:
            touched_tables = {t for t in parsed.tables if t}
        if not touched_tables:
            # 兜底：从 SQL 文本里猜表名（K1 全部表名中出现者）
            try:
                names = snapshot.k1_schema.all_table_names()
                touched_tables = {
                    n for n in names
                    if re.search(rf"\b{re.escape(n)}\b", sql, re.IGNORECASE)
                }
            except Exception:
                touched_tables = set()

        touched_cols: list[tuple[str, str]] = parsed.columns if parsed else []

        # K4/K3：SQL 涉及的列
        for table, column in touched_cols[:30]:
            cs = snapshot.get_column(table, column)
            if cs is not None:
                parts.append(f"[K4] {table}.{column}: {cs.description}")
            cat = snapshot.get_field_type(table, column)
            if cat is not None:
                parts.append(f"[K3] {table}.{column}: {cat.value}")

        # K5：涉及表约束/描述 + K1 观测实例样本
        schema = snapshot.k1_schema
        for table in sorted(touched_tables)[:10]:
            ts = snapshot.get_table_semantic(table)
            if ts is not None:
                parts.append(f"[K5] {table}: {ts.description}")
            schema_table = schema.get_table(table)
            if schema_table is not None:
                for col in schema_table.columns:
                    if any(tbl == table and column == col.name
                           for tbl, column in touched_cols):
                        if col.sample_values:
                            observed = ", ".join(str(v) for v in col.sample_values[:5])
                            parts.append(
                                f"[K1 observed sample] {table}.{col.name}: "
                                f"{observed} (not a complete domain)")
                        parts.append(
                            f"[K1 metadata] {table}.{col.name}: "
                            f"entropy={col.entropy_level}, rows={schema_table.row_count}")

        # K6：涉及表之间的关系
        for rel in snapshot.k6_relations:
            if rel.source_table in touched_tables or rel.target_table in touched_tables:
                src = (f"{rel.source_table}.{rel.source_column}"
                       if rel.source_column else rel.source_table)
                tgt = (f"{rel.target_table}.{rel.target_column}"
                       if rel.target_column else rel.target_table)
                reason = f" ({rel.reason})" if rel.reason else ""
                parts.append(f"[K6] {src} → {tgt} {rel.relationship_type}{reason}")

        return "\n".join(parts)
