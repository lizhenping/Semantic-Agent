"""准入裁决（app/domain/admission.py）

论文 §III.E 末段 + doc/02 §3 反思状态机：
- accepted：B^(k)=∅ 且 Ẽ^(k)=∅（可执行、问题有效、无保留诊断）
- rejected：问题无效（q 不可修复）或最终 SQL 不可执行
- unresolved：达到修正上限后仍有语义问题（不伪装为通过）

三类互斥终态；三份修正对应至多四次完整验证；
零次修正也必须执行一次验证。技术失败（编排异常/审查失败）
按候选失败处理，不计入论文语义错误。

结果形态闸门（result_gate）：结果为空 / 取回上限命中 / 问题问"所有"
却被 SQL 的 LIMIT 截断 —— 这三类都是**数据事实问题**（改 SQL 无用），
直接 rejected；训练语料不留"问题-答案口径不一致或结果不完整"的样本。
"""

import re
from typing import Optional

from app.contracts.diagnosis import Diagnosis, ErrorType
from app.contracts.execution import AdmissionDecision

# 问题是否要求"全部"（决定 SQL 自带 LIMIT 是否构成口径冲突）。
# 注意：不能用"每个"——"计算每个组合的前 5 名"是分组语义，不是要求返回全部；
# 曾因 `每(?:一)?个` 误拒 59/102 条合法的 Top-N 样本。
_ASKS_ALL_RE = re.compile(r"所有|全部|一切|all\b", re.I)
# SQL 顶层 LIMIT 值（取最后一处：外层查询的限制）
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)", re.I)


def result_form_gate(result_count: Optional[int], rows_capped: bool,
                     sql: str, question_text: str) -> Optional[str]:
    """结果形态闸门：返回拒绝原因（None = 通过）。

    规则：
    1. 结果 0 行 → 数据前提不成立
    2. 取回上限命中 → 结果不完整（result_count 只是下限）
    3. SQL 自带 LIMIT 且恰好返回 LIMIT 行（真截断）：
       - 问题问"所有/全部" → 口径冲突（truncated_despite_all_intent）
       - 问题里找不到该数量（既没要"前 N"也没提这个数）→ 模型自作主张截断
         （truncated_without_top_n_intent）
       - 问题明确写了这个数量（如"前 5 条"→LIMIT 5）→ 合法放行
    """
    if result_count is None and not rows_capped:
        return None      # 未执行（--no-execute / diagnose 输入）→ 无法判定
    if rows_capped:
        return "result_rows_capped"
    if result_count == 0:
        return "empty_result"
    limit = _sql_limit(sql)
    if limit is None or result_count != limit:
        return None
    if _ASKS_ALL_RE.search(question_text or ""):
        return "truncated_despite_all_intent"
    if not _question_mentions_count(question_text, limit):
        return "truncated_without_top_n_intent"
    return None


def _question_mentions_count(question_text: str, count: int) -> bool:
    """问题里是否出现了该数量（"前 5 条" ↔ LIMIT 5 的口径对账）。

    按独立数字匹配（(?<!\\d)5(?!\\d)），避免把 2015 误当成 5。
    """
    if not question_text:
        return False
    return re.search(rf"(?<!\d){count}(?!\d)", question_text) is not None


def _sql_limit(sql: str) -> Optional[int]:
    """SQL 顶层 LIMIT 值（多层时取最后一处 = 最外层）。"""
    matches = _LIMIT_RE.findall(sql or "")
    return int(matches[-1]) if matches else None


def decide(
    final_execution_success: bool | None,
    terminal_diagnoses: list[Diagnosis],
    *,
    result_count: Optional[int] = None,
    rows_capped: bool = False,
    sql: str = "",
    question_text: str = "",
) -> tuple[AdmissionDecision, str]:
    """按终态诊断集合、最终执行结果与结果形态裁决。

    与旧 pipeline._eq4_loop 尾部裁决逐字对齐：
    - SQL 不可执行 → rejected / final_sql_not_executable
    - 结果形态闸门（空结果/取回上限/口径冲突）→ rejected / <gate reason>
    - 问题无效（artifact="q"）→ rejected / question_invalid_unrepairable
    - 其余残余诊断 → unresolved / diagnosis_errors_remain_after_limit
    - 无诊断且可执行 → accepted / executable_and_no_detected_errors
    """
    if final_execution_success is not True:
        return AdmissionDecision.REJECTED, "final_sql_not_executable"
    gate = result_form_gate(result_count, rows_capped, sql, question_text)
    if gate:
        return AdmissionDecision.REJECTED, gate
    if terminal_diagnoses:
        if any((d.artifact or "") == "q" for d in terminal_diagnoses):
            # 论文：问题无效不可修复，直接拒绝（而非 unresolved）
            return AdmissionDecision.REJECTED, "question_invalid_unrepairable"
        return AdmissionDecision.UNRESOLVED, "diagnosis_errors_remain_after_limit"
    return AdmissionDecision.ACCEPTED, "executable_and_no_detected_errors"


def is_question_invalid(diagnoses: list[Diagnosis]) -> bool:
    """诊断中是否存在问题无效（artifact="q"）。"""
    return any((d.artifact or "") == "q" for d in diagnoses)


def has_review_failure(diagnoses: list[Diagnosis]) -> bool:
    """是否含审查执行失败（SEMANTIC_REVIEW_FAILED）。

    审查失败不是可修复的语义发现：停止迭代，按候选失败处理。
    """
    return any(d.type == ErrorType.SEMANTIC_REVIEW_FAILED for d in diagnoses)
