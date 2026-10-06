"""执行与准入契约（app/contracts/execution.py）

- SQLExecutionResult：s 的执行反馈（成功/失败/受控摘要）。
  执行是生成后的反馈阶段，不是语义验证的全部：
  execution_success=True 不足以推出 accepted。
- AdmissionDecision：accepted / rejected / unresolved 三互斥终态。
- AdmissionRecord：终态裁决记录（metrics 聚合对象）。
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AdmissionDecision(str, Enum):
    """样本准入状态（三互斥终态，终态之间没有箭头）。

    只有 ACCEPTED 进入训练数据；rejected/unresolved 保留摘要用于质量分析。
    """

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNRESOLVED = "unresolved"


class SQLExecutionResult(BaseModel):
    """s + 执行反馈（成功/失败 + 受控行采样）。

    result_rows 保留前缀采样行（上限 result_rows_limit），供训练数据
    记录 SQL 的实际输出结果；行数本身仍以 result_count 为准。
    """

    sql: str = ""
    dialect: str = "sqlite"
    executed: bool = False
    execution_success: Optional[bool] = None
    result_count: Optional[int] = None
    execution_error: Optional[str] = None
    corrected_by_reflection: bool = False   # 是否经反思阶段修正
    # 受控结果采样（执行成功时填充）
    result_columns: Optional[list] = None
    result_rows: Optional[list] = None
    result_rows_truncated: bool = False
    # 取回上限命中：网关在 row_cap 行处主动停止，result_count 只是下限。
    # 结果不完整，准入闸门据此拒绝（见 app/domain/admission.py）。
    rows_capped: bool = False


DEFAULT_RESULT_ROWS_LIMIT = 20

# 网关 QueryTimeout 产生的执行错误统一携带该标记（见 adapters/database_sqlalchemy.py）
EXECUTION_TIMEOUT_MARK = "query timeout"


def is_execution_timeout(error_text: Optional[str]) -> bool:
    """执行错误是否为超时（超时 → 整个问题放弃，不进入修正循环）。"""
    return bool(error_text) and EXECUTION_TIMEOUT_MARK in error_text


def apply_result_rows(result: SQLExecutionResult, gateway_ret: dict,
                      rows_limit: int = DEFAULT_RESULT_ROWS_LIMIT) -> None:
    """把网关执行摘要中的数据行（前缀采样）附加到执行反馈。

    网关 execute_readonly 成功时返回 {row_count, columns, data, rows_capped, ...}；
    这里按 rows_limit 截断采样；truncated 表示"结果未完整留存"——
    行数超过采样上限，或网关的取回上限（row_cap）已命中。
    """
    if not gateway_ret.get("success"):
        return
    columns = gateway_ret.get("columns") or []
    data = gateway_ret.get("data") or []
    capped = bool(gateway_ret.get("rows_capped"))
    result.result_columns = list(columns)
    result.result_rows = data[:max(0, rows_limit)]
    result.result_rows_truncated = capped or len(data) > rows_limit
    result.rows_capped = capped


class AdmissionRecord(BaseModel):
    """准入终态记录（跨库聚合的最小统计单元）。"""

    question_id: str
    decision: AdmissionDecision
    reason: str = ""
    database_id: str = ""
    knowledge_snapshot_id: str = ""
    final_execution_success: Optional[bool] = None
    diagnosis_iterations: int = 0
    correction_rounds: int = 0
