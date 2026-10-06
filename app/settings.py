"""应用配置入口（app/settings.py）

唯一的配置读取点（唯一允许读环境变量的模块）。
移植自旧 config/settings.py；Neo4j 配置已删除（风险修复 2：
数据库实现早已不支持 Neo4j）。
"""

import os

from pydantic import BaseModel, Field


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"false", "0", "no", "off"}


class Settings(BaseModel):
    """全局应用设置（bootstrap 的唯一输入）。"""

    # ---------- LLM ----------
    # 端点已锁定（2026-09-18 实测调通）：本机 SSH 隧道 8090 → 服务器 vLLM 8092。
    # 隧道断开先查 netstat :8090，再启动 D:/学术/tkde/code/ssh-proxy。
    llm_model: str = Field(
        default=os.getenv("SEMANTICSQL_LLM_MODEL", "Qwen3.8-27B"))
    llm_base_url: str = Field(
        default=os.getenv("SEMANTICSQL_LLM_BASE_URL", "http://127.0.0.1:8090/v1"))
    llm_api_key: str = Field(
        default=os.getenv("SEMANTICSQL_LLM_API_KEY", "empty"))
    llm_temperature: float = Field(
        default=_env_float("SEMANTICSQL_LLM_TEMPERATURE", 0.1))
    llm_max_tokens: int = Field(
        default=_env_int("SEMANTICSQL_LLM_MAX_TOKENS", 8000))
    # 用户要求：Qwen3 系列必须关闭 thinking（chat_template_kwargs.enable_thinking=false）
    llm_enable_thinking: bool = Field(
        default=_env_bool("SEMANTICSQL_LLM_ENABLE_THINKING", False))
    llm_timeout: int = 1200
    llm_max_retries: int = 2   # 隧道/vLLM 偶发断连自动重试（技术重试≠反思轮次）
    llm_fake: bool = Field(default=False,
                           description="注入 FakeChatGateway（离线测试用）")

    # ---------- 数据库（MySQL 模式默认值；SQLite 由 --sqlite 指定文件） ----------
    db_type: str = Field(default=os.getenv("SEMANTICSQL_DB_TYPE", "sqlite"))
    db_host: str = Field(default=os.getenv("SEMANTICSQL_DB_HOST", "127.0.0.1"))
    db_port: int = Field(default=_env_int("SEMANTICSQL_DB_PORT", 13306))
    db_database: str = Field(default=os.getenv("SEMANTICSQL_DB_DATABASE", "testdb"))
    db_username: str = Field(default=os.getenv("SEMANTICSQL_DB_USERNAME", "testuser"))
    db_password: str = Field(default=os.getenv("SEMANTICSQL_DB_PASSWORD", "testpass"))

    # ---------- 路径 ----------
    data_root: str = Field(
        default=os.getenv("SEMANTICSQL_DB_ROOT", "data"),
        description="benchmark 数据库根目录（项目内 data/，含 spider/bird/spider2/ehrsql/science_benchmark）")
    # SQL 执行结果行采样上限：training.accepted 的 execution.result_rows 只保留前缀采样
    result_rows_limit: int = Field(
        default=_env_int("SEMANTICSQL_RESULT_ROWS_LIMIT", 20),
        description="SQL 执行结果行采样上限（0 表示不采样，只记 result_count）")
    runs_root: str = Field(
        default=os.getenv("SEMANTICSQL_OUTPUT_ROOT", "runs"),
        description="运行产物根目录 runs/<benchmark>/<database>/<session>/")
    knowledge_root: str = Field(
        default=os.getenv("SEMANTICSQL_KNOWLEDGE_ROOT", "knowledge/snapshots"),
        description="知识快照根目录 knowledge/snapshots/<benchmark>/<database>/<snapshot_id>/")
    configs_dir: str = "configs"
    templates_dir: str = "prompts/templates"

    # ---------- 反思策略 ----------
    max_correction_iterations: int = 3   # 论文：三轮修正语义（至多四次完整验证）
    enable_llm_review: bool = True
    # 旧 SEMANTICSQL_SQL_NO_R=1 的消融开关
    include_rationale_hints: bool = Field(
        default=not _env_bool("SEMANTICSQL_SQL_NO_R", False))

    # ---------- 执行保护 ----------
    # 用户规则：SQL 执行超过 15s 视为超时，整个问题直接放弃（不进入修正循环）。
    # 实测（600 条语料去 LIMIT 复跑）：p50 14ms / p99 5.0s，仅 0.17% 触及 15s。
    sql_timeout: float = Field(
        default=_env_float("SEMANTICSQL_SQL_TIMEOUT", 15.0))
    domain_scan_timeout: float = Field(
        default=_env_float("SEMANTICSQL_DOMAIN_SCAN_TIMEOUT", 30.0))
    # 答案落盘执行的取回上限：不改写 SQL，逐行流式计数到该值即停并标记
    # rows_capped（结果不完整 → 准入拒绝）。内存占用只与 result_rows_limit 有关。
    execution_row_cap: int = Field(
        default=_env_int("SEMANTICSQL_EXECUTION_ROW_CAP", 100000),
        description="单次答案执行的取回行数上限（0 表示不设上限，不推荐）")

    # ---------- 批处理 ----------
    # 样本级并发度：Phase 2 生成与 Phase 3 反思的线程池大小（LLM 调用为 I/O 密集，
    # vLLM 支持并发推理）；Phase 1 跨库并行构建复用同一配置
    synthesis_workers: int = Field(
        default=_env_int("SEMANTICSQL_SYNTHESIS_WORKERS", 20),
        description="样本级并发队列深度（生成+反思线程池），经服务端压测定为 20")
    # 0 行结果自检：问-答组合在数据上不可满足时不进语料
    # （丢弃恒开；本项为重生成补位次数，负数 = 关闭自检调试用）
    empty_result_retries: int = Field(
        default=_env_int("SEMANTICSQL_EMPTY_RESULT_RETRIES", 2),
        description="0 行结果的重生成补位次数（负数 = 关闭自检）")
    # 表格子集裁剪：大库每次生成只带 N 张表的接地上下文（q 与 s 同子集），
    # 控制 KV 缓存占用与 prefill 成本（25 表全量 ≈25.5k tokens → 10 表 ≈8k）
    context_table_cap: int = Field(
        default=_env_int("SEMANTICSQL_CONTEXT_TABLE_CAP", 10),
        description="单次生成的提示词表数上限（0 = 不裁剪，全库上下文）")
    batch_size: int = 25

    # ---------- 溯源 ----------
    builder_version: str = "1.0.0"
    code_version: str = "0.5.0"
