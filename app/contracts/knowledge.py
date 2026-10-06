"""知识库契约（app/contracts/knowledge.py）

论文 §III.C 的 K = K1..K6 六层知识 + KnowledgeSnapshot（概念模型 §3.3/§3.4）。

字段名与既有 JSONL 产物（schema_extraction.jsonl 等）逐字对齐，
保证重构前后知识层内容可逐条比较（doc/03-physical-model.md §10）。

K1 是直接数据库事实；K2-K6 是模型推断（证据，不自动等于真理）。
"""

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# K 层标识（layer key 与 JSONL 文件名映射的唯一真相源见存储适配器）
# ============================================================

class KnowledgeLayer(str, Enum):
    """六层知识标识。value 与既有 JSONL 的 layer 名一致。"""

    K1_SCHEMA = "schema"       # K1: Schema Metadata
    K2_DOMAIN = "domain"       # K2: Domain Knowledge
    K3_FIELD = "field"         # K3: Field Classification
    K4_COLUMN = "column"       # K4: Column Semantics
    K5_TABLE = "table"         # K5: Table Semantics
    K6_RELATIONS = "er"        # K6: Cross-table Relations


# ============================================================
# K3 字段类型
# ============================================================

class FieldCategory(str, Enum):
    """K3 字段类型分类。

    语义校验关键：IDENTIFIER 不可聚合（Fig.1 的 AVG(CDSCode) 解法），
    MEASURE 才可 SUM/AVG。
    """

    IDENTIFIER = "identifier"   # 标识符，不可聚合
    MEASURE = "measure"         # 度量（可 SUM/AVG）
    DIMENSION = "dimension"     # 维度（可 GROUP BY）
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    OTHER = "other"


# ============================================================
# K1: Schema Metadata（直接数据库事实）
# ============================================================

class ColumnInfo(BaseModel):
    """K1 基础列元数据"""

    name: str
    data_type: str
    nullable: bool = True
    primary_key: bool = False
    foreign_key: bool = False
    foreign_key_table: Optional[str] = None
    foreign_key_column: Optional[str] = None
    default: Optional[str] = None
    comment: Optional[str] = None
    sample_values: list[Any] = Field(default_factory=list)
    entropy_level: str = "medium"  # low / medium / high（cardinality ratio）


class TableInfo(BaseModel):
    """K1 基础表元数据"""

    name: str
    columns: list[ColumnInfo] = Field(default_factory=list)
    primary_keys: list[str] = Field(default_factory=list)
    row_count: Optional[int] = None
    comment: Optional[str] = None


class SchemaMetadata(BaseModel):
    """K1 完整 schema（DatabaseSchema）"""

    database_name: str
    database_desc: str = ""
    tables: list[TableInfo] = Field(default_factory=list)

    def get_table(self, table_name: str) -> Optional[TableInfo]:
        for t in self.tables:
            if t.name == table_name:
                return t
        return None

    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnInfo]:
        t = self.get_table(table_name)
        if t is None:
            return None
        for c in t.columns:
            if c.name == column_name:
                return c
        return None

    def all_table_names(self) -> list[str]:
        return [t.name for t in self.tables]


# ============================================================
# K2: Domain Knowledge（域约束，推断）
# ============================================================

class DomainKnowledge(BaseModel):
    """K2 域知识"""

    domain_type: str = ""
    description: str = ""
    business_concepts: list[str] = Field(default_factory=list)
    naming_patterns: list[str] = Field(default_factory=list)


# ============================================================
# K3: Field Classification（字段类型，推断）
# ============================================================

class FieldClassification(BaseModel):
    """K3 单字段分类"""

    table_name: str
    column_name: str
    category: FieldCategory
    confidence: float = 0.8          # 仅承载 K3 分类置信度一种含义
    reasoning: str = ""
    data_type: Optional[str] = None

    @property
    def field_key(self) -> str:
        return f"{self.table_name}.{self.column_name}"


# ============================================================
# K4: Column Semantics（列语义，推断）
# ============================================================

class ColumnSemantics(BaseModel):
    """K4 列语义描述"""

    table_name: str
    column_name: str
    description: str
    confidence: float = 0.8
    source: str = "generated"      # "existing" | "generated" | "corrected"
    corrected: bool = False
    correction_reason: Optional[str] = None
    field_category: Optional[FieldCategory] = None
    data_type: Optional[str] = None
    is_nullable: Optional[bool] = None
    is_primary_key: Optional[bool] = None

    @property
    def field_key(self) -> str:
        return f"{self.table_name}.{self.column_name}"


# ============================================================
# K5: Table Semantics（表语义，推断）
# ============================================================

class TableSemantics(BaseModel):
    """K5 表语义描述"""

    table_name: str
    description: str
    business_type: str = "data_table"  # entity/relation/config/log/data_table
    confidence: float = 0.8
    key_columns: list[str] = Field(default_factory=list)


# ============================================================
# K6: Cross-table Relations（跨表关系，推断）
# ============================================================

class CrossTableRelation(BaseModel):
    """K6 跨表关系（ER）。

    统一表示物理外键和推断的概念关系：source_table.source_column
    → target_table.target_column。推断关系不得伪装成声明外键。
    """

    source_table: str
    source_column: Optional[str] = None
    target_table: str
    target_column: Optional[str] = None
    relationship_type: str = "many_to_one"
    confidence: float = 0.5
    reason: str = ""

    def involves(self, table_name: str) -> bool:
        return table_name in (self.source_table, self.target_table)


# ============================================================
# 实测数据画像（JOIN 命中率 / 键规范化 / 基数）
# ============================================================

class JoinProfile(BaseModel):
    """一对候选 JOIN 键的实测画像（Phase 1 后的旁路探测，不入 K1-K6 摘要）。

    命中统计基于**源列键值抽样**（成本与表规模无关，避免大表 TRIM 全量连接超时）：
    抽样 N 个去重源值 → 逐个探测在目标列是否存在（原生 / TRIM 后）。

    用于：① K6 回填（可连接的候选提升为可执行关系）
          ② SQL 提示词的覆盖警告（"抽样值中 N/M 能连上"）
          ③ check_joins 的覆盖判据扩展与 JOIN_UNJOINABLE 诊断。
    """

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    source_rows: int = 0             # 源表行数（K1 实测）
    sample_size: int = 0             # 抽样探测的源键值个数
    sample_hits: int = 0             # 原生（不 TRIM）能连上的样本值个数
    trim_hits: Optional[int] = None  # TRIM 归一化后能连上的样本值个数
    distinct_target: Optional[int] = None   # 目标键去重数
    target_unique: Optional[bool] = None    # 目标键是否唯一（N:1 / N:N 判据）

    @property
    def hit_rate(self) -> float:
        if not self.sample_size:
            return 0.0
        return self.sample_hits / self.sample_size

    @property
    def needs_trim(self) -> bool:
        """原生全连不上、TRIM 后能连上 → 键含前导/尾随空格。"""
        return self.sample_hits == 0 and bool(self.trim_hits)

    @property
    def joinable(self) -> bool:
        return self.sample_hits > 0 or bool(self.trim_hits)

    @property
    def cardinality(self) -> str:
        return "many_to_one" if self.target_unique is True else "many_to_many"

    def label(self) -> str:
        """提示词用的紧凑描述（覆盖警告 / 脏键提示 / 不可连接禁令）。"""
        pair = (f"{self.source_table}.{self.source_column} ⋈ "
                f"{self.target_table}.{self.target_column}")
        if not self.joinable:
            return (f"{pair}：抽样 {self.sample_size} 个键值全部连不上——"
                    f"该连接对绝大多数记录不成立，禁止在此连接上生成问题")
        pct = self.hit_rate * 100
        note = (f"实测可连接覆盖 ≈{pct:.0f}%（抽样 {self.sample_size} 个 "
                f"{self.source_table}.{self.source_column} 取值）")
        if self.needs_trim:
            note += ("；⚠ 键值含前导/尾随空格，JOIN 必须写 "
                     f"TRIM({self.source_table}.{self.source_column}) = "
                     f"TRIM({self.target_table}.{self.target_column})")
        elif self.hit_rate < 0.5:
            note += "，存在大量连不上的记录，ROUND 需谨慎"
        return f"{pair}：{note}"


# ============================================================
# KnowledgeSnapshot（概念模型 §3.4）
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeSnapshot(BaseModel):
    """知识库的不可变版本引用（发布后不得原地修改）。

    生成和反思必须使用同一知识快照；重新构建知识必须形成新快照。
    每条 accepted 样本都要能回溯到这里的 snapshot_id。
    """

    snapshot_id: str
    database_id: str
    dialect: str = "sqlite"
    created_at: str = Field(default_factory=_utc_now_iso)
    # 版本溯源（概念模型 §3.4：模型/Prompt/算法版本纳入摘要）
    builder_version: str = "1.0.0"
    model_id: str = ""
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    sampling_policy: str = ""
    # --- K1..K6 内容 ---
    k1_schema: SchemaMetadata
    k2_domain: DomainKnowledge = Field(default_factory=DomainKnowledge)
    k3_field_types: list[FieldClassification] = Field(default_factory=list)
    k4_column_semantics: list[ColumnSemantics] = Field(default_factory=list)
    k5_table_semantics: list[TableSemantics] = Field(default_factory=list)
    k6_relations: list[CrossTableRelation] = Field(default_factory=list)

    # ---------- 查询便捷方法（只读） ----------

    def get_column(self, table_name: str, column_name: str) -> Optional[ColumnSemantics]:
        for c in self.k4_column_semantics:
            if c.table_name == table_name and c.column_name == column_name:
                return c
        return None

    def get_field_type(self, table_name: str, column_name: str) -> Optional[FieldCategory]:
        for f in self.k3_field_types:
            if f.table_name == table_name and f.column_name == column_name:
                return f.category
        return None

    def get_table_semantic(self, table_name: str) -> Optional[TableSemantics]:
        for t in self.k5_table_semantics:
            if t.table_name == table_name:
                return t
        return None

    def foreign_keys(self) -> list[CrossTableRelation]:
        """K6 外键子集（推断关系含 many_to_one/one_to_many）。"""
        return [r for r in self.k6_relations
                if r.relationship_type in ("many_to_one", "one_to_many")]

    # ---------- 版本摘要 ----------

    def content_digest(self) -> str:
        """K1-K6 内容 + 版本信息的稳定摘要（16 hex）。

        同一输入与同一模型响应 -> 同一摘要；快照变化必然体现。
        """
        payload = {
            "builder_version": self.builder_version,
            "model_id": self.model_id,
            "prompt_versions": self.prompt_versions,
            "k1": self.k1_schema.model_dump(),
            "k2": self.k2_domain.model_dump(),
            "k3": [f.model_dump() for f in self.k3_field_types],
            "k4": [c.model_dump() for c in self.k4_column_semantics],
            "k5": [t.model_dump() for t in self.k5_table_semantics],
            "k6": [r.model_dump() for r in self.k6_relations],
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def summary(self) -> dict:
        """K 的摘要统计（调试/manifest 用）"""
        return {
            "database": self.database_id,
            "snapshot_id": self.snapshot_id,
            "content_digest": self.content_digest(),
            "K1_schema": f"{len(self.k1_schema.tables)} 表",
            "K2_domain": self.k2_domain.domain_type or "未设置",
            "K3_field_types": f"{len(self.k3_field_types)} 字段",
            "K4_column_semantics": f"{len(self.k4_column_semantics)} 列",
            "K5_table_semantics": f"{len(self.k5_table_semantics)} 表",
            "K6_relations": f"{len(self.k6_relations)} 关系",
        }
