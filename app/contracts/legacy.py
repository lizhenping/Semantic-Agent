"""旧通用表示（app/contracts/legacy.py）

SemanticTriple 是历史遗留的"通用三元组"表示（原 models/schemas.py）。
按 doc/03-physical-model.md §1/§4：它不是候选样本（CandidateSample），
新业务代码禁止依赖；仅保留给旧审计/回放脚本读取历史产物。
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class PredicateType(Enum):
    """标准化谓词类型枚举（历史遗留）"""

    # 数据库结构关系
    HAS_TABLE = "has_table"
    HAS_COLUMN = "has_column"
    HAS_CONSTRAINT = "has_constraint"
    HAS_INDEX = "has_index"
    HAS_RELATIONSHIP = "has_relationship"

    # 业务语义关系
    BELONGS_TO = "belongs_to"
    CONTAINS = "contains"
    REFERENCES = "references"
    DERIVED_FROM = "derived_from"
    SIMILAR_TO = "similar_to"

    # 分析结果关系
    FIELD_TYPE = "field_type"
    BUSINESS_MEANING = "business_meaning"
    BUSINESS_ROLE = "business_role"
    CORE_ENTITY = "core_entity"

    # 生成内容关系
    GENERATES_QUESTION = "generates_question"
    GENERATES_SQL = "generates_sql"
    CORRESPONDS_TO = "corresponds_to"
    EXECUTES_WITH = "executes_with"

    # 质量评估关系
    QUALITY_SCORE = "quality_score"
    VALIDATION_RESULT = "validation_result"
    REFLECTION_RESULT = "reflection_result"


class EntityType(Enum):
    """实体类型枚举（历史遗留）"""

    DATABASE = "Database"
    TABLE = "Table"
    COLUMN = "Column"
    DOMAIN = "Domain"
    ENTITY = "Entity"
    QUESTION = "Question"
    SQL = "SQL"
    SCENARIO = "Scenario"
    OPERATION = "Operation"
    RESULT = "Result"


class SemanticTriple(BaseModel):
    """语义三元组（历史遗留表示，禁止新业务依赖）。

    其 confidence 字段历史上承载过 K 置信度、检索分、语义正确率和
    准入分四种含义（doc/03-physical-model.md §9 风险 7），
    新代码请分别使用各契约对象自己的字段。
    """

    subject: str = Field(description="主体实体")
    predicate: str = Field(description="关系谓词")
    object: str = Field(description="客体实体")

    subject_type: str = Field(default="Entity", description="主体类型")
    object_type: str = Field(default="Entity", description="客体类型")
    confidence: Optional[float] = Field(default=None, description="置信度(0-1)")
    source_tool: str = Field(default="", description="来源工具名称")

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8], description="会话ID"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(), description="创建时间戳"
    )

    def to_simple_tuple(self) -> Tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def __str__(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"

    def __repr__(self) -> str:
        return (
            f"SemanticTriple({self.subject}, {self.predicate}, {self.object}, "
            f"source={self.source_tool})"
        )


class TripleCollection(BaseModel):
    """三元组集合（历史遗留容器）"""

    triples: List[SemanticTriple] = Field(default_factory=list)
    source_tool: str = Field(description="生成工具名称")
    generation_context: Dict[str, Any] = Field(default_factory=dict)
    summary: str = Field(default="", description="集合摘要描述")

    def add_triple(
        self,
        subject: str,
        predicate: str,
        object: str,
        subject_type: str = "Entity",
        object_type: str = "Entity",
        confidence: Optional[float] = None,
    ) -> SemanticTriple:
        triple = SemanticTriple(
            subject=subject,
            predicate=predicate,
            object=object,
            subject_type=subject_type,
            object_type=object_type,
            confidence=confidence,
            source_tool=self.source_tool,
        )
        self.triples.append(triple)
        return triple

    def filter_by_predicate(self, predicate: str) -> List[SemanticTriple]:
        return [t for t in self.triples if t.predicate == predicate]

    def filter_by_subject(self, subject: str) -> List[SemanticTriple]:
        return [t for t in self.triples if t.subject == subject]

    def count(self) -> int:
        return len(self.triples)

    def to_simple_list(self) -> List[Tuple[str, str, str]]:
        return [t.to_simple_tuple() for t in self.triples]

    def merge(self, other: "TripleCollection") -> "TripleCollection":
        merged = TripleCollection(
            source_tool=f"{self.source_tool}+{other.source_tool}",
            summary=f"Merged: {self.summary} + {other.summary}",
        )
        merged.triples = self.triples + other.triples
        return merged


def create_triple(
    subject: str, predicate: str, object: str, source_tool: str = "", **kwargs
) -> SemanticTriple:
    """快速创建三元组的便利函数（历史遗留）"""
    return SemanticTriple(
        subject=subject,
        predicate=predicate,
        object=object,
        source_tool=source_tool,
        **kwargs,
    )


def create_triple_collection(source_tool: str, summary: str = "") -> TripleCollection:
    """快速创建三元组集合的便利函数（历史遗留）"""
    return TripleCollection(source_tool=source_tool, summary=summary)
