"""契约层统一出口（app/contracts）。

跨层只 import 本包，不深入子模块路径，便于未来内部重组。
"""

from app.contracts.database import (
    DatabaseContext,
    DatabaseDialect,
    DatabaseSnapshot,
    DatabaseSource,
)
from app.contracts.diagnosis import (
    SEMANTIC_CLASS_EVIDENCE,
    Correction,
    CorrectionRecord,
    Diagnosis,
    DiagnosisIteration,
    DiagnosisTrace,
    Error,
    ErrorCategory,
    ErrorLocation,
    ErrorType,
    Evidence,
    EvidenceBundle,
    ReflectionTrace,
    ReviewStatus,
    ReviewedDiagnosis,
    ReviewedDiagnosisSet,
    SampleDecision,
    SemanticClass,
)
from app.contracts.execution import (
    AdmissionDecision,
    AdmissionRecord,
    SQLExecutionResult,
)
from app.contracts.candidate import (
    CandidateProvenance,
    CandidateSample,
    ColumnOperation,
    Question,
    Rationale,
    SQLStrategy,
    TableSelection,
    Triple,
)
from app.contracts.knowledge import (
    ColumnInfo,
    ColumnSemantics,
    CrossTableRelation,
    DomainKnowledge,
    FieldCategory,
    FieldClassification,
    KnowledgeLayer,
    KnowledgeSnapshot,
    SchemaMetadata,
    TableInfo,
    TableSemantics,
)
from app.contracts.scenario import (
    ComplexityLevel,
    GenerationMetadata,
    GenerationScenario,
)
from app.contracts.legacy import (
    EntityType,
    PredicateType,
    SemanticTriple,
    TripleCollection,
    create_triple,
    create_triple_collection,
)

__all__ = [
    # database
    "DatabaseContext", "DatabaseDialect", "DatabaseSnapshot", "DatabaseSource",
    # knowledge
    "ColumnInfo", "ColumnSemantics", "CrossTableRelation", "DomainKnowledge",
    "FieldCategory", "FieldClassification", "KnowledgeLayer",
    "KnowledgeSnapshot", "SchemaMetadata", "TableInfo", "TableSemantics",
    # candidate
    "CandidateProvenance", "CandidateSample", "ColumnOperation", "Question",
    "Rationale", "SQLStrategy", "TableSelection", "Triple",
    # diagnosis
    "SEMANTIC_CLASS_EVIDENCE", "Correction", "CorrectionRecord", "Diagnosis",
    "DiagnosisIteration", "DiagnosisTrace", "Error", "ErrorCategory",
    "ErrorLocation", "ErrorType", "Evidence", "EvidenceBundle",
    "ReflectionTrace", "ReviewStatus", "ReviewedDiagnosis",
    "ReviewedDiagnosisSet", "SampleDecision", "SemanticClass",
    # execution
    "AdmissionDecision", "AdmissionRecord", "SQLExecutionResult",
    # scenario
    "ComplexityLevel", "GenerationMetadata", "GenerationScenario",
    # legacy
    "EntityType", "PredicateType", "SemanticTriple", "TripleCollection",
    "create_triple", "create_triple_collection",
]
