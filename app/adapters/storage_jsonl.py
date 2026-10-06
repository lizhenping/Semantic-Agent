"""JSONL 存储适配器（app/adapters/storage_jsonl.py）

- JsonlKnowledgeStore：K 层记录读写（layer→文件名映射与旧产物一致，
  可直接读旧 history/ 目录做回归对比）。
- JsonlRunRecordStore：runs/<benchmark>/<database>/<session>/ 会话布局
  （manifest / knowledge/ / candidates / reflection / training）。
- InMemory 实现：单测用。

约束（doc/03-physical-model.md §6）：修正历史追加在反思记录中，
不覆盖写；training.accepted 只收 accepted。
"""

import hashlib
import json
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


# layer 名 -> 文件名映射（唯一真相源；与旧 JSONLKnowledgeStore 一致）
LAYER_FILES = {
    "schema":          "schema_extraction.jsonl",   # K1
    "domain":          "domain_analysis.jsonl",     # K2
    "field":           "field_analysis.jsonl",      # K3
    "column":          "column_analysis.jsonl",     # K4
    "table":           "table_analysis.jsonl",      # K5
    "er":              "er_analysis.jsonl",         # K6
    "questions":       "questions.jsonl",           # Phase 2 产物
    "sql":             "sql_results.jsonl",         # Phase 2 产物
    "diagnosis_trace": "diagnosis_trace.jsonl",     # 反思轨迹（旧布局）
    "corpus_manifest": "corpus_manifest.jsonl",     # 准入裁决（旧布局）
}

# 会话布局的 K 层文件（runs/<...>/knowledge/<layer>.jsonl）
SESSION_LAYER_FILES = {
    "k1_schema": "k1_schema.jsonl",
    "k2_domain": "k2_domain.jsonl",
    "k3_field": "k3_field.jsonl",
    "k4_column": "k4_column.jsonl",
    "k5_table": "k5_table.jsonl",
    "k6_relations": "k6_relations.jsonl",
}

# 实测数据画像侧车文件（刻意不进 SESSION_LAYER_FILES：不算完整性判据）
PROFILE_FILE = "profile.jsonl"


def _write_jsonl(path: Path, records: list[dict], mode: str = "a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("跳过损坏的 JSONL 行: %s", path)
    return records


class JsonlKnowledgeStore:
    """K 层 JSONL 存储（路径即根目录）。"""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, layer: str) -> Path:
        if layer not in LAYER_FILES:
            raise ValueError(f"未知知识层: {layer}（支持: {list(LAYER_FILES)}）")
        return self.root_dir / LAYER_FILES[layer]

    def load(self, layer: str) -> list[dict]:
        return _read_jsonl(self._path(layer))

    def save(self, layer: str, data: Union[list[dict], dict]) -> None:
        records = [data] if isinstance(data, dict) else list(data)
        _write_jsonl(self._path(layer), records, mode="w")

    def append(self, layer: str, record: dict) -> None:
        _write_jsonl(self._path(layer), [record], mode="a")

    def exists(self, layer: str) -> bool:
        p = self._path(layer)
        return p.exists() and p.stat().st_size > 0

    def list_layers(self) -> list[str]:
        return [layer for layer in LAYER_FILES if self.exists(layer)]


class JsonlRunRecordStore:
    """runs/<benchmark>/<database>/<session_id>/ 会话产物存储。

    session_id 缺省自动生成（UTC 时间戳 + 随机后缀），保证每轮产物独立、
    不互相覆盖（阶段 0 风险修复：旧实现同目录追加导致跨轮混淆）。
    """

    def __init__(
        self,
        runs_root: Union[str, Path],
        benchmark: str,
        database: str,
        session_id: Optional[str] = None,
    ):
        if session_id is None:
            session_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.benchmark = benchmark
        self.database = database
        self.session_id = session_id
        self._dir = Path(runs_root) / benchmark / database / session_id
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / "knowledge").mkdir(exist_ok=True)

    def session_dir(self) -> str:
        return str(self._dir)

    # ---------- manifest ----------

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        manifest.setdefault("benchmark", self.benchmark)
        manifest.setdefault("database", self.database)
        manifest.setdefault("session_id", self.session_id)
        manifest.setdefault("written_at", time.strftime("%Y-%m-%dT%H:%M:%S"))
        path = self._dir / "manifest.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

    # ---------- K 层（会话布局） ----------

    def write_knowledge_layer(self, layer: str, records: list[dict]) -> None:
        filename = SESSION_LAYER_FILES.get(layer)
        if filename is None:
            # 兼容 "schema"/"domain" 等旧 layer 名
            alias = {
                "schema": "k1_schema", "domain": "k2_domain",
                "field": "k3_field", "column": "k4_column",
                "table": "k5_table", "er": "k6_relations",
            }
            filename = SESSION_LAYER_FILES[alias[layer]]
        _write_jsonl(self._dir / "knowledge" / filename, records, mode="w")

    # ---------- 候选 / 反思 / 训练 ----------

    def append_candidates(self, candidates: list[dict], stage: str = "initial") -> None:
        if not candidates:
            return
        _write_jsonl(self._dir / f"candidates.{stage}.jsonl", candidates)

    def append_reflection(self, traces: list[dict]) -> None:
        if not traces:
            return
        _write_jsonl(self._dir / "reflection.jsonl", traces)

    def append_training(self, records: list[dict]) -> None:
        if not records:
            return
        _write_jsonl(self._dir / "training.accepted.jsonl", records)

    def count_training(self) -> int:
        path = self._dir / "training.accepted.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in open(path, encoding="utf-8") if line.strip())

    @property
    def training(self) -> list[dict]:
        """已落盘训练样本（与 InMemoryRunRecordStore.training 对齐）。"""
        return _read_jsonl(self._dir / "training.accepted.jsonl")

    def training_path(self) -> Path:
        return self._dir / "training.accepted.jsonl"


class InMemoryKnowledgeStore:
    """内存 K 层存储（单测用）。"""

    def __init__(self):
        self._data: dict[str, list[dict]] = {}

    def load(self, layer: str) -> list[dict]:
        return list(self._data.get(layer, []))

    def save(self, layer: str, data: Union[list[dict], dict]) -> None:
        records = [data] if isinstance(data, dict) else list(data)
        self._data[layer] = records

    def append(self, layer: str, record: dict) -> None:
        self._data.setdefault(layer, []).append(record)

    def exists(self, layer: str) -> bool:
        return bool(self._data.get(layer))

    def list_layers(self) -> list[str]:
        return [k for k, v in self._data.items() if v]


class InMemoryRunRecordStore:
    """内存会话产物存储（单测用），接口与 JsonlRunRecordStore 一致。"""

    def __init__(self, benchmark: str = "mem", database: str = "mem",
                 session_id: str = "mem-session"):
        self.benchmark = benchmark
        self.database = database
        self.session_id = session_id
        self.manifest: dict[str, Any] = {}
        self.knowledge: dict[str, list[dict]] = {}
        self.candidates: dict[str, list[dict]] = {}
        self.reflection: list[dict] = []
        self.training: list[dict] = []

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest.update(manifest)

    def write_knowledge_layer(self, layer: str, records: list[dict]) -> None:
        self.knowledge[layer] = records

    def append_candidates(self, candidates: list[dict], stage: str = "initial") -> None:
        self.candidates.setdefault(stage, []).extend(candidates)

    def append_reflection(self, traces: list[dict]) -> None:
        self.reflection.extend(traces)

    def append_training(self, records: list[dict]) -> None:
        self.training.extend(records)

    def count_training(self) -> int:
        return len(self.training)

    def session_dir(self) -> str:
        return f"mem://{self.benchmark}/{self.database}/{self.session_id}"


def digest_records(records: list[dict]) -> str:
    """记录列表的稳定摘要（知识快照落库校验用）。"""
    raw = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class KnowledgeSnapshotRepository:
    """知识快照仓储：knowledge/snapshots/<benchmark>/<database>/<snapshot_id>/。

    每个快照独立目录（manifest.json + K 层 JSONL），发布后只读；
    load_latest 按 manifest 的 created_at 取最新版本。
    """

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)

    def _snapshot_dir(self, benchmark: str, database: str,
                      snapshot_id: str) -> Path:
        return self.root_dir / benchmark / database / snapshot_id

    def save(self, snapshot) -> Path:
        out = self._snapshot_dir(snapshot.database_id.split("/")[0],
                                 snapshot.database_id.split("/")[-1],
                                 snapshot.snapshot_id)
        out.mkdir(parents=True, exist_ok=True)
        (out / "knowledge").mkdir(exist_ok=True)
        manifest = {
            "snapshot_id": snapshot.snapshot_id,
            "database_id": snapshot.database_id,
            "dialect": snapshot.dialect,
            "created_at": snapshot.created_at,
            "builder_version": snapshot.builder_version,
            "model_id": snapshot.model_id,
            "prompt_versions": snapshot.prompt_versions,
            "content_digest": snapshot.content_digest(),
            "summary": snapshot.summary(),
        }
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)

        layers = {
            "k1_schema": [snapshot.k1_schema.model_dump()],
            "k2_domain": [snapshot.k2_domain.model_dump()],
            "k3_field": [f.model_dump() for f in snapshot.k3_field_types],
            "k4_column": [c.model_dump() for c in snapshot.k4_column_semantics],
            "k5_table": [t.model_dump() for t in snapshot.k5_table_semantics],
            "k6_relations": [r.model_dump() for r in snapshot.k6_relations],
        }
        for layer, records in layers.items():
            _write_jsonl(out / "knowledge" / SESSION_LAYER_FILES[layer],
                         records, mode="w")
        return out

    # ---------- 实测数据画像（侧车，不参与 SESSION_LAYER_FILES 完整性判据） ----------

    def save_profile(self, benchmark: str, database: str, snapshot_id: str,
                     records: list[dict]) -> Path:
        """写 profile.jsonl（JOIN 命中率画像）。

        刻意不加入 SESSION_LAYER_FILES：既有快照不应因缺该文件被判"不完整"
        而触发整体重建。
        """
        out = self._snapshot_dir(benchmark, database, snapshot_id)
        (out / "knowledge").mkdir(parents=True, exist_ok=True)
        path = out / "knowledge" / PROFILE_FILE
        _write_jsonl(path, records, mode="w")
        return path

    def load_profile(self, benchmark: str, database: str,
                     snapshot_id: Optional[str] = None) -> list[dict]:
        """读最新（或指定）快照的画像；缺失返回空列表（消费方自动降级）。"""
        base = self.root_dir / benchmark / database
        if not base.exists():
            return []
        if snapshot_id:
            return _read_jsonl(self._snapshot_dir(benchmark, database,
                                                  snapshot_id)
                               / "knowledge" / PROFILE_FILE)
        latest = self._latest_snapshot_dir(benchmark, database)
        if latest is None:
            return []
        return _read_jsonl(latest / "knowledge" / PROFILE_FILE)

    def _latest_snapshot_dir(self, benchmark: str, database: str) -> Optional[Path]:
        base = self.root_dir / benchmark / database
        best: Optional[tuple[str, Path]] = None
        for d in base.iterdir() if base.exists() else []:
            manifest = d / "manifest.json"
            if not manifest.exists():
                continue
            try:
                meta = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            created = str(meta.get("created_at", ""))
            if best is None or created > best[0]:
                best = (created, d)
        return best[1] if best else None

    def load_latest(self, benchmark: str, database: str):
        """加载最新知识快照；不存在返回 None。"""
        from app.contracts.knowledge import (
            ColumnSemantics,
            CrossTableRelation,
            DomainKnowledge,
            FieldClassification,
            KnowledgeSnapshot,
            SchemaMetadata,
            TableSemantics,
        )
        base = self.root_dir / benchmark / database
        if not base.exists():
            return None
        best: tuple[str, Path] | None = None
        for d in base.iterdir():
            manifest = d / "manifest.json"
            if not manifest.exists():
                continue
            try:
                with open(manifest, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue
            created = str(meta.get("created_at", ""))
            if best is None or created > best[0]:
                best = (created, d)
        if best is None:
            return None
        d = best[1]

        def _layer(path: Path) -> list[dict]:
            return _read_jsonl(path)

        k1_records = _layer(d / "knowledge" / "k1_schema.jsonl")
        if not k1_records:
            return None
        snapshot_id = d.name
        with open(d / "manifest.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        return KnowledgeSnapshot(
            snapshot_id=snapshot_id,
            database_id=str(meta.get("database_id", database)),
            dialect=str(meta.get("dialect", "sqlite")),
            created_at=str(meta.get("created_at", "")),
            builder_version=str(meta.get("builder_version", "")),
            model_id=str(meta.get("model_id", "")),
            prompt_versions=dict(meta.get("prompt_versions", {})),
            k1_schema=SchemaMetadata(**k1_records[0]),
            k2_domain=DomainKnowledge(**(_layer(d / "knowledge" / "k2_domain.jsonl") or [{}])[0]),
            k3_field_types=[FieldClassification(**r) for r in _layer(d / "knowledge" / "k3_field.jsonl")],
            k4_column_semantics=[ColumnSemantics(**r) for r in _layer(d / "knowledge" / "k4_column.jsonl")],
            k5_table_semantics=[TableSemantics(**r) for r in _layer(d / "knowledge" / "k5_table.jsonl")],
            k6_relations=[CrossTableRelation(**r) for r in _layer(d / "knowledge" / "k6_relations.jsonl")],
        )

    # ---------- K1-K6 完整性检查（跳过/重建的简易判据） ----------

    def find_complete_snapshot_dir(self, benchmark: str,
                                   database: str) -> Optional[Path]:
        """返回 K1-K6 六层文件齐全的最新快照目录；不存在返回 None。"""
        base = self.root_dir / benchmark / database
        if not base.exists():
            return None
        best: Optional[tuple[str, Path]] = None
        for d in sorted(base.iterdir()):
            if not d.is_dir():
                continue
            kdir = d / "knowledge"
            if not all((kdir / f).exists() for f in SESSION_LAYER_FILES.values()):
                continue
            created = d.name   # snapshot_id 含时间戳，可直接排序兜底
            try:
                with open(d / "manifest.json", "r", encoding="utf-8") as f:
                    created = str(json.load(f).get("created_at", "")) or d.name
            except Exception:
                pass
            if best is None or created > best[0]:
                best = (created, d)
        return best[1] if best else None

    def purge_snapshot_dirs(self, benchmark: str, database: str,
                            only_incomplete: bool = True) -> int:
        """删除该库的快照目录（默认只删 K1-K6 不全的）；返回删除数。"""
        base = self.root_dir / benchmark / database
        if not base.exists():
            return 0
        removed = 0
        for d in base.iterdir():
            if not d.is_dir():
                continue
            complete = all((d / "knowledge" / f).exists()
                           for f in SESSION_LAYER_FILES.values())
            if only_incomplete and complete:
                continue
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
        return removed
