"""多数据库批处理用例（app/application/batch_service.py）

DatabaseCatalog：每个 DatabaseSource 独立 context / 快照 / 会话产物，
指标层聚合，绝不共享可变知识库或连接（doc/02 §4.2）。

移植自旧 cli.run-benchmark：总量均分、分片、断点续跑（按已落盘
accepted 条数判断）、分批生成+即时落盘、单库失败继续。
数据库之间串行（有界并发的外部并行由 --shard 分片实现，与旧方案一致）。
"""

import logging
from pathlib import Path
from typing import Callable, Optional

from app.ports.database import DatabaseGateway
from app.domain.candidate_generation import UnusableDatabaseError

logger = logging.getLogger(__name__)


def split_total(total: int, n_dbs: int) -> list[int]:
    """总量均分到每个库：count_i = total // n_dbs，前 (total % n_dbs) 个 +1。"""
    if n_dbs <= 0:
        return []
    base, extra = divmod(total, n_dbs)
    return [base + (1 if i < extra else 0) for i in range(n_dbs)]


class BatchService:
    """一键跑整个 benchmark 的所有 train 库。

    train_db_lister 由装配根注入（application 不直接依赖具体适配器）。
    """

    def __init__(
        self,
        settings,           # app.settings.Settings
        gateway: DatabaseGateway,
        pipeline_factory,   # callable(benchmark, db_name, sqlite_path) -> bundle
        train_db_lister: Callable[[str], list[str]],
    ):
        self.settings = settings
        self.gateway = gateway
        self.pipeline_factory = pipeline_factory
        self.train_db_lister = train_db_lister

    def run_benchmark(
        self,
        benchmark: str,
        total: Optional[int] = None,
        count: Optional[int] = None,
        max_iters: int = 3,
        batch_size: int = 25,
        shard: Optional[int] = None,
        num_shards: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> dict:
        """遍历 train 库逐库执行 Analysis → Generation → Reflection。"""
        train_dbs = self.train_db_lister(benchmark)
        if not train_dbs:
            raise RuntimeError(
                f"{benchmark} 没有可用的 train 库（检查 {self.settings.data_root}/"
                f"{benchmark}/databases/ 是否存在）")
        if limit:
            train_dbs = train_dbs[:limit]

        if total is not None:
            per_db_counts = split_total(total, len(train_dbs))
        else:
            per_db_counts = [count if count is not None else 10] * len(train_dbs)

        if num_shards is not None and num_shards > 1:
            if shard is None:
                raise ValueError("使用分片时必须同时指定 shard")
            if not (0 <= shard < num_shards):
                raise ValueError(f"shard 必须在 [0, {num_shards}) 内")
            train_dbs = train_dbs[shard::num_shards]
            per_db_counts = per_db_counts[shard::num_shards]

        logger.info("🚀 run-benchmark: %s，本片 %d 个库", benchmark, len(train_dbs))

        total_produced = 0
        failed_dbs: list[str] = []
        skipped_dbs: list[str] = []
        skipped_unusable: list[str] = []
        for i, (db_name, per_db) in enumerate(zip(train_dbs, per_db_counts), 1):
            have = self._count_training(benchmark, db_name)
            if have >= per_db:
                logger.info("[%d/%d] %s/%s 已有 %d/%d 条，跳过",
                            i, len(train_dbs), benchmark, db_name, have, per_db)
                skipped_dbs.append(db_name)
                continue

            sqlite_path = str(Path(self.settings.data_root) / benchmark
                              / "databases" / db_name / f"{db_name}.sqlite")
            logger.info("[%d/%d] %s/%s（已有 %d/%d 条）",
                        i, len(train_dbs), benchmark, db_name, have, per_db)
            try:
                produced = self._run_database(
                    benchmark, db_name, sqlite_path, per_db, have,
                    max_iters, batch_size)
                total_produced += produced
            except UnusableDatabaseError as e:
                logger.warning("  ⏭️  %s 跳过：%s", db_name, e)
                skipped_unusable.append(db_name)
            except Exception as e:  # noqa: BLE001
                logger.error("  ❌ %s 失败: %s", db_name, e)
                failed_dbs.append(db_name)

        summary = {
            "benchmark": benchmark,
            "databases": len(train_dbs),
            "produced": total_produced,
            "skipped": skipped_dbs,
            "skipped_unusable": skipped_unusable,
            "failed": failed_dbs,
        }
        logger.info("📊 汇总：%s", summary)
        return summary

    def _run_database(self, benchmark: str, db_name: str, sqlite_path: str,
                      per_db: int, have: int, max_iters: int,
                      batch_size: int) -> int:
        """单库执行（含分批续跑），返回本库新增条数。

        bundle.synthesis / bundle.reflection 各自持有同一会话的
        RunRecordStore；训练导出由 ReflectionService 统一落盘
        （training.accepted 只收 accepted），这里只负责计数。
        """
        bundle = self.pipeline_factory(benchmark, db_name, sqlite_path)
        # 库级停用硬规则：非空表不足 2 张（K1 row_count 已实测）→ 整库跳过，
        # 不消耗任何 LLM 算力（如 sdss 6 表中 5 表为空）
        nonempty = [t for t in bundle.snapshot.k1_schema.tables
                    if t.row_count != 0]
        if len(nonempty) < 2:
            raise UnusableDatabaseError(
                f"非空表 {len(nonempty)}/"
                f"{len(bundle.snapshot.k1_schema.tables)} 张，不具备提问空间")
        produced = have
        attempts = 0
        max_attempts = per_db * 3 + batch_size  # 安全阀：准入率过低时止损

        while produced < per_db and attempts < max_attempts:
            n = min(batch_size, per_db - produced)
            samples = bundle.synthesis.generate(count=n, execute=True)
            attempts += n
            if not samples:
                continue
            samples, _traces = bundle.reflection.refine(samples, max_iters=max_iters)
            produced = self._count_training(benchmark, db_name)
            accepted = sum(1 for s in samples
                           if s.rationale.admission_decision == "accepted")
            logger.info("  ✅ 批次完成：累计 %d/%d（本批 %d 条，%d 准入）",
                        produced, per_db, len(samples), accepted)

        if produced < per_db:
            logger.warning("  ⚠️ 达到尝试上限，仅 %d/%d 条（准入率过低）", produced, per_db)
        return produced - have

    def _count_training(self, benchmark: str, db_name: str) -> int:
        """跨会话累计该库已落盘的 accepted 训练条数（断点续跑依据）。"""
        root = Path(self.settings.runs_root) / benchmark / db_name
        if not root.exists():
            return 0
        count = 0
        for session in root.iterdir():
            path = session / "training.accepted.jsonl"
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    count += sum(1 for line in f if line.strip())
        return count
