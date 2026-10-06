"""候选生成用例（app/application/synthesis_service.py）

KnowledgeSnapshot + GenerationScenario -> CandidateSample(q, s, r)
（论文 Phase 2 / DS）。生成顺序固定：q 与 r 先于 s。
单条生成失败只影响当前样本，不中断批次。
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.contracts.candidate import CandidateSample, Question
from app.contracts.database import DatabaseContext
from app.contracts.knowledge import KnowledgeSnapshot
from app.domain.candidate_generation import UnusableDatabaseError
from app.ports.records import RunRecordStore

logger = logging.getLogger(__name__)


class SynthesisService:
    """q/r/s 生成用例。"""

    def __init__(
        self,
        snapshot: KnowledgeSnapshot,
        context: DatabaseContext,
        question_generator,
        sql_generator,
        run_store: Optional[RunRecordStore] = None,
        legacy_store=None,
        workers: int = 10,
        empty_result_retries: int = 2,
    ):
        self.snapshot = snapshot
        self.context = context
        self.question_generator = question_generator
        self.sql_generator = sql_generator
        self.run_store = run_store
        self.legacy_store = legacy_store
        # 样本级并发度（LLM 调用 I/O 密集；vLLM 支持并发推理）
        self.workers = max(1, int(workers))
        # 0 行结果自检：重生成补位次数（用尽仍 0 行 → 丢弃，不进反思/准入）
        self.empty_result_retries = max(0, int(empty_result_retries))
        # 逐条落盘锁：并发下每完成一条候选立即追加写入
        self._persist_lock = threading.Lock()

    def generate_questions(self, count: int, seed_rng=None) -> list[Question]:
        try:
            return self.question_generator.generate(self.snapshot, count)
        except UnusableDatabaseError as e:
            logger.warning("⏭️  该库不可用于合成：%s", e)
            return []

    def generate(self, count: int, execute: bool = True) -> list[CandidateSample]:
        """合成 count 个候选三元组（q → (s, r)）。

        执行失败不抛错（记入 SQLResult，进入诊断阶段处理）。
        样本间相互独立：workers>1 时线程池并发生成，结果保持输入顺序；
        每条候选完成即落盘（崩溃不丢已完成工作，进度实时可见）。
        库不可用（非空表不足）时不抛错，返回空列表由编排层跳过。
        """
        try:
            questions = self.question_generator.generate(self.snapshot, count)
        except UnusableDatabaseError as e:
            logger.warning("⏭️  该库不可用于合成：%s", e)
            return []
        logger.info("合成 %d 个问题，开始生成 SQL", len(questions))

        if self.workers > 1 and len(questions) > 1:
            samples = self._generate_parallel(questions, execute)
        else:
            samples = self._generate_serial(questions, execute)

        samples = self._refill_empty_results(samples, execute)
        logger.info("✅ 生成完成：%d/%d 条候选", len(samples), len(questions))
        return samples

    # ---------- 0 行结果自检（源头拦：数据前提不成立的不进反思/准入） ----------

    @staticmethod
    def _is_empty_result(sample: CandidateSample) -> bool:
        r = sample.sql_result
        return bool(r.executed and r.execution_success
                    and r.result_count == 0)

    def _refill_empty_results(self, samples: list[CandidateSample],
                              execute: bool) -> list[CandidateSample]:
        """0 行样本重生成补位；随后丢弃仍为空的（0 行不进语料）。

        未执行（execute=False）时无法判定，原样返回。
        empty_result_retries < 0 表示关闭自检（调试用）。
        被丢弃的样本仍留在 candidates.initial（崩溃可追溯），
        但不进入反思/准入，因此不会污染 training.accepted。
        """
        if not execute or self.empty_result_retries < 0 or not samples:
            return samples
        for attempt in range(1, self.empty_result_retries + 1):
            n_empty = sum(1 for s in samples if self._is_empty_result(s))
            if not n_empty:
                break
            logger.info("🔁 0 行结果 %d 条，第 %d/%d 次重生成补位",
                        n_empty, attempt, self.empty_result_retries)
            try:
                new_qs = self.question_generator.generate(self.snapshot, n_empty)
            except UnusableDatabaseError as e:
                logger.warning("补位失败（库不可用）：%s", e)
                break
            if not new_qs:
                break
            if self.workers > 1 and len(new_qs) > 1:
                fresh = self._generate_parallel(new_qs, execute)
            else:
                fresh = self._generate_serial(new_qs, execute)
            pool = iter(fresh)
            for i, s in enumerate(samples):
                if self._is_empty_result(s):
                    r = next(pool, None)
                    if r is None:
                        break
                    samples[i] = r
        kept = [s for s in samples if not self._is_empty_result(s)]
        dropped = len(samples) - len(kept)
        if dropped:
            logger.warning("⏭️  丢弃 %d 条 0 行结果样本（数据前提不成立，不进反思）",
                           dropped)
        return kept

    def _persist_one(self, sample: CandidateSample) -> None:
        """单条候选落盘（candidates.initial + 旧布局层），线程安全。"""
        with self._persist_lock:
            if self.run_store is not None:
                self.run_store.append_candidates(
                    [sample.model_dump(mode="json")], stage="initial")
            if self.legacy_store is not None:
                self.legacy_store.append("questions", {
                    "question_id": sample.question_id,
                    "question_text": sample.question.text,
                    "question_focus": sample.question.question_focus,
                    "business_rules": sample.question.business_rules,
                    "metadata": sample.question.metadata.model_dump(),
                })
                self.legacy_store.append("sql", {
                    "question_id": sample.question_id,
                    "sql": sample.sql_result.sql,
                    "dialect": sample.sql_result.dialect,
                    "executed": sample.sql_result.executed,
                    "execution_success": sample.sql_result.execution_success,
                    "result_count": sample.sql_result.result_count,
                    "execution_error": sample.sql_result.execution_error,
                })

    def _generate_serial(self, questions: list[Question],
                         execute: bool) -> list[CandidateSample]:
        samples: list[CandidateSample] = []
        for i, question in enumerate(questions, 1):
            try:
                sample = self.sql_generator.generate(self.snapshot, question,
                                                     execute=execute)
                samples.append(sample)
                self._persist_one(sample)
                logger.info("  [%d/%d] %s: %s", i, len(questions),
                            question.question_id,
                            "✅" if sample.sql_result.execution_success else "⚠️")
            except Exception as e:
                logger.warning("  [%d/%d] %s 生成失败: %s", i, len(questions),
                               question.question_id, e)
        return samples

    def _generate_parallel(self, questions: list[Question],
                           execute: bool) -> list[CandidateSample]:
        def _one(i: int, question: Question) -> Optional[CandidateSample]:
            try:
                sample = self.sql_generator.generate(self.snapshot, question,
                                                     execute=execute)
                self._persist_one(sample)   # 完成一条立即落盘
                logger.info("  [%d/%d] %s: %s", i, len(questions),
                            question.question_id,
                            "✅" if sample.sql_result.execution_success else "⚠️")
                return sample
            except Exception as e:
                logger.warning("  [%d/%d] %s 生成失败: %s", i, len(questions),
                               question.question_id, e)
                return None

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(_one, i, q)
                       for i, q in enumerate(questions, 1)]
            results = [f.result() for f in futures]
        return [s for s in results if s is not None]
