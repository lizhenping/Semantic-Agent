"""SemanticSQL Agent CLI（cli.py）

知识引导的 Text-to-SQL 数据合成框架——论文 §III 三阶段固定流水线的薄入口。

命令 ↔ 论文阶段：
  analyze        ↔  Phase 1 Analysis  (DA, K1..K6)
  generate       ↔  Phase 2 Generation (DS, q/r/s 三元组)
  diagnose       ↔  Phase 3 Reflection (DT, Eq.4 循环)
  run            ↔  单库三阶段串联
  run-benchmark  ↔  整个 benchmark 的批处理（自动跳过 dev/test 库）
  list-dbs       ↔  查看某 benchmark 可用 train 库

数据库定位：{db_root}/{benchmark}/databases/{database}/{database}.sqlite
db_root 默认 data/（指向 41G benchmark 数据集的挂载点），--db-root 可覆盖。
产物布局（doc/03-physical-model.md §6）：
  runs/<benchmark>/<database>/<session>/{manifest,knowledge/,candidates.*,reflection,training.accepted}
  knowledge/snapshots/<benchmark>/<database>/<snapshot_id>/
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import click

from app.bootstrap import Bootstrap
from app.contracts.candidate import CandidateSample
from app.contracts.database import DatabaseDialect
from app.settings import Settings


# ----------------------------------------------------------------
# 日志
# ----------------------------------------------------------------
def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handlers = [logging.StreamHandler()]
    log_file = os.getenv("LOG_FILE")
    if log_file:
        from logging.handlers import RotatingFileHandler
        handlers.append(RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024,
            backupCount=5, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


# ----------------------------------------------------------------
# 数据库路径解析
# ----------------------------------------------------------------
def _resolve_sqlite_path(
    bootstrap: Bootstrap,
    benchmark: Optional[str],
    database: Optional[str],
    sqlite: Optional[str],
    allow_eval: bool = False,
) -> str:
    """把 (benchmark, database) 或 --sqlite 解析成 sqlite 文件路径。

    数据泄露防护：默认拒绝评估（dev/test）库（--allow-eval 仅限调试）。
    """
    if sqlite:
        if not Path(sqlite).exists():
            raise click.ClickException(f"sqlite 文件不存在: {sqlite}")
        return sqlite

    if not (benchmark and database):
        raise click.ClickException(
            "需要 --benchmark + --database，或直接 --sqlite 指定数据库文件")

    if not allow_eval:
        from app.adapters.dataset_split import DatasetSplitError, assert_train_only
        try:
            assert_train_only(benchmark, database)
        except DatasetSplitError as e:
            raise click.ClickException(str(e))

    path = (Path(bootstrap.settings.data_root) / benchmark / "databases"
            / database / f"{database}.sqlite")
    if not path.exists():
        raise click.ClickException(
            f"数据库文件不存在: {path}\n"
            f"检查 --db-root（当前: {bootstrap.settings.data_root}）/ "
            f"--benchmark / --database 是否正确")
    return str(path)


def _open_context(bootstrap: Bootstrap, sqlite_path: str,
                  database_name: Optional[str], benchmark: str):
    source = bootstrap.sqlite_source(sqlite_path, database_name, benchmark)
    return bootstrap.gateway.open(source)


def _analysis(bootstrap: Bootstrap, benchmark: str, database: str,
              context, reuse_knowledge: bool, rebuild_knowledge: bool = False):
    """加载或构建知识快照。

    跳过判据（从简）：knowledge/snapshots/<benchmark>/<database>/ 下存在
    K1-K6 六层文件齐全的快照目录 → 复用；缺任何一层 → 删除该库全部
    快照目录后整体重建（resume）。
    """
    analysis = bootstrap.analysis_service(benchmark, database)
    repo = bootstrap.snapshot_repository()
    if rebuild_knowledge:
        removed = repo.purge_snapshot_dirs(benchmark, database,
                                           only_incomplete=False)
        if removed:
            click.echo(f"🧹 已清理 {removed} 个旧快照目录，强制重建")
    else:
        if repo.find_complete_snapshot_dir(benchmark, database) is not None:
            snapshot = analysis.load_latest(benchmark, database)
            if snapshot is not None:
                click.echo("♻️  复用完整知识快照 "
                           f"{snapshot.snapshot_id}（digest={snapshot.content_digest()}）")
                return snapshot
        removed = repo.purge_snapshot_dirs(benchmark, database,
                                           only_incomplete=True)
        if removed:
            click.echo(f"🧹 清理 {removed} 个不完整快照目录，重新构建")
    snapshot = analysis.analyze(context)
    analysis.publish(context, snapshot, benchmark=benchmark)
    return snapshot


# ----------------------------------------------------------------
# CLI 主组
# ----------------------------------------------------------------
@click.group()
@click.version_option(version="0.5.0")
def cli():
    """SemanticSQL Agent — 知识引导的 Text-to-SQL 合成框架

    固定三阶段流水线（论文 §III）：
      Phase 1 Analysis → Phase 2 Generation → Phase 3 Reflection
    """


@cli.command("list-dbs")
@click.option("--benchmark", "-b", required=True, help="benchmark 名")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
def list_dbs(benchmark, db_root):
    """列出某 benchmark 可用于合成的 train 库（排除 dev/test，防数据泄露）"""
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    from app.adapters.dataset_split import EVAL_DBS, list_train_databases
    train_dbs = list_train_databases(benchmark, bootstrap.settings.data_root)
    eval_dbs = sorted(EVAL_DBS.get(benchmark, set()))
    click.echo(f"📊 {benchmark} 可合成库（train，{len(train_dbs)} 个）：")
    for db in train_dbs:
        click.echo(f"  ✅ {db}")
    if eval_dbs:
        click.echo(f"\n🚫 评估库（dev/test，禁止合成，{len(eval_dbs)} 个）：")
        for db in eval_dbs:
            click.echo(f"  ⛔ {db}")


# ----------------------------------------------------------------
# Phase 1 并行预构建：build-knowledge
# ----------------------------------------------------------------
@cli.command("build-knowledge")
@click.option("--benchmark", "-b", required=True, help="benchmark 名")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--workers", "-w", default=10, type=int,
              help="并行构建的库数（默认 10）")
@click.option("--force", is_flag=True, help="忽略已有快照，全部重建")
@click.option("--limit", type=int, default=None, help="只处理前 N 个 train 库")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def build_knowledge(benchmark, db_root, workers, force, limit, debug):
    """跨库并行构建知识快照（Phase 1）：benchmark 下所有 train 库。

    跳过判据（从简）：knowledge/snapshots/<benchmark>/<db>/ 下存在
    K1-K6 六层文件齐全的快照目录 → 跳过；缺任何一层 → 删除该库全部
    快照目录后整体重建。库间并行，库内流程与 analyze 一致。

    \b
    示例：
      python cli.py build-knowledge -b bird -w 10
      python cli.py build-knowledge -b spider --force --limit 20
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    from concurrent.futures import ThreadPoolExecutor

    from app.adapters.dataset_split import list_train_databases
    dbs = list_train_databases(benchmark, bootstrap.settings.data_root)
    if limit:
        dbs = dbs[:limit]
    if not dbs:
        click.echo(f"⚠️ {benchmark} 无可用 train 库")
        return
    click.echo(f"🚀 并行构建 {benchmark} 知识快照：{len(dbs)} 个库，workers={workers}")
    analysis = bootstrap.analysis_service(benchmark)
    repo = bootstrap.snapshot_repository()

    def _build_one(db: str):
        try:
            if not force and repo.find_complete_snapshot_dir(benchmark, db) is not None:
                return db, "skipped", "K1-K6 齐全"
            removed = repo.purge_snapshot_dirs(benchmark, db,
                                               only_incomplete=not force)
            note = f"（清理 {removed} 个残缺目录）" if removed else ""
            sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, db, None)
            source = bootstrap.sqlite_source(sqlite_path, db, benchmark)
            ctx = bootstrap.gateway.open(source)
            try:
                snap = analysis.analyze(ctx)
                analysis.publish(ctx, snap, benchmark=benchmark)
                return db, "built", f"{snap.snapshot_id}{note}"
            finally:
                bootstrap.gateway.close(ctx)
        except Exception as e:  # noqa: BLE001
            return db, "failed", str(e)[:150]

    counts = {"built": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for db, status, info in pool.map(_build_one, dbs):
            counts[status] += 1
            icon = {"built": "✅", "skipped": "♻️", "failed": "❌"}[status]
            click.echo(f"  {icon} {db}: {info}")
    click.echo(f"📊 汇总：构建 {counts['built']}，跳过 {counts['skipped']}，"
               f"失败 {counts['failed']}")


# ----------------------------------------------------------------
# Phase 1 后置：profile-kb（实测 JOIN 画像侧车）
# ----------------------------------------------------------------
@cli.command("profile-kb")
@click.option("--benchmark", "-b", required=True, help="benchmark 名")
@click.option("--database", "-d", default=None, help="只处理单个库")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--workers", "-w", default=8, type=int, help="并行探测的库数")
@click.option("--max-pairs", default=40, type=int,
              help="每库最多探测的候选 JOIN 表对数")
@click.option("--limit", type=int, default=None, help="只处理前 N 个 train 库")
@click.option("--force", is_flag=True, help="已有画像也重新探测")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def profile_kb(benchmark, database, db_root, workers, max_pairs, limit, force,
               debug):
    """实测 JOIN 画像（命中率/脏键/基数）→ 快照目录 profile.jsonl 侧车。

    回答"这些表之间真的连得上吗"：K6 声明外键不可信的库（无 PK/FK 的
    SQLite 导出）靠实测命中率补齐覆盖判据；脏键（前导空格）与
    0 命中连接据此在生成期被拦下。属于 Phase 1 之后的可选步骤，
    缺失时生成流程自动降级（不报错）。

    \b
    示例：
      python cli.py profile-kb -b spider2 -w 8
      python cli.py profile-kb -b spider2 -d Db-IMDB --force --max-pairs 20
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    from concurrent.futures import ThreadPoolExecutor

    from app.adapters.dataset_split import list_train_databases
    from app.domain.data_profile import DataProfiler

    dbs = ([database] if database
           else list_train_databases(benchmark, bootstrap.settings.data_root))
    if limit:
        dbs = dbs[:limit]
    if not dbs:
        click.echo(f"⚠️ {benchmark} 无可用 train 库")
        return
    click.echo(f"🚀 实测 JOIN 画像：{benchmark} {len(dbs)} 个库，workers={workers}")
    analysis = bootstrap.analysis_service(benchmark)
    repo = bootstrap.snapshot_repository()

    def _profile_one(db: str):
        try:
            snapshot = repo.load_latest(benchmark, db)
            if snapshot is None:
                return db, "skipped", "无快照（先 build-knowledge）"
            if not force and repo.load_profile(benchmark, db,
                                               snapshot.snapshot_id):
                return db, "skipped", "画像已存在"
            sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, db, None)
            source = bootstrap.sqlite_source(sqlite_path, db, benchmark)
            ctx = bootstrap.gateway.open(source)
            try:
                profiler = DataProfiler(bootstrap.gateway, ctx,
                                        max_pairs=max_pairs)
                profiles = profiler.build(snapshot, snapshot.k1_schema)
                if profiles:
                    repo.save_profile(benchmark, db, snapshot.snapshot_id,
                                      [p.model_dump() for p in profiles])
                    unjoinable = sum(1 for p in profiles if not p.joinable)
                    trim = sum(1 for p in profiles if p.needs_trim)
                    return db, "built", (f"{len(profiles)} 条（不可连接 "
                                         f"{unjoinable}，需 TRIM {trim}）")
                return db, "built", "0 条候选"
            finally:
                bootstrap.gateway.close(ctx)
        except Exception as e:  # noqa: BLE001
            return db, "failed", str(e)[:150]

    counts = {"built": 0, "skipped": 0, "failed": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for db, status, info in pool.map(_profile_one, dbs):
            counts[status] += 1
            icon = {"built": "✅", "skipped": "♻️", "failed": "❌"}[status]
            click.echo(f"  {icon} {db}: {info}")
    click.echo(f"📊 汇总：探测 {counts['built']}，跳过 {counts['skipped']}，"
               f"失败 {counts['failed']}")


# ----------------------------------------------------------------
# Phase 1: analyze
# ----------------------------------------------------------------
@cli.command()
@click.option("--benchmark", "-b", help="benchmark 名（spider/bird/spider2/ehrsql/science_benchmark）")
@click.option("--database", "-d", help="数据库名（sqlite 文件的 stem）")
@click.option("--sqlite", help="直接指定 .sqlite 文件路径（覆盖 benchmark+database）")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--summary", "-s", is_flag=True, help="额外打印 K1..K6 摘要")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def analyze(benchmark, database, sqlite, db_root, summary, debug):
    """Phase 1 Analysis：抽取 K1..K6 知识快照（论文 §III.C）

    \b
    示例：
      python cli.py analyze -b spider -d aan_1
      python cli.py analyze --sqlite data/bird/databases/financial/financial.sqlite
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root

    sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, database, sqlite)
    db_name = Path(sqlite_path).stem
    bench = benchmark or ""
    click.echo(f"📊 Phase 1 Analysis: {sqlite_path}")

    context = _open_context(bootstrap, sqlite_path, db_name, bench)
    analysis = bootstrap.analysis_service(bench, db_name)
    snapshot = analysis.analyze(context)
    out = analysis.publish(context, snapshot, benchmark=bench)

    click.echo(f"✅ Phase 1 完成：知识快照 {snapshot.snapshot_id} -> {out}")
    if summary:
        for k, v in snapshot.summary().items():
            click.echo(f"   {k}: {v}")


# ----------------------------------------------------------------
# Phase 2: generate
# ----------------------------------------------------------------
@cli.command()
@click.option("--benchmark", "-b", help="benchmark 名")
@click.option("--database", "-d", help="数据库名")
@click.option("--sqlite", help="直接指定 .sqlite 文件路径")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--count", "-n", default=10, type=int, help="合成样本数")
@click.option("--output", "-o", help="训练数据 JSONL 输出路径（默认写入当前会话 training.accepted 前）")
@click.option("--reuse-knowledge", is_flag=True,
              help="（已默认启用）K1-K6 齐全即自动复用快照")
@click.option("--rebuild-knowledge", is_flag=True, help="忽略已有快照，强制重建 Phase 1")
@click.option("--seed", type=int, help="场景随机种子（可复现）")
@click.option("--no-execute", is_flag=True, help="跳过 SQL 执行反馈")
@click.option("--workers", "-w", default=None, type=int,
              help="样本级并发度（默认 settings.synthesis_workers=10）")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def generate(benchmark, database, sqlite, db_root, count, output,
             reuse_knowledge, rebuild_knowledge, seed, no_execute, workers,
             debug):
    """Phase 2 Generation：合成 (q, s, r) 训练三元组（论文 §III.D）

    \b
    示例：
      python cli.py generate -b spider -d aan_1 -n 50
      python cli.py generate --rebuild-knowledge -d aan_1 -n 50
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    if workers:
        bootstrap.settings.synthesis_workers = workers

    sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, database, sqlite)
    db_name = Path(sqlite_path).stem
    bench = benchmark or ""

    context = _open_context(bootstrap, sqlite_path, db_name, bench)
    snapshot = _analysis(bootstrap, bench, db_name, context,
                         reuse_knowledge, rebuild_knowledge)

    click.echo(f"▶ Phase 2 Generation（n={count}）...")
    bundle = bootstrap.pipeline_bundle(context, snapshot, bench, db_name, seed=seed)
    samples = bundle.synthesis.generate(count=count, execute=not no_execute)

    if output:
        _save_samples(samples, output)
        out_msg = output
    else:
        out_msg = bundle.run_store.session_dir()
    ok = sum(1 for s in samples if s.sql_result.execution_success)
    click.echo(f"✅ 完成：{len(samples)} 条三元组（{ok} 条 SQL 执行通过）→ {out_msg}")


# ----------------------------------------------------------------
# Phase 3: diagnose
# ----------------------------------------------------------------
@cli.command()
@click.option("--input", "-i", required=True, help="待诊断的训练数据 JSONL（Phase 2 产物）")
@click.option("--benchmark", "-b", help="benchmark 名")
@click.option("--database", "-d", help="数据库名")
@click.option("--sqlite", help="直接指定 .sqlite 文件路径")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--output", "-o", help="修正后输出路径（默认: <input>.corrected.jsonl）")
@click.option("--max-iters", default=3, type=int, help="Eq.4 修正循环最大迭代")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def diagnose(input, benchmark, database, sqlite, db_root, output, max_iters, debug):
    """Phase 3 Reflection：Diagnose→Retrieve→Correct 循环（论文 §III.E, Eq.4）

    复用已有知识快照 + 已有 Phase 2 三元组，只跑 Phase 3。
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root

    sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, database, sqlite)
    db_name = Path(sqlite_path).stem
    bench = benchmark or ""

    context = _open_context(bootstrap, sqlite_path, db_name, bench)
    snapshot = _analysis(bootstrap, bench, db_name, context, reuse_knowledge=True)

    samples = _load_samples(input)
    if not samples:
        click.echo(f"❌ 未从 {input} 加载到任何样本，请先运行 generate")
        sys.exit(1)
    click.echo(f"📖 加载 {len(samples)} 条待诊断样本")

    click.echo(f"▶ Phase 3 Reflection（max_iters={max_iters}）...")
    bundle = bootstrap.pipeline_bundle(context, snapshot, bench, db_name)
    refined, _traces = bundle.reflection.refine(samples, max_iters=max_iters)

    out_path = output or (str(Path(input).with_suffix("")) + ".corrected.jsonl")
    accepted = [s for s in refined
                if s.rationale.admission_decision == "accepted"]
    _save_samples(accepted, out_path)
    ok = sum(1 for s in refined if s.sql_result.execution_success)
    corrected = sum(1 for s in refined if s.rationale.correction_history)
    click.echo(f"✅ 完成：{len(accepted)}/{len(refined)} 条通过训练准入"
               f"（{ok} 条 SQL 执行通过），{corrected} 条经修正 → {out_path}")


# ----------------------------------------------------------------
# 全流程: run
# ----------------------------------------------------------------
@cli.command()
@click.option("--benchmark", "-b", help="benchmark 名")
@click.option("--database", "-d", help="数据库名")
@click.option("--sqlite", help="直接指定 .sqlite 文件路径")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--count", "-n", default=10, type=int, help="合成样本数")
@click.option("--max-iters", default=3, type=int, help="Eq.4 修正循环最大迭代")
@click.option("--output", "-o", help="最终训练数据 JSONL 路径（默认写入会话目录）")
@click.option("--reuse-knowledge", is_flag=True,
              help="（已默认启用）K1-K6 齐全即自动复用快照")
@click.option("--rebuild-knowledge", is_flag=True, help="忽略已有快照，强制重建 Phase 1")
@click.option("--seed", type=int, help="场景随机种子（可复现）")
@click.option("--workers", "-w", default=None, type=int,
              help="样本级并发度（默认 settings.synthesis_workers=10）")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def run(benchmark, database, sqlite, db_root, count, max_iters, output,
        reuse_knowledge, rebuild_knowledge, seed, workers, debug):
    """三阶段全流程：Analysis → Generation → Reflection

    \b
    示例：
      python cli.py run -b spider -d aan_1 -n 100
      python cli.py run -b spider -d aan_1 -n 100 -w 10
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    if workers:
        bootstrap.settings.synthesis_workers = workers

    sqlite_path = _resolve_sqlite_path(bootstrap, benchmark, database, sqlite)
    db_name = Path(sqlite_path).stem
    bench = benchmark or ""
    click.echo(f"▶ run: {sqlite_path}")

    context = _open_context(bootstrap, sqlite_path, db_name, bench)
    snapshot = _analysis(bootstrap, bench, db_name, context,
                         reuse_knowledge, rebuild_knowledge)
    click.echo("✅ Phase 1 完成")

    bundle = bootstrap.pipeline_bundle(context, snapshot, bench, db_name, seed=seed)

    click.echo(f"▶ Phase 2 Generation（n={count}）...")
    samples = bundle.synthesis.generate(count=count)
    click.echo(f"✅ Phase 2 完成（{len(samples)} 条）")

    click.echo(f"▶ Phase 3 Reflection（max_iters={max_iters}）...")
    samples, _traces = bundle.reflection.refine(samples, max_iters=max_iters)

    accepted = sum(1 for s in samples
                   if s.rationale.admission_decision == "accepted")
    if output:
        accepted_samples = [s for s in samples
                            if s.rationale.admission_decision == "accepted"]
        _save_samples(accepted_samples, output)
        out_msg = output
    else:
        out_msg = bundle.run_store.training_path()
    ok = sum(1 for s in samples if s.sql_result.execution_success)
    click.echo(f"✅ 完成：{accepted}/{len(samples)} 条通过训练准入"
               f"（{ok} 条执行通过）→ {out_msg}")


# ----------------------------------------------------------------
# 批量：run-benchmark
# ----------------------------------------------------------------
@cli.command("run-benchmark")
@click.option("--benchmark", "-b", required=True,
              help="benchmark 名（spider/bird/spider2/ehrsql/science_benchmark）")
@click.option("--count", "-n", default=None, type=int, help="每库合成样本数（与 --total 二选一）")
@click.option("--total", "-t", default=None, type=int,
              help="benchmark 合成总量，均分到每个 train 库")
@click.option("--max-iters", default=3, type=int, help="Eq.4 修正循环最大迭代")
@click.option("--db-root", default=None, help="数据库根目录（默认 data/）")
@click.option("--batch-size", default=25, type=int, help="每批次合成+落盘的样本数")
@click.option("--shard", type=int, help="分片编号（0 起），与 --num-shards 配合并行")
@click.option("--num-shards", type=int, help="总分片数；>1 时本 worker 只跑 shard::num_shards")
@click.option("--limit", type=int, help="只跑前 N 个库（调试用）")
@click.option("--workers", "-w", default=None, type=int,
              help="样本级并发度（默认 settings.synthesis_workers=10）")
@click.option("--debug", is_flag=True, help="启用 DEBUG 日志")
def run_benchmark(benchmark, count, total, max_iters, db_root, batch_size,
                  shard, num_shards, limit, workers, debug):
    """一键跑整个 benchmark：遍历所有 train 库，跳过 dev/test（防泄露）

    \b
    示例：
      python cli.py run-benchmark -b bird --total 30000
      python cli.py run-benchmark -b spider --total 40000 --limit 2 -w 10
    """
    _setup_logging(debug)
    bootstrap = Bootstrap()
    if db_root:
        bootstrap.settings.data_root = db_root
    bootstrap.settings.batch_size = batch_size
    if workers:
        bootstrap.settings.synthesis_workers = workers

    service = bootstrap.batch_service()
    try:
        summary = service.run_benchmark(
            benchmark, total=total, count=count, max_iters=max_iters,
            batch_size=batch_size, shard=shard, num_shards=num_shards,
            limit=limit)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"📊 汇总：{summary['databases']} 库，共 {summary['produced']} 条")
    if summary["failed"]:
        click.echo(f"⚠️  失败的库: {', '.join(summary['failed'])}")


# ----------------------------------------------------------------
# 审计：audit-summary
# ----------------------------------------------------------------
@cli.command("audit-summary")
@click.option("--trace-file", "trace_path", required=True,
              help="diagnosis_trace.jsonl（旧布局）或 reflection.jsonl（会话布局）路径")
def audit_summary(trace_path):
    """汇总反思裁决、错误分类和修正结果（不调用模型）。"""
    path = Path(trace_path)
    if not path.exists():
        raise click.ClickException(f"未找到反思轨迹: {path}")
    traces = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                traces.append(json.loads(line))
    if not traces:
        click.echo("没有可汇总的反思记录")
        return

    from collections import Counter
    decisions = Counter(t.get("decision", "unresolved") for t in traces)
    error_types: Counter = Counter()
    categories: Counter = Counter()
    detectors: Counter = Counter()
    semantic_classes: Counter = Counter()
    corrected = 0
    for trace in traces:
        iterations = trace.get("iterations", [])
        if any(item.get("action") == "corrected" for item in iterations):
            corrected += 1
        for item in iterations:
            for error in item.get("errors", []):
                error_types[error.get("type", "unknown")] += 1
                categories[error.get("category", "unknown")] += 1
                detectors[error.get("detector", "unknown")] += 1
                if error.get("semantic_class"):
                    semantic_classes[error["semantic_class"]] += 1

    total = len(traces)
    click.echo(f"样本数: {total}")
    click.echo("准入裁决: " + ", ".join(
        f"{name}={decisions[name]} ({decisions[name] / total:.1%})"
        for name in ("accepted", "rejected", "unresolved")))
    click.echo(f"曾进入修正: {corrected} ({corrected / total:.1%})")
    if semantic_classes:
        order = ["I-1", "I-2", "I-3", "I-4", "II-1", "II-2", "III-1"]
        click.echo("七类语义错误（论文 Table taxonomy）: " + ", ".join(
            f"{name}={semantic_classes[name]}" for name in order
            if semantic_classes[name]))
    if categories:
        click.echo("错误边界: " + ", ".join(
            f"{name}={count}" for name, count in sorted(categories.items())))
    if detectors:
        click.echo("检测来源: " + ", ".join(
            f"{name}={count}" for name, count in sorted(detectors.items())))
    if error_types:
        click.echo("错误类型:")
        for name, count in error_types.most_common():
            click.echo(f"  {name}: {count}")


# ----------------------------------------------------------------
# 样本加载/保存
# ----------------------------------------------------------------
def _load_samples(input_path: str) -> list[CandidateSample]:
    """从训练数据 JSONL 重建 list[CandidateSample]（diagnose 命令用）。"""
    from app.contracts.candidate import Question, Rationale, SQLStrategy
    from app.contracts.execution import SQLExecutionResult
    from app.contracts.scenario import GenerationMetadata

    path = Path(input_path)
    if not path.exists():
        return []
    samples: list[CandidateSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            think = rec.get("think", {}) or {}
            metadata = GenerationMetadata(**(think.get("metadata", {}) or {}))
            q = Question(
                question_id=rec.get("question_id", ""),
                text=rec.get("question", ""),
                metadata=metadata,
            )
            strategy_raw = think.get("sql_strategy", {}) or {}
            rationale = Rationale(
                focus=think.get("focus", ""),
                metadata=metadata,
                sql_strategy=SQLStrategy(
                    operations=strategy_raw.get("operations", []),
                    approach=strategy_raw.get("approach", ""),
                    no_need=strategy_raw.get("no_need", [])),
                expected_output=think.get("expected_output", ""),
            )
            s = SQLExecutionResult(sql=rec.get("answer", rec.get("sql", "")))
            samples.append(CandidateSample(question=q, sql_result=s,
                                           rationale=rationale))
    return samples


def _save_samples(samples: list[CandidateSample], output: str) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_training_record(), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    cli()
