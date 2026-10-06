"""数据集隔离适配器（app/adapters/dataset_split.py）

防止数据泄露（data leakage）：合成训练数据绝不能使用评估（dev/test）库。
移植自旧 infra/dataset_split.py，黑名单与策略保持不变。

隔离策略（按 benchmark 分三类）：
1. spider / bird：维护 dev/test 库黑名单，合成时拒绝
2. science_benchmark：train 与 dev 共用库（按 QA 切分）→ 库级不隔离，
   管线只从 schema 合成、不读官方 QA，天然隔离
3. ehrsql / spider2：同上
"""

from pathlib import Path


class DatasetSplitError(Exception):
    """数据隔离违规：试图在评估库上合成训练数据"""


EVAL_DBS: dict[str, set[str]] = {
    "spider": {
        # dev.json 的 20 个库
        "battle_death", "car_1", "concert_singer", "course_teach",
        "cre_Doc_Template_Mgt", "dog_kennels", "employee_hire_evaluation",
        "flight_2", "museum_visit", "network_1", "news_report",
        "perpetual_student", "riding_club", "sports_competition",
        "voter_1", "world_1", "tvshow", "student_1", "storm_record",
        "tracking_software_problems",
        # test.json 的 40 个库
        "documents_products", "gas_company", "game_injury", "apartment_rentals",
        "loan_1", "student_2", "tracking_grants_for_research", "shipping_records",
        "assets_maintenance", "bike_1", "manufacturer", "climbing",
        "railway", "icfp_1", "local_govt_in_detail", "race_track",
        "insurance_policies", "farm", "musical", "games_1",
        "program_exhibition", "museum", "preschool", "sports_1",
        "public_review", "wrestler", "ship_1", "hockey_1",
        "codebase_subsite", "concert_singer", "debate", "film_rank",
        "feeding_the_homeless", "architecture", "company_1", "device",
        "election", "school_busines", "coffee_1", "swimming",
    },
    "bird": {
        # dev.json 的 11 个库
        "california_schools", "card_games", "codebase_community",
        "debit_card_specializing", "european_football_2", "financial",
        "formula_1", "student_club", "superhero", "thrombosis_prediction",
        "toxicology",
    },
    # train 与 eval 共用库的 benchmark：库级无法隔离，靠管线不读官方 QA
    "science_benchmark": set(),
    "ehrsql": set(),
    "spider2": set(),
}


def assert_train_only(benchmark: str, database: str) -> None:
    """校验某库是否允许用于合成（违规抛 DatasetSplitError）。"""
    eval_dbs = EVAL_DBS.get(benchmark, set())
    if database in eval_dbs:
        raise DatasetSplitError(
            f"数据泄露防护：库 '{database}' 属于 {benchmark} 的评估集（dev/test），"
            f"不能用于合成训练数据。请改用 train 划分的库。\n"
            f"  spider train 库示例: activity_1, academic, baseball_1, ...\n"
            f"  bird train 库示例: address, airline, app_store, ..."
        )


def is_eval_db(benchmark: str, database: str) -> bool:
    return database in EVAL_DBS.get(benchmark, set())


def list_train_databases(benchmark: str, db_root: str) -> list[str]:
    """扫描 {db_root}/{benchmark}/databases/，返回排除评估库后的库名（字母序）。"""
    eval_dbs = EVAL_DBS.get(benchmark, set())
    dbs_dir = Path(db_root) / benchmark / "databases"
    if not dbs_dir.exists():
        return []
    train_dbs = []
    for db_dir in dbs_dir.iterdir():
        if not db_dir.is_dir():
            continue
        db_name = db_dir.name
        if db_name in eval_dbs:
            continue
        if (db_dir / f"{db_name}.sqlite").exists():
            train_dbs.append(db_name)
    return sorted(train_dbs)
