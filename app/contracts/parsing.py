"""SQL AST 解析结果契约（app/contracts/parsing.py）

解析器输出的结构事实（供确定性校验用）。
论文 Fig.1 反例 AVG(CDSCode) 的检测链路：
    parse(sql) -> aggregates=[AVG(CDSCode)] -> 查 K3 identifier -> 报错
"""

from pydantic import BaseModel, Field


class AggregateCall(BaseModel):
    """聚合函数调用（如 AVG(CDSCode)）"""

    func: str           # SUM/AVG/COUNT/MIN/MAX
    table: str          # 解析别名后的真实表名
    column: str         # 列名
    clause: str = ""    # 所在子句（SELECT/WHERE/HAVING/ORDER BY）


class JoinClause(BaseModel):
    """JOIN 子句的 ON 等值条件"""

    left_table: str
    right_table: str
    left_column: str
    right_column: str


class PredicateClause(BaseModel):
    """WHERE 等值/IN 谓词（值域成员检查用，论文消融 C）"""

    table: str              # 解析别名后的真实表名（多表歧义留空则跳过检查）
    column: str
    op: str = "="           # "=" / "IN"
    literal: str = ""       # 字面值原文（字符串已去引号，数值为原文本）
    is_string: bool = True  # True=字符串字面值，False=数值/日期字面值


class ParsedSql(BaseModel):
    """一次成功解析的结构摘要。"""

    sql: str = ""
    dialect: str = "sqlite"
    tables: list[str] = Field(default_factory=list)
    columns: list[tuple[str, str]] = Field(default_factory=list)  # [(table, column)]
    joins: list[JoinClause] = Field(default_factory=list)
    aggregates: list[AggregateCall] = Field(default_factory=list)
    predicates: list[PredicateClause] = Field(default_factory=list)
