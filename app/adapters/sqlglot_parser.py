"""SQL 解析适配器（app/adapters/sqlglot_parser.py）

移植自旧 infra/sql_ast.py 的 SqlglotParser，并修复其在 sqlglot 30.x
下的三处静默失效（旧实现在该版本下聚合/JOIN/列提取长期返回空，
详见 doc/03-physical-model.md 验收记录）：

1. 聚合节点：30.x 用具体类（Avg/Sum/Count/Min/Max），AggFunc 是基类，
   按类名匹配永远落空 → 改用 isinstance(exp.AggFunc)。
2. JOIN 条件：30.x 的 Join.on 是构建器方法而非属性，旧代码把函数对象
   当条件解析 → 改读 join.args.get("on")。
3. SELECT 输出别名误杀：旧逻辑把 named_selects 全部当别名，导致
   `SELECT County ...` 中裸列 County 被跳过 → 只对 Alias 表达式登记别名；
   无表前缀的裸列在"查询仅涉及一张表"时推断归属（多表歧义时留空，
   由验证层的 any-table 回退处理）。

风险修复 5：parse() 失败抛 SqlParseError，绝不静默返回空结果。
"""

import logging
from typing import Optional

from app.contracts.parsing import (
    AggregateCall,
    JoinClause,
    ParsedSql,
    PredicateClause,
)
from app.ports.parsing import SqlParseError

logger = logging.getLogger(__name__)

_AGG_FUNCS = {"SUM", "AVG", "COUNT", "MIN", "MAX"}


class SqlglotParser:
    """sqlglot 实现（纯 Python 轻量解析器）。"""

    def __init__(self, dialect: str = "sqlite"):
        self.dialect = dialect

    # ---------- 协议入口 ----------

    def parse(self, sql: str) -> ParsedSql:
        try:
            import sqlglot
            ast = sqlglot.parse_one(sql, dialect=self.dialect)
        except SqlParseError:
            raise
        except Exception as e:
            raise SqlParseError(sql, str(e)) from e
        return ParsedSql(
            sql=sql,
            dialect=self.dialect,
            tables=self._extract_tables(ast),
            columns=self._extract_columns(ast),
            joins=self._extract_joins(ast),
            aggregates=self._extract_aggregates(ast),
            predicates=self._extract_predicates(ast),
        )

    # ---------- 逐项提取 ----------

    def _extract_aggregates(self, ast) -> list[AggregateCall]:
        import sqlglot.expressions as exp

        aggregates: list[AggregateCall] = []
        aliases = self._alias_map(ast)
        tables_in_query = [t for t in self._extract_tables(ast)]
        # 30.x 聚合节点是具体类（Avg/Sum/...），必须 isinstance 基类过滤，
        # 按类名 "AggFunc" 匹配永远为空（旧实现的静默失效点）
        for node in self._walk_nodes(ast):
            if not isinstance(node, exp.AggFunc):
                continue
            func_name = (str(node.key).upper() if hasattr(node, "key")
                         else node.sql_name().upper())
            if func_name not in _AGG_FUNCS:
                continue
            arg = node.this if hasattr(node, "this") else None
            if arg is not None:
                table, column = self._extract_table_column(arg)
                table = aliases.get(table, table)
                # 无表前缀的聚合参数：单表查询时推断归属（Fig.1 AVG(CDSCode)）
                if not table and len(tables_in_query) == 1:
                    table = tables_in_query[0]
                aggregates.append(AggregateCall(
                    func=func_name,
                    table=table,
                    column=column,
                    clause=self._find_containing_clause(ast, node),
                ))
        return aggregates

    def _extract_joins(self, ast) -> list[JoinClause]:
        joins: list[JoinClause] = []
        aliases = self._alias_map(ast)
        for join_node in self._nodes_by_classname(ast, "Join"):
            right_table = ""
            if hasattr(join_node, "this") and join_node.this is not None:
                right_table = getattr(join_node.this, "name", "") or ""
            right_table = aliases.get(right_table, right_table)

            # 30.x：Join.on 是方法，条件在 args["on"]
            on_condition = join_node.args.get("on") if isinstance(join_node.args, dict) else None

            if on_condition is not None:
                left_col, right_col = self._extract_join_columns(on_condition)
                left_table = self._find_table_for_column(ast, left_col, right_table)
                left_table = aliases.get(left_table, left_table)
                if left_col and right_col:
                    joins.append(JoinClause(
                        left_table=left_table,
                        right_table=right_table,
                        left_column=left_col,
                        right_column=right_col,
                    ))
        return joins

    def _extract_predicates(self, ast) -> list[PredicateClause]:
        """提取 WHERE 中的等值/IN 谓词（值域成员检查用）。

        只收 col = literal 与 col IN (literal, ...) 两种形态；
        字符串字面值去引号，数值/日期保留原文本并标 is_string=False。
        """
        import sqlglot.expressions as exp

        predicates: list[PredicateClause] = []
        aliases = self._alias_map(ast)
        tables_in_query = self._extract_tables(ast)

        def _resolve_table(col_node) -> str:
            raw_table = (getattr(col_node, "table", "") or "").strip()
            table = aliases.get(raw_table, raw_table)
            if not table and len(tables_in_query) == 1:
                table = tables_in_query[0]
            return table

        def _literal(node) -> Optional[PredicateClause]:
            if not isinstance(node, exp.Literal):
                return None
            text = str(node.this) if node.this is not None else ""
            if not text:
                return None
            return PredicateClause(
                table="", column="", op="=",
                literal=text,
                is_string=bool(getattr(node, "is_string", True)))

        def _emit(column: str, table: str, op: str, lit: PredicateClause):
            predicates.append(PredicateClause(
                table=table, column=column, op=op,
                literal=lit.literal, is_string=lit.is_string))

        for node in self._nodes_by_classname(ast, "Where"):
            for cond in self._walk_nodes(node):
                if isinstance(cond, exp.EQ):
                    left, right = cond.this, cond.expression
                    if isinstance(left, exp.Column) and right is not None:
                        lit = _literal(right)
                        if lit is not None and left.name:
                            _emit(left.name, _resolve_table(left), "=", lit)
                    elif isinstance(right, exp.Column) and left is not None:
                        lit = _literal(left)
                        if lit is not None and right.name:
                            _emit(right.name, _resolve_table(right), "=", lit)
                elif isinstance(cond, exp.In):
                    target = cond.this
                    if isinstance(target, exp.Column) and target.name:
                        for item in cond.expressions or []:
                            lit = _literal(item)
                            if lit is not None:
                                _emit(target.name, _resolve_table(target),
                                      "IN", lit)
        return predicates

    def _extract_columns(self, ast) -> list[tuple[str, str]]:
        """提取引用列 [(table, column)]；别名已解析、CTE/派生表输出列
        与 SELECT 别名引用已剔除；单表查询的裸列归属该表。"""
        aliases = self._alias_map(ast)
        derived_columns = self._derived_relation_columns(ast)
        output_aliases = self._select_output_aliases(ast)
        tables_in_query = self._extract_tables(ast)

        columns: list[tuple[str, str]] = []
        seen: set = set()
        for col_node in self._nodes_by_classname(ast, "Column"):
            raw_table = (getattr(col_node, "table", "") or "").strip()
            column = (col_node.name or "").strip()
            if not column or column == "*":
                continue
            # 无表前缀且命中 SELECT 输出别名（AS 名）→ 别名引用，跳过
            if not raw_table and column in output_aliases:
                continue
            table = aliases.get(raw_table, raw_table)
            if (table.lower() in derived_columns
                    and column.lower() in derived_columns[table.lower()]):
                continue
            # 单表查询的裸列归属该表；多表歧义留空由验证层回退
            if not table and len(tables_in_query) == 1:
                table = tables_in_query[0]
            key = (table, column)
            if key not in seen:
                seen.add(key)
                columns.append(key)
        return columns

    def _extract_tables(self, ast) -> list[str]:
        derived_names = set(self._derived_relation_columns(ast))
        aliases = self._alias_map(ast)
        tables: list[str] = []
        seen: set = set()
        for table_node in self._nodes_by_classname(ast, "Table"):
            name = (table_node.name or "").strip()
            if not name or name.lower() in derived_names:
                continue
            real_name = aliases.get(name, name)
            if real_name.lower() in derived_names or real_name.lower() in seen:
                continue
            seen.add(real_name.lower())
            tables.append(real_name)
        return tables

    # ---------- AST 辅助 ----------

    @staticmethod
    def _select_output_aliases(ast) -> set[str]:
        """SELECT 输出别名集合：仅 Alias 表达式（如 SUM(x) AS total）。
        裸列 County 的输出名不进集合——它同时是真实 schema 列。"""
        aliases: set[str] = set()
        try:
            import sqlglot.expressions as exp
            for e in getattr(ast, "expressions", None) or []:
                if isinstance(e, exp.Alias) and e.alias:
                    aliases.add(e.alias)
        except Exception:
            pass
        return aliases

    def _alias_map(self, ast) -> dict:
        """{alias: real_table}：FROM SalesOrderHeader AS soh -> {"soh": ...}"""
        mapping = {}
        for t in self._nodes_by_classname(ast, "Table"):
            name = (getattr(t, "name", "") or "").strip()
            alias = (getattr(t, "alias", "") or "").strip()
            if name and alias:
                mapping[alias] = name
        return mapping

    @staticmethod
    def _nodes_by_classname(ast, class_name: str):
        """按类名遍历 AST 节点（兼容 sqlglot 各版本 walk()）。"""
        for node in ast.walk():
            n = node[0] if isinstance(node, tuple) else node
            if n.__class__.__name__ == class_name:
                yield n

    @staticmethod
    def _walk_nodes(ast):
        """遍历全部 AST 节点（不按类名过滤）。"""
        for node in ast.walk():
            yield node[0] if isinstance(node, tuple) else node

    def _derived_relation_columns(self, ast) -> dict[str, set[str]]:
        """CTE 与派生表的输出列名（小写）。"""
        derived: dict[str, set[str]] = {}
        for node in self._nodes_by_classname(ast, "CTE"):
            name = (getattr(node, "alias_or_name", "") or "").strip()
            query = getattr(node, "this", None)
            names = getattr(query, "named_selects", None) or []
            if name:
                derived[name.lower()] = {
                    str(value).strip().lower() for value in names if value
                }
        for node in self._nodes_by_classname(ast, "DerivedTable"):
            if node.__class__.__name__ == "CTE":
                continue
            name = (getattr(node, "alias_or_name", "") or "").strip()
            query = getattr(node, "this", None)
            names = getattr(query, "named_selects", None) or []
            if name:
                derived[name.lower()] = {
                    str(value).strip().lower() for value in names if value
                }
        return derived

    @staticmethod
    def _extract_table_column(node) -> tuple[str, str]:
        if node is None:
            return ("", "")
        if hasattr(node, "table") and hasattr(node, "name"):
            return (node.table or "", node.name or "")
        if hasattr(node, "name"):
            return ("", node.name or "")
        return ("", str(node))

    @staticmethod
    def _find_containing_clause(ast, node) -> str:
        try:
            parent = node.parent
            while parent is not None:
                cls_name = parent.__class__.__name__
                if cls_name == "Select":
                    return "SELECT"
                if cls_name == "Where":
                    return "WHERE"
                if cls_name == "Having":
                    return "HAVING"
                if cls_name == "Order":
                    return "ORDER BY"
                parent = parent.parent
        except Exception:
            pass
        return "UNKNOWN"

    @staticmethod
    def _extract_join_columns(on_condition) -> tuple[str, str]:
        try:
            if hasattr(on_condition, "left") and hasattr(on_condition, "right"):
                left = on_condition.left
                right = on_condition.right
                left_col = left.name if hasattr(left, "name") else str(left)
                right_col = right.name if hasattr(right, "name") else str(right)
                return (left_col, right_col)
        except Exception:
            pass
        return ("", "")

    def _find_table_for_column(self, ast, column: str, exclude_table: str = "") -> str:
        try:
            for col_node in self._nodes_by_classname(ast, "Column"):
                if (col_node.name == column and col_node.table
                        and col_node.table != exclude_table):
                    return col_node.table
        except Exception:
            pass
        return ""


class FakeSqlParser:
    """假解析器（单测用）：返回预置 ParsedSql 或配置抛错。"""

    def __init__(self, parsed: Optional[ParsedSql] = None,
                 error: Optional[str] = None):
        self._parsed = parsed or ParsedSql()
        self._error = error
        self.calls: list[str] = []

    def parse(self, sql: str) -> ParsedSql:
        self.calls.append(sql)
        if self._error is not None:
            raise SqlParseError(sql, self._error)
        return self._parsed
