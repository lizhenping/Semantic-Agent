"""适配器层（app/adapters）：ports 的具体实现。

- llm_openai.py          ChatGateway（vLLM/OpenAI 兼容）+ Fake
- database_sqlalchemy.py DatabaseGateway（SQLite/MySQL）
- sqlglot_parser.py      SqlParser（sqlglot）
- storage_jsonl.py       KnowledgeStore / RunRecordStore（JSONL，含内存实现）
- prompt_jinja.py        PromptRenderer（Jinja2 + 模板版本指纹）
- scenario_catalog.py    ScenarioCatalog（configs/*.yaml）
"""
