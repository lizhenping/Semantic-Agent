"""SemanticSQL Agent 目标工程结构（重构基线，见 doc/03-physical-model.md）。

依赖方向（唯一允许的方向）：
    CLI / API -> application -> domain -> contracts / ports
    adapters -> ports + contracts
    bootstrap -> application / domain / adapters

约束：
- contracts 不导入 LLM、SQLAlchemy 等运行框架或具体存储。
- domain 不创建客户端、不读环境变量、不写文件。
- application 负责顺序、迭代预算和持久化调用，不负责 Prompt 细节。
- adapters 不调用 application，不决定 accepted/rejected。

旧实现（D:/学术/tkde/code/semanticsql-agent）仅作为迁移参考与历史产物
存放地；本目录内的新代码不得 import 旧目录模块。
"""
