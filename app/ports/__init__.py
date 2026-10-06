"""端口层（app/ports）：domain/application 依赖的抽象接口。

实现见 app/adapters。domain 不得 import adapters；
application 只通过 ports 组装依赖。
"""
