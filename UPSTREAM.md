# CoreCoder 派生来源

- 上游：https://github.com/he-yufeng/CoreCoder
- 复制时间：2026-10-09。
- 本地基线 HEAD：`c2d841719019a96346c42173ed96dbe71c5cdeda`，本地版本 0.7.0。
- 基于 CoreCoder 的独立副本派生，包含原有学习注释，不依赖外部上游目录。
- 排除 `.git`、`.venv`、`.env`、缓存和构建输出，独立安装依赖。
- 保留 `LICENSE`、`corecoder/`、原示例与测试。上游介绍见原仓库；本地学习文档不上传。
- 新增 `skillflow/` 为业务层，实际继承 `corecoder.tools.base.Tool`，加入 LangChain 工具适配；业务编排由 LangGraph 完成，未声称复用 CoreCoder 原有 coding-agent 主循环。
- Pyproject 包名改为 `skillflow-agent`，原 corecoder CLI 保留用于阅读与实验。它不属于聊天 API 的注册工具。

禁止将保留的上游测试数量、README 指标或作者成果算作本项目新增成果。

上游版本：`0.7.0`。以下统计只描述本仓库保留的 CoreCoder 副本，不代表 SkillFlow Agent 的代码量：
`engine-1327_LoC`；25 files；2,851 physical lines；2,231 net。
