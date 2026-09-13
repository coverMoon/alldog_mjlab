# HIMLoco 任务迁移原型归档

2026-09-13 整理正式 tasks 目录时，将之前的 Black HIM 任务原型移至此处。它是迁移参考代码，不是可直接运行的任务包；不参与 `mjlab.tasks` entry point 的包扫描与任务注册。

- `tasks/velocity/black/him/`：原 `src/nwwolf_mjlab/tasks/velocity/black/him/` 的 9 个 Python 源文件，逐文件保留原内容。
- `tasks/velocity/black/registration_before_cleanup.py`：整理前的 Black 注册入口，供核对原型任务关系。
- `task_source_sha256.json`：归档时源文件校验值。

相关算法原型仍在 `src/nwwolf_mjlab/algorithms/him/`，本次不调整算法层。正式任务导入不再加载该算法。原型中的相对导入保留原样，移出包后不可直接导入运行；后续应按整体规划整理成 `algorithms/himloco` 和显式观测分组，而非直接恢复旧入口。

已取消注册 `black-him-flat`、`black-him-rough`。未来正式 HIM 任务应使用规划中的 `black-flat-him`、`black-rough-him` 命名。当前正式入口为标准 PPO 的 `black-flat`。
