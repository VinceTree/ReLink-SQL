# 当前代码与复现方式

本目录保留当前全量实验入口、ReLink-SQL 核心方法、论文基线适配、评测与计费代码，以及它们实际引用的共享模块。原方法文件按原字节复制，历史清单见 `source_manifest.json`。公开包的实际文件清单和校验值见上一级 `upstream_manifest.json`。早期实验的题目与 schema 快照由各自基准的官方数据源提供。旧编号目录中仍有共享依赖，因此保留原有布局。

## 先核对交接包结果

在交接包根目录执行 `python3 verify_snapshot.py`。这一步只需要 Python 标准库，会核对文件摘要、每题调用账本、初答一致性、接受后停止，以及汇总的正确数和 token；不会调用模型。

快照已保存 SQL、接受记录和计费，但不包含完整数据库执行结果及参考答案。该核对证明汇总与已保存的轨迹一致，不代替在数据库上重新评判 SQL。原始完整证据仍保留在原项目中。

## 环境

当前推理使用 macOS、Apple Silicon、Python 3.11 和 MLX。未提供经过验证的 CUDA/Windows 推理适配。依赖版本记录于 `requirements.txt`，完整环境记录见 `environment.lock.txt`。

```sh
python3.11 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r requirements.txt
```

## 数据与模型

交接包不重复装入数 GB 的数据库、模型权重和虚拟环境。资源路径、模型仓库及 revision 见 `resources.json`。

如果仍在原机器上，执行下面的命令，把已有只读资源连接到本代码目录。它只在交接包内建立链接，不连接任何实验输出目录，不改变原项目正在运行的实验：

```sh
experiments/.venv/bin/python prepare_assets.py --from-project /实际路径/pipeline
```

在新机器上，请按 `resources.json` 的相对目录放置相同版本的数据和 SHARE 的三个本地量化模型。两个 Qwen 模型由 `huggingface_hub.snapshot_download(repo_id, revision=...)` 下载到该机器的 Hugging Face 缓存；repo_id 和 revision 使用清单中的精确值。SHARE 应使用清单中的作者 checkpoint 及相同的 4 位、group_size=64、affine 转换配置，并保留 `share_provenance.json`。转换所得配置必须符合记录；更换模型或后端需要作为新配置报告。

随后运行 `experiments/.venv/bin/python prepare_assets.py` 检查资源。资源不齐时检查会返回失败，不启动推理。

## 当前实验的完整运行顺序

在本 `code` 目录执行。下列命令建立新的输出，不会直接接续压缩包中的结果快照。首次生成及全量比较会实际消耗本地推理资源。

```sh
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.dataset prepare
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.dataset references
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.initial
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m unittest experiments.iteration17.test_protocol -v
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.run smoke
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.run full
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m experiments.iteration17.analyze
```

中断后使用相同的 `run full` 命令续跑；程序校验已保存阶段，复用已完成调用。不要编辑已有运行的受保护输入或源码，也不要手工伪造完成标记。更改方法时另建新的代码目录和实验记录。

轻量代码检查：

```sh
PYTHONDONTWRITEBYTECODE=1 experiments/.venv/bin/python -m pytest -q experiments/iteration15/test_contract.py experiments/iteration15/test_repair.py experiments/iteration15/test_compile_repair.py experiments/iteration15/test_multiplicity.py experiments/iteration17/test_protocol.py
```

## 主要文件

| 文件 | 用途 |
|---|---|
| iteration17/run.py | 全量流程、断点恢复、逐阶段交付及计费检查 |
| iteration17/dataset.py、initial.py | 公开题目准备、独立参考评测、共同初答 |
| iteration15/contract.py | 独立提取问题要求、绑定表列、选择历史 SQL、生成输入 |
| iteration15/compile_repair.py、multiplicity_safe.py | 有边界的局部修改及解析异常处理 |
| iteration15/repair.py | 完整结果标识及失败历史表示 |
| iteration17/methods.py 及共享 methods.py | 论文方法的 SQL 适配 |
| iteration17/analyze.py | 总体正确率、全过程 token、相近质量回放和统计区间 |

上述文件均位于 `experiments/` 下。其余共享模块由这些入口引用；保留它们用于运行依赖，不代表将旧研究方案并入当前模型。

本包不包含原项目的全部历史运行缓存；重新生成时可能出现后端数值差异，不能承诺与历史批处理逐字相同。保留每次真实调用及计费，比较时注明环境与配置。
