# vLLM 学习与贡献路线图（个人学习计划）

> 这是一份个人学习计划，**不是 vLLM 项目文档**，不应被合并到 upstream。
> 它记录了从零开始学习 vLLM、做出第一个贡献，并最终转入 LLM 推理领域工作的完整路径。

---

## 0. 总体心法

vLLM 处于 **LLM 推理 / 系统 / GPU kernel / 编译器 / 分布式** 的交叉点。
学习它需要同时积累 4 类能力：

1. **LLM 推理基本原理**：KV cache、attention、采样、speculative decoding、PD 分离…
2. **系统/调度能力**：continuous batching、prefix caching、请求调度、内存管理
3. **GPU/算子知识**：CUDA、Triton、torch.compile、CUDA Graph、PagedAttention
4. **工程能力**：大型 Python 项目结构、配置系统、多进程、CI、性能分析

**核心原则**：不要试图"读完代码"。代码 ~50 万行，盲读会爆炸。
正确做法是 **「以问题驱动 + 以可运行的最小例子驱动 + 以一次次可提交的小贡献驱动」**。

---

## 1. 先决知识（1～2 周）

| 主题 | 推荐资源 | 你要能解释 |
|---|---|---|
| Transformer 推理 | nanoGPT、HuggingFace 教程 | prefill vs decode、KV cache |
| Attention | "Attention Is All You Need"、FlashAttention 论文 | 为什么 attention 是显存瓶颈 |
| PagedAttention | vLLM SOSP'23 论文 | 为什么需要"分页" |
| Continuous batching | Orca 论文 (OSDI'22) | 与 static batching 的区别 |
| CUDA 基础 | CUDA C++ Programming Guide 前 4 章 | warp、SM、shared memory |
| PyTorch 基础 | torch.compile, custom op | `torch.library` 注册一个 op |
| 多进程 / asyncio | Python 官方文档 | 进程间通信的几种方式 |

**必读论文**：
- *Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP'23)
- *Orca: A Distributed Serving System for Transformer-Based Generative Models* (OSDI'22)
- *DistServe* / *SGLang* (对比阅读)

---

## 2. 阶段一：跑起来 + 黑盒使用（第 1 周）

**目标**：能在自己的机器上把 vLLM 用起来，理解它对外暴露的 API。

### 任务

- [ ] 按 `AGENTS.md` 配置环境：
  ```bash
  uv venv --python 3.12
  source .venv/bin/activate
  uv pip install -r requirements/lint.txt
  pre-commit install
  VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
  ```
- [ ] 跑通 4 个 demo：
  - Offline inference: `examples/offline_inference/basic.py`
  - Online server: `vllm serve facebook/opt-125m`
  - Chat 模板：跑一个 instruct 模型
  - LoRA 加载：试一个带 LoRA 的例子
- [ ] 读这些文档（一遍即可，不用记细节）：
  - `docs/design/arch_overview.md` ← 最重要的入口
  - `docs/design/paged_attention.md`
  - `docs/design/prefix_caching.md`
  - `docs/contributing/README.md`

### 检验问题

- vLLM 暴露了哪几个入口？（`LLM` 类、`vllm serve` CLI、`api_server`）
- 一个 prompt 进入 vLLM 后大致经过哪几层？
- 什么是 prefill / decode / continuous batching？

---

## 3. 阶段二：建立架构地图（第 2～3 周）

**目标**：能在脑子里画出 vLLM 架构图，知道每个目录干什么。

> ⚠️ 重点学习 **V1 架构**（`vllm/v1/`），新代码都在这里。老 engine 知道存在即可。

### 目录地图

| 目录 | 作用 | 优先级 |
|---|---|---|
| `vllm/entrypoints/` | LLM 类、OpenAI server、CLI | ★★★ |
| `vllm/v1/engine/` | V1 引擎核心 | ★★★ |
| `vllm/v1/core/` | 调度器、KV cache 管理 | ★★★ |
| `vllm/v1/worker/` | Worker 进程，跑模型 forward | ★★★ |
| `vllm/v1/attention/` | V1 attention backends | ★★ |
| `vllm/v1/sample/` | 采样、logits 处理 | ★★ |
| `vllm/v1/spec_decode/` | Speculative decoding | ★ |
| `vllm/v1/kv_cache_interface.py` | KV cache 抽象 | ★★ |
| `vllm/model_executor/` | 模型实现 + layers | ★★★ |
| `vllm/attention/` | 老 attention 后端 | ★★ |
| `vllm/distributed/` | TP、PP、EP 通信 | ★★ |
| `vllm/compilation/` | torch.compile / CUDA Graph | ★★ |
| `vllm/lora/` | LoRA | ★ |
| `vllm/multimodal/` | 多模态 (VLM) | ★ |
| `vllm/platforms/` | 硬件抽象 | ★ |
| `csrc/` | C++/CUDA kernels | ★★ |
| `vllm/config/` | 配置系统 | ★★ |
| `tests/` | **每个子系统的最佳"使用手册"** | ★★★ |

### 必读 design docs（按顺序）

1. `docs/design/arch_overview.md` ★★★
2. `docs/design/paged_attention.md` ★★★
3. `docs/design/prefix_caching.md`
4. `docs/design/torch_compile.md`
5. `docs/design/cuda_graphs.md`
6. `docs/design/attention_backends.md`
7. `docs/design/model_runner_v2.md`
8. `docs/design/multiprocessing.md`
9. `docs/design/optimization_levels.md`
10. `docs/design/hybrid_kv_cache_manager.md`
11. `docs/design/huggingface_integration.md`
12. `docs/design/fused_moe_modular_kernel.md`、`docs/design/moe_kernel_features.md`

### 关键学习方法：追一个请求的生命周期

```
client request
  → vllm/entrypoints/openai/api_server.py
  → vllm/entrypoints/openai/serving_chat.py / serving_completion.py
  → vllm/v1/engine/async_llm.py (AsyncLLM)
  → vllm/v1/engine/processor.py (tokenize + 预处理)
  → vllm/v1/engine/core.py (EngineCore，调度循环)
  → vllm/v1/core/sched/scheduler.py (Scheduler)
  → vllm/v1/core/kv_cache_manager.py (分配 KV blocks)
  → vllm/v1/executor/* (Executor)
  → vllm/v1/worker/gpu_worker.py + gpu_model_runner.py
  → vllm/model_executor/models/<某个模型>.py (forward)
  → vllm/v1/attention/backends/* (attention kernel)
  → vllm/v1/sample/ (采样)
  → 结果回流，detokenize
```

打通这条线后，剩下的学习会快得多。

### 检验问题

- 当 batch 里同时有 prefill 和 decode 请求时，scheduler 怎么处理？
- KV cache block 是什么粒度？free / allocate 流程？
- prefix caching 在哪个文件实现？命中时跳过了什么计算？
- torch.compile 在 vLLM 里默认开吗？编译什么？什么时候触发？
- TP=2 时，哪些层会被切？通信发生在哪里？

---

## 4. 阶段三：选一个主攻方向深耕（第 4～8 周）

从下面 6 个方向里选 **1 个主、1 个辅**：

### 方向 A：调度与内存管理（System / Serving）

- **关键文件**：`vllm/v1/core/sched/`、`vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`
- **关键概念**：continuous batching、preemption、swap、prefix caching、chunked prefill、PD disaggregation
- **适合**：对操作系统/数据库/调度感兴趣的人。**就业面非常广**。

### 方向 B：Attention / Kernel

- **关键文件**：`csrc/attention/`、`vllm/v1/attention/backends/`、`vllm/attention/ops/`
- **关键概念**：FlashAttention、PagedAttention、FlashInfer、Triton kernel
- **适合**：CUDA / 数值计算 / HPC 背景。门槛高，回报也高。

### 方向 C：模型支持（Models / HuggingFace 集成）

- **关键文件**：`vllm/model_executor/models/`、`vllm/transformers_utils/`、`vllm/multimodal/`
- **入门最容易**：找一个未支持的模型移植进 vLLM
- **适合**：新手最容易上手的方向。

### 方向 D：量化（Quantization）

- **关键文件**：`vllm/model_executor/layers/quantization/`、`csrc/quantization/`
- **关键概念**：FP8、INT8、AWQ、GPTQ、Marlin、BitsAndBytes
- **适合**：业界非常缺人。

### 方向 E：分布式 / 多机推理

- **关键文件**：`vllm/distributed/`、`vllm/v1/executor/multiproc_executor.py`、`vllm/v1/executor/ray_executor.py`
- **关键概念**：TP、PP、EP、SP、NCCL、Ray、PD 分离、KV transfer

### 方向 F：torch.compile / CUDA Graph / 编译优化

- **关键文件**：`vllm/compilation/`
- **适合**：和 PyTorch 团队、Meta、NVIDIA 接得上。

> **建议**：如果目标是"尽快进入这个行业找到工作"，
> 选 **C（模型支持）** 作为入门方向，**A（调度）** 作为长期主攻。

---

## 5. 阶段四：第一次贡献（第 6～10 周）

**目标**：合并第一个 PR。

### 找任务

- [Good first issues](https://github.com/vllm-project/vllm/issues?q=is%3Aissue+state%3Aopen+label%3A%22good+first+issue%22)
- [Selected onboarding tasks](https://github.com/orgs/vllm-project/projects/6)
- [New model requests](https://github.com/vllm-project/vllm/issues?q=is%3Aissue+state%3Aopen+label%3A%22new-model%22)

### 难度阶梯

1. **文档/示例补充**（不要做 typo-only 的"水 PR"）
2. **写一个测试**
3. **修一个 bug**
4. **新增一个小 feature 或 model**
5. **重大 feature**（>500 LOC 必须先发 RFC）

### 必须遵守的硬规则（来自 `AGENTS.md`）

- ✅ PR 标题前缀：`[Bugfix]`、`[Model]`、`[Kernel]`、`[Core]`、`[Frontend]`、`[Doc]`、`[CI/Build]`、`[Hardware][Vendor]`、`[Misc]`
- ✅ `git commit -s`（DCO 签名）
- ✅ AI 辅助必须在 PR 描述中声明，commit trailer 加 `Co-authored-by: Claude`
- ✅ 不开纯 agent PR — 你必须真懂、能 debug、能解释每一行
- ✅ 开 PR 前用 `gh pr list` 搜重复
- ❌ 不开 typo / 单行 cleanup 的"水 PR"

### 本地必跑检查

```bash
pre-commit run --all-files
pre-commit run mypy-3.10 --all-files --hook-stage manual
.venv/bin/python -m pytest tests/<相关测试> -v
```

---

## 6. 阶段五：建立专业声誉（第 3～6 个月）

- [ ] 在主攻方向累积 5～10 个有意义的 PR
- [ ] 主动 review 别人的 PR
- [ ] 参加 vLLM Office Hours / Slack
- [ ] 在 GitHub 上回答用户问题
- [ ] 写一篇博客（"我是怎么给 vLLM 加 X 模型的" / "vLLM v1 调度器解读"）
- [ ] 关注 RFC 讨论
- [ ] 跟 MLSys、OSDI、SOSP、NSDI 上的 LLM serving 论文

---

## 7. 阶段六：求职准备（6 个月以后）

### 目标雇主

| 类型 | 代表 | 看你什么 |
|---|---|---|
| vLLM 母公司 / 衍生 | Neural Magic (Red Hat)、UC Berkeley Sky Computing | 贡献历史 |
| 推理服务公司 | Anyscale、Together、Fireworks、Modal、Lepton、Replicate | 系统能力 + 部署经验 |
| 大模型公司 inference | OpenAI、Anthropic、Mistral、Cohere、Deepseek | 模型 + 系统 |
| GPU / 编译器 | NVIDIA、AMD、Google、Intel | kernel + 编译能力 |

### 简历核心 3 件套

1. GitHub 上合并的 PR（最好按方向集中）
2. 一篇高质量技术博客
3. 一个 side project（例如：实现一个迷你版 PagedAttention）

---

## 8. 时间线总结

| 时间 | 阶段 | 里程碑 |
|---|---|---|
| 第 1～2 周 | 先决知识 + 跑通 demo | 能用 vLLM 起服务 |
| 第 3～4 周 | 架构地图 + 追请求生命周期 | 能画出数据流图 |
| 第 5～8 周 | 选定方向深耕 + 第 1 个 PR | **第一个 PR 合并** |
| 第 2～3 个月 | 持续贡献 + 写测试 + 修 bug | 3~5 个 PR 合并 |
| 第 3～6 个月 | 主攻方向积累 + 参与 review | 在某子系统"混脸熟" |
| 6 个月+ | 写博客 + 投简历 | 拿到面试 |

---

## 9. 关键建议

1. 不要只读代码，要"改 + 跑 + 测"
2. 善用 `tests/` 目录，比 docs 更新更及时
3. 关注 v1，不要在老 engine 上浪费时间
4. 从 issue 倒推学习路径，比"系统性读代码"高效 10 倍
5. 不要怕问问题，但先 grep + read 做功课
6. 承担 review 责任前不要贸然交 PR
7. 每天 git pull，养成读 changelog 的习惯
8. 盯着 release notes — 那是面试官会问的方向

---

## 10. 第一周立刻可做的 5 件事

- [ ] 通读 `AGENTS.md`、`docs/design/arch_overview.md`、`docs/design/paged_attention.md`
- [ ] 装环境，用 `facebook/opt-125m` 跑通 offline + online
- [ ] 在 IDE 里给 `vllm/v1/engine/async_llm.py` 和 `vllm/v1/engine/core.py` 设断点跟请求
- [ ] 读 PagedAttention 论文
- [ ] 把 `good first issue` 翻一遍（不做，先理解每个 issue 在说什么）

---

## 学习日志

> 在这里记录每周的进度、卡住的地方、收获。

### Week 1
- [ ] ...

### Week 2
- [ ] ...
