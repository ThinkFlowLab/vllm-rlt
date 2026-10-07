# M2 Engine / EngineCore：R1 重构计划（Scheduler–EngineCore 边界）

> - **状态**：R1 已集成到最新上游，CPU 回归与两种 KV 布局的基线对照通过；当前版本的正式 GPU、性能、质量与多卡 PD 验收待执行。
> - **关联**：[RFC #32](https://github.com/ThinkFlowLab/vllm-rlt/issues/32) 的 R1 交付物。
> - **当前交付基线**：`upstream/main @ ecb1f8b505b7e831815b40aec3b4598619cca23a`（2026-10-07）。下方迁移计划中的旧行号与历史实验仍基于 `d286233`，不能当作当前版本的验证。
> - **旧 CPU 回归基线**（`d286233`，未加 `--run-gpu`）：`tests/test_engine.py tests/test_async_pipeline.py tests/test_async_state.py tests/test_cdb_runtime.py tests/test_prefix_growth.py tests/test_pd.py tests/test_speculative.py tests/test_serving.py` 结果为 **192 passed, 57 skipped**。
> - **当前 CPU 验证**（2026-10-07）：在隔离的 Python 3.10 评测环境中，`python -m pytest -q -ra` 为 **465 passed, 115 skipped**；相同环境的上游基线为 **442 passed, 115 skipped**。115 个跳过项包括 112 个 GPU 用例与 3 个未缓存的官方 Nanbeige 检查。

## 当前集成与验证（2026-10-07）

- 保留最新上游的 profiling 包装、`close()`、Sampler 接口与已合入 #48 的同步 speculative CUDA Graph 支持。两条执行 driver 继续共用类型化结果应用入口。
- 上游测试使用 `tests.helpers`；补充测试包入口，避免已安装的同名 `tests` 包覆盖仓库测试。基线 CPU 验证使用相同的测试包入口，未改动其运行时代码。
- 全仓 pre-commit 检查发现上游 speculative GPU readback 用例仍调用已删除的 `OuroConfig.tiny`；改为现有 `tiny_ouro_config`，保留原配置与断言。CPU 跳过该 GPU 用例，不能由 CPU 通过推断它已经执行。
- CPU tiny-model 对照覆盖 LAST_EXITED / SHARED × refill / no-refill × sync eager / sync padded / async single / async multi，共 16 个组合。tokens、exit depths、stage/request 调度序列与 KV pool 大小全部匹配当前上游基线。
- 固定官方 revision 的 Ouro-1.4B BF16 CPU 短输出对照通过：参数与 KV 为 BF16，tokens、exit depths `[4, 3]`（首 token 的满深 prefill 与后一 token 的自适应退出）、结束原因、调度序列与请求 KV 回收匹配。此检查仅覆盖 CPU Torch attention 的一条两输出请求，不代替 GPU 验收或任务准确率。
- 新增 `benchmarks/r1_runtime.py`，记录运行时代码哈希、控制参数、原始输出、阶段序列、吞吐、TTFT、ITL/TPOT、host 时间与 GPU 峰值内存。profiling 使用独立诊断轮，不进入速度数据。
- 当前 GPU 协议在运行前冻结：BF16、单 CPU 线程和固定亲和性、sync / async / graph × B1 / B2、每请求 32 输出、每进程 3 次预热与 5 次测量、ABBA 顺序。每对进程中位吞吐至少保留基线的 95%，TTFT/TPOT 至多增加 5%；原始范围同时报告。这是限定负载的退化筛查，不宣称加速。
- 当前版本尚未执行该 GPU 协议，也未执行配对 GSM8K10 smoke 或 2/4 卡 PD。下面的 2026-09-26 GPU/性能结果只属于旧基线。当前宿主机仅有一张 A5000，且无 GPU 调度器；正式 GPU 执行需要明确可用资源。

## 旧基线实现与实验记录（2026-09-26）

- `Scheduler.add_request` 分配单调递增的 generation；每个非空 `_take` 分配 seq；`ScheduledItem` 冻结 position、loop depth 和 CODA output index。复用 ID 与 preemption 的测试验证 generation 语义。
- `worker/output.py` 定义 `Progress`、`ExitSignal` 和按列存放的 `ModelRunnerOutput`；`engine/output_adapter.py` 暂时包装现有的四个 runner 接缝，不增加 host 读取。
- `Scheduler.update_from_output` 是结果应用的唯一入口。它负责 PREFILL/PRELUDE、RECURRENT 退出策略、CODA 的 placeholder/EOS/长度，以及 SPECULATIVE 的 token 与 KV 截断；返回六类动作。`LLMEngine` 保留独立的 sync/async driver，只执行提交、signal readback、finalize、finish、stats、device-token shim 和输出构造。
- async delayed signal 采用第 6.1 节 Q4 的方案 P：只为下一次提交必定会消费的信号保留 handle。CPU 测试固定了 sync/async 到 max 深度时消费次数的原有差异，以及 preemption 后的恢复行为。
- 本地 CPU tiny-model CDB 对照基于 `d286233`，8 种配置的输出与调度序列完全一致。两次 ABBA 吞吐量对照的幅度和方向不稳定；针对 refill 的 sync eager / async multi 又各做了 20 次逐步 host 计时，发现 CODA/PRELUDE 边界有约 10–20 µs 的新增开销，但总时长的观测范围重叠。真实 BF16 测量见下文。
- A5000（SM86）探索性全 GPU 运行曾有 17 个 FlashAttention 用例在初始化时因 FA2 block-size/FA4 架构限制失败。后续按本机可执行范围重跑，补齐了 PD 双卡用例缺失的设备数量跳过条件；这没有改动 PD 运行逻辑。
- PD 兼容性复测：在隔离路径 `/tmp/vllm-rlt-r1-deps` 安装 `nixl==1.4.1` 和 `nixl-cu12==1.4.1` 后，`PYTHONPATH=/tmp/vllm-rlt-r1-deps:$PWD CUDA_VISIBLE_DEVICES=0 python -m pytest -q --run-gpu tests/test_pd.py` 为 **9 passed, 9 skipped**。NIXL 可导入；真实传输仍需 2/4 张 GPU，不能由单卡推断通过。
- 本机 FA2 依赖检查：A5000 支持 FA2，但当前 Torch 2.6.0+cu124 与官方 v2.8.3/v2.8.3.post1 预编译 wheel 的 `flash_attn_2_cuda` 导入遇到 `c10::Error` C++ ABI 未解析符号。只在隔离 `/tmp` 路径尝试，未改项目或全局环境；本机仅有 CUDA 11.8 `nvcc`，无法直接从源码构建要求 CUDA 12+ 的 FA2。详见 [FlashAttention 依赖说明](flash_attention.md)。

**剩余门禁**：在已确认的 GPU 分配上运行第 3 节的正式 G-GPU 全套；FA4 用例需要其支持的 GPU，PD 双卡用例需要两张已分配 GPU。M3/M5 owner 还需 review 第 6 节的本地契约选择；若集成时 #48/#50/#52/#59/#60 已先合入，需按 Q9/Q19 调整对应 stage。

## 本机验收记录（2026-09-26）

- 单卡测试命令：`CUDA_VISIBLE_DEVICES=0 python -m pytest -q --run-gpu --ignore=tests/test_flash_attention.py -k 'not flash_attn and not fa4 and not prefill_uva'`，结果 **370 passed, 20 skipped, 18 deselected**。未覆盖的重点是 FA4 和 PD 双卡；18 个 deselected 是按测试名排除的 FlashAttention/UVA 组合，`test_flash_attention.py` 整个文件另行排除。这里使用用户要求的本机验收范围，并未取得调度器分配。
- BF16 A/B 使用固定 revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` 的 `ByteDance/Ouro-1.4B`，基线 `d286233` 与当前工作树分别加载相同的本地权重。两边的参数与 KV tensor 都实测为 BF16，使用 Triton、相同 cache/scheduler 配置与两条并发请求。性能负载为每条 16-token 固定 prompt、32 个输出、`ouro_delayed`、阈值 0.5、忽略 EOS；每个进程先预热一轮，再记录两轮。计时从请求加入后到全部完成，包含 engine 调度与 GPU 执行；TTFT、ITL、TPOT 为 engine 输出时间，不含服务层。每轮都完整输出 64 tokens，token IDs、exit depths 和 step 数逐项匹配基线，结束后 KV 使用块数为 0。

| 模式 | tokens/s 基线 → 当前 | TTFT ms 基线 → 当前 | ITL 中位数 ms 基线 → 当前 | TPOT ms 基线 → 当前 | 每步 host 中位数 ms 基线 → 当前 |
|---|---:|---:|---:|---:|---:|
| sync eager | 19.47 → 17.35 | 114.0 → 116.6 | 102.3 → 114.7 | 102.9 → 115.2 | 21.40 → 27.04 |
| async eager | 17.59 → 17.82 | 135.1 → 116.5 | 111.4 → 107.8 | 113.0 → 112.2 | 25.76 → 24.69 |
| async + graphs | 61.51 → 61.09 | 120.5 → 124.7 | 29.67 → 29.76 | 29.67 → 29.78 | 5.77 → 5.98 |

这些是两轮中位数，不应当作稳定的性能差值。sync 又做了 ABBA 顺序的四轮暖态对照：不固定 CPU 亲和性时，基线 17.77–18.86 tokens/s、当前 17.19–17.94；固定到 CPU 0–3 后，基线 15.65–17.15、当前 16.05–18.79，差值方向反转。当前数据不支持确定性的吞吐退化或提升结论；没有预设的吞吐提升门槛。CPU tiny-model 中约 10–20 µs 的边界开销仍是本次重构可归因的 host 成本。

- 另用两条自然语言 prompt、阈值 0.01、每条 8 个输出对比真实自适应路径。sync、async、graph 各自的当前版与基线 token IDs 和 exit depths 完全一致，实际覆盖 3/4 层混合退出。graph 与 eager 有一处末 token 差异，但在基线和当前版都出现，非 R1 新增差异。
- `torch.cuda.set_sync_debug_mode("warn")` 的独立诊断轮中，sync 的基线/当前各 181 条警告，async 各 21 条；graph 首轮各 2 条、预热后各 0 条。graph 两边均为 2 次 capture、51 次 replay、0 次 fallback。这个 PyTorch 诊断模式并不能检测全部同步，只能支持同协议下“没有新增警告”的比较。诊断开销未计入上表。

## 范围

- **M2 负责**：engine 接口、组件装配、执行循环（sync 与 async 两个 driver）、in-flight 编排、控制请求（`add_request`、`abort_request`）、关闭和错误传播。
- **本文只定义 M2 用到的和交出去的契约**，不设计以下内容：
  - M3 的内部结构，例如 helper 和 policy 对象怎么组织；
  - M5 的 runner、buffers、async state、CUDA graphs；
  - M4 的逻辑 KV；
  - M9 的 PD。
- **明确不做**：不引入 vLLM 依赖，也不照搬它的类层级；不把 sync 和 async 合成一个循环；不新增执行模式；不扩展模型覆盖、并行规模或部署能力。

## 术语

- **提交（submission）**：host 把一个 `SchedulerOutput` 交给 runner。在 async 下，host 上的进度（`loops_done`、`num_prefilled_tokens`）表示"已提交"，不表示"已完成"。
- **GPU 完成**：本次提交的 event 已经完成。它只用作资源释放的条件，不会触发调度状态更新。
- **CPU 交付**：结果值（CODA 的 token、RECURRENT 的 score）已经能在 host 上读到。
- **handle**：EngineCore 持有的一个引用，指向某个 `Submission` 里的某一行，之后可以凭它取回这一行的值。
- **placeholder**：即 `Request.num_output_placeholders`，表示已经提交、但还没交付的 CODA 输出。每个请求最多一个。
- **generation / seq**：generation 是请求的注册代次，seq 是每个 `SchedulerOutput` 的执行序号，见 2.1。

## 核心决定

- **不把 `step()` 和 `_step_async()` 合成一个循环。**
  - 两个 driver 都保留：sync 分支抽成 `_step_sync()`，`step()` 只负责分派；async 的异常处理原样保留。
  - 两个 driver 共享一组契约：`SchedulerOutput → ModelRunnerOutput → Scheduler.update_from_output() → SchedulerUpdate`。
  - 两个 driver 也共享 EngineCore 侧同一段副作用执行：finalize、finish 顺序、构造输出。
- **结果应用只有 `update_from_output` 一个入口，但 sync 和 async 调用它的时刻不同。**
  - sync：`execute()` 返回后调用一次（`COMPLETED`）。
  - async：`submit()` 之后调用一次（`SUBMITTED`）；CODA 的 token 到达 CPU 时再调用一次（`DELIVERED`）。
  - GPU 完成不触发调度状态更新，只作为 M4/M5 的资源释放条件。
- **R1 不新建 EngineCore 类。** 文中的 "EngineCore" 指 `LLMEngine` 里负责 driver 和 in-flight 状态的那部分代码。是否拆成单独的类，等 M1 的稳定接口确定后再议。
- **有两个行为目前没有测试覆盖，Step 0 先用测试把它们固定下来。** 两者都已在基线上运行验证，见附录 A：
  1. 在 max 深度退出时，sync 消费了 `signal_depth` 1..M−1，async 只消费了 1..M−2；两者的 exit depth 相同。
  2. async 下，如果请求在 `_pending_exit_signals` 里有条目时被 preempt，这个条目会保留下来；resume 后照常被消费，输出不变。

---

## 调用链与时序图

从请求入口到一个 batch 的调用链如下。实线箭头表示当前调用，`submit` 返回后的虚线表示之后的 `step()` 才会处理；下文第 1 节逐 stage 展开这些动作。

```text
add_request → Scheduler.add_request → WAITING
step → Scheduler.schedule → _admit / _take → SchedulerOutput
     → sync: ModelRunner.execute → LLMEngine._update / _update_speculative
     → async: ModelRunner.submit → Submission + event
              └─→ LLMEngine._update（PREFILL、PRELUDE）或提交进度（RECURRENT、CODA）
        后续 step → _inflight 退休 → _collect_coda → _deliver_coda
     → _finish → ModelRunner.release → KVCacheManager.poll_prefixes → Scheduler.finish
```

最简单的同步请求按 stage 串行推进。`PREFILL` 可分成多个 chunk；第一个 token 的 `CODA` 不需要 `PRELUDE/RECURRENT`，因为 prefill 已经完成全部深度。

```text
step: PREFILL chunk 1 ──execute──> 更新 prefill 进度 ──> PREFILL chunk 2
step: PREFILL chunk 2 ──execute──> publish_prefix ──> CODA
step: CODA ──execute/CPU token──> 输出 token 1 ──> PRELUDE
step: PRELUDE ──execute──> RECURRENT
step: RECURRENT depth 0 ──execute/CPU score──> 决定继续或退出
 ...
step: RECURRENT 退出 ──finalize──> CODA
step: CODA ──execute/CPU token──> 输出 token 2 / finish
```

异步请求把“提交”和“交付”分开。`PREFILL/PRELUDE` 在提交后就推进 host 状态；`RECURRENT` 在提交第 r 轮之后才取第 r−1 轮的 score；`CODA` 先增加一个 placeholder，之后才把 token 交付给 CPU。GPU event 只决定依赖和回收，不直接改调度状态。

```text
host step       ModelRunner / GPU                  Scheduler / EngineCore
─────────       ─────────────────                  ──────────────────────
PREFILL         submit chunk ──event─────────────> 推进进度；prefix 等 event 发布
CODA 1          submit sample ──event────────────> placeholder=1；device token 已可路由
PRELUDE         submit(device token) ────────────> 重置 loop；不等 CODA 1 的 CPU 值
RECURRENT r=0   submit score s1 ──event───────────> loops_done=1；保留 s1 handle
RECURRENT r=1   submit score s2 ──event───────────> collect(s1)；判定是否退出
...             GPU event 完成                      可退休 submission / 回收 readback lease
后续 step       collect(CODA 1 token) ───────────> placeholder=0；输出 token 1
退出时          finalize 等最后 core event ──────> CODA 2 可提交
```

对同一请求，下一次 `CODA` 提交前必须交付上一个 placeholder；不同请求的 CODA ticket 可以按 ready 顺序交付。preemption 会把 Request 和 KV 保存后放回 WAITING，但不清掉 pending signal；resume 后仍按原来的位置和深度消费它。abort 或 finish 则释放资源并丢弃 handle。

---

## 1. 时序表与状态归属

### 1.1 sync 路径（`step()`，llm_engine.py:209-226）

| Stage | 提交 | GPU 完成 | CPU 交付 | 今天的结果应用（`_update`） |
|---|---|---|---|---|
| PREFILL | `model_runner.execute()` 在当前流上跑完这个 chunk 的全部深度，返回 `None`，host 不等（model_runner.py:353-355, 434-437） | host 不观察，由单流顺序保证 | 无 | 执行完立刻做：`num_prefilled_tokens += token_count`；`publish_prefix(event=None)` 立即发布；prompt 已全部处理则设 `loops_done=total_ut_steps` 并进入 CODA，否则继续 PREFILL（:269-282） |
| PRELUDE | 用 host 上的 `input_token_id` 做 embedding，返回 `None`（model_runner.py:377-383） | 同上 | 无 | 重置 `loops_done/remaining_probability/pending_exit_depth`，进入 RECURRENT（:283-288） |
| RECURRENT | 跑一轮 core（depth=`loops_done`），对 gate/lookahead 做 sigmoid，然后 `.cpu().tolist()` | 非 trace 模式：在 `.cpu()` 处阻塞，这一刻即完成。trace 模式：没有结果也没有 readback，host 不观察 | 与完成同时拿到本轮 score | `loops_done += 1`，再按模式判断是否退出：trace；`ouro`（先乘 hazard，再调 `_should_exit`）；delayed（调 `_delayed_exit`）。退出则 `finalize` 并进入 CODA，否则留在 RECURRENT（:289-302） |
| CODA | 跑 coda，在 device 上采样，然后 `.cpu().tolist()` | 在 `.cpu()` 处阻塞 | 与完成同时拿到 token id | 追加 token 和 `loops_done`；遇 EOS 调 `_finish(STOP)`，到长度上限调 `_finish(LENGTH)`，否则进入 PRELUDE 或 SPECULATIVE；每行生成一个 `RequestOutput`（:303-317） |
| SPECULATIVE | `speculative_runner.execute()` 执行 draft 和 verify，内部有多处 `.item()`（例如 speculative.py:86, 104, 123） | execute 内部已同步 | `SpeculativeResult` 返回时 | `_update_speculative`：逐个追加 token（depth=`target_loops`），遇 EOS 或长度上限截断；更新 stats；未结束的请求执行 `truncate_suffix(token_start+emitted)`、设 `loops_done=0`，回到 SPECULATIVE（:228-257） |

补充说明：

- **为什么 prefix 可以立即发布**：engine 的 sync 路径中，`model_runner.events` 一直是空的。只有 `submit()` 和 async 的 `finalize*` 会记录 event（model_runner.py:488-494, 561, 577-580）。所以 prefix 直接发布，正确性靠单流顺序保证。
- **空调度**：如果不存在非 RECEIVING 的请求，返回 `[]`；否则抛出 `"scheduler made no progress"`（:211-215）。
- **失败**：执行或结果应用出错时，对该 batch 中仍在注册表里的请求执行 abort，然后重新抛出（:221-226）。

### 1.2 async 路径（`_step_async()`，llm_engine.py:404-496）

每步开始时依次做三件事：

1. 退休已 ready 的 `_inflight`，释放它们的 readback lease（:406）。
2. 交付已 ready 的 CODA（:407）。
3. 调用 `schedule(prefer_recurrent=_overlap_boundary and 存在未完成的 RECURRENT)`（:408-415）。

| Stage | 提交（host 进度 = 已提交进度） | GPU 完成（event） | CPU 交付 | 今天的结果应用 |
|---|---|---|---|---|
| PREFILL | 在 core stream 上 `submit`；先等本请求的上一个 event；没有 readback；event 记入 `events[rid]`（model_runner.py:448-494） | 用于判断：prefix 能否发布（`poll_prefixes`）、后续提交何时可以开始（`wait_event`）、`release` 何时可以回收 | 无 | 提交后立即调用 `_update(batch, None)`，逻辑与 sync 相同，只是 `publish_prefix(event)` 要等 event 完成才真正发布（:456-457） |
| PRELUDE | 在 boundary stream 上执行；输入是上一个 CODA 留在 device 上的 token（`input_token_tensor` 或 routing token pool），不调用 `.item()`；用完即清空（model_runner.py:330-333, 360-376） | 作为下一轮 core 的依赖 | 无 | 提交后立即调用 `_update(batch, None)` |
| RECURRENT | 在 core stream 上执行；score 用非阻塞 D2H 拷贝到预分配的 pinned slot（:482-487） | 完成后：readback 可读、`_inflight` 可退休、`finalize` 不再等待；也用于 overlap hint | 本轮 score 要等**同一请求下一次** RECURRENT 提交之后才被 `collect()`：那时 host 等的是第 r−1 轮，第 r 轮已经在 GPU 上运行 | 见下文"RECURRENT 提交后"（:458-495） |
| CODA | 提交前：如果 batch 里有请求还带着 placeholder，先强制交付，再按身份过滤；过滤后为空就直接返回（:424-440）。提交时：在 boundary stream 上采样，写入 routing token pool，发起 readback，记录 event | readback 完成后 `ready()` 为真 | 见下文"CODA 交付"（:366-390, 392-402） | 见下文"CODA 提交后"（:443-455） |
| SPECULATIVE | 不存在：构造时就拒绝 async（:44-47） | – | – | – |

**RECURRENT 提交后**（:458-495）：

1. pop 出这个请求上一次登记的 handle。
2. `loops_done += 1`。
3. 先按 trace 或 max 判断是否退出。
4. 若不退出，校验 position 和 depth，不一致就抛 `"stale lookahead signal"`；一致则 `collect()` 上一轮的 score，交给 `_delayed_signal` 判断。
5. 退出：设 `pending_exit_depth=loops_done`，进入 CODA。
6. 不退出：如果是 delayed 模式且 `exit_threshold<1`，登记新 handle；然后留在 RECURRENT。
7. 最后调用 `finalize_many(exited)`。

**CODA 提交后**（:443-455）：

- 把本次提交放入 `_pending_coda`，并设置 `_overlap_boundary`。
- 每行 `num_output_placeholders += 1`。
- 若 `num_scheduled_outputs < max_tokens`，设 `input_token_tensor = device_values[i]`，并进入 PRELUDE。
- 最后一个 token 不进入任何队列，只等交付。

**CODA 交付**：

- **时机**（:392-402）：每步开头交付所有已 ready 的；遇到空调度或需要强制交付时，阻塞交付一个。
- **逐行处理**（:366-390）：
  - 身份不符的行直接跳过。
  - `output_index != len(generated_token_ids)` 时抛 `"out-of-order coda delivery"`。
  - placeholder 不等于 1 时抛 `"invalid pending output count"`。
  - 通过检查后：placeholder 减 1，追加 token 和**提交时快照的深度**，再判断 EOS 和长度上限。

**空调度**：若有 `_pending_coda`，强制交付一个；否则只要存在非 RECEIVING 的请求就抛错（:417-423）。

**失败**：先 `synchronize()`，然后 abort 全部请求，清空 `_pending_exit_signals`、`_pending_coda`、`_inflight`，最后重新抛出（:196-208）。

### 1.3 LLMEngine 上的状态与 R1 后的 owner

| 状态 | 内容 | R1 后的 owner | 依据与不确定点 |
|---|---|---|---|
| `_pending_exit_signals`（:115-117） | `rid → (Submission, row, position, depth)` | **拆分**：handle（Submission、row、身份）归 EngineCore；信号的校验、消费和累计退出状态归 Scheduler | `Submission` 持有 event 和 pinned readback slot（model_runner.py:36-65），属于设备资源句柄，不应进入 scheduler；RFC 的状态表也把 "Logical records of pending execution" 划给 scheduler。**不确定**：R3 时是否改由 M5 持有常驻 device 的信号，并随下一次提交一起返回 |
| `_pending_coda`（:118） | 尚未交付 CPU 的 CODA Submission | EngineCore | 逻辑上的 placeholder 已经在 `Request.num_output_placeholders`，属于 scheduler 侧 |
| `_inflight`（:119） | 尚未 ready 的 Submission | R1 归 EngineCore；**不确定**，R3 可能并入 M5 | 它有两个作用：维持 readback lease 的寿命（:405；model_runner.py:47-49, 439-446），以及判断能否 overlap（:412-413）。M5 自己已经有 `submission_events`（model_runner.py:140, 450-453） |
| `_overlap_boundary`（:120） | 一次性的 overlap hint | EngineCore | 取决于 M5 的 stream 拓扑（:445-449）；通过已有的 `prefer_recurrent` 参数传给 M3 |
| `last_schedule`（:121） | 最近一次的 SchedulerOutput（可能被过滤过），或 None | EngineCore（诊断用） | 测试和 benchmarks/cdb_runtime.py:27 会读它 |
| `_exit_traces`（:112-114） | trace 表 | Engine 接口（输入校验）；**不确定** | 只有 `add_request` 读它（:156-171），真正消费它的是 `_trace_exit`，后者归 M3。pd/engine.py:301-314 还有一份重复的校验。它也可以改为 M3 exit policy 的配置 |
| `preemption` 及回调绑定（:122-125） | PreemptionManager | 装配归 EngineCore；策略归 M3，设备拷贝归 M5，KV 归 M4 | PR #50 在把队列迁移移入 scheduler；M2 不改这部分 |
| `model`、各 config、`memory_plan`、`cache_manager`、`scheduler`、`model_runner`、`speculative_runner` | 组件引用 | EngineCore（装配） | 结果应用需要的常量（eos ids、`total_ut_steps`、exit mode、`target_loops`）在构造 scheduler 时传入 |

### 1.4 engine 写入的其它模块状态

| 被写的状态 | R1 后由谁写 |
|---|---|
| Request 上的 `num_prefilled_tokens/loops_done/remaining_probability/pending_exit_depth/generated_token_ids/exit_depths/num_output_placeholders`，以及通过 `enqueue` 修改的 stage | Scheduler，经由 `update_from_output` |
| `Request.input_token_tensor`（:454） | 本质上是 ModelRunner 的 device token。R1 期间由 EngineCore 里的 shim 写入，R3 交还给 M5 |
| `speculative_runner.stats.committed/accepted_tokens`（:247-248） | EngineCore，按 scheduler 返回的计数更新（见 Q18） |
| `publish_prefix`、`truncate_suffix` | Scheduler 的结果应用 helper。这两个是 M4 的公开 API；完成证据由调用方原样透传 |
| `poll_prefixes`、`finalize*`、`release`、`synchronize` | EngineCore |

---

## 2. 共享契约

### 2.1 SchedulerOutput（归 M3；下面列出的是 M2 用到的字段）

| 字段 | 今天 | R1 | 说明 |
|---|---|---|---|
| `stage`、`items[i].token_start/token_count` | 有 | 不变 | PREFILL 和 SPECULATIVE 的 token 区间 |
| `items[i].request` | 有 | 只为兼容 M5 而保留 | runner 仍从这里读 hidden、token 和 RNG；M2 不再读它的调度字段；R3 由 M5 去掉 |
| `seq` | 无 | 新增 | execution sequence。每个非空 `_take` 返回的 SchedulerOutput 唯一；async CODA 用 `replace` 过滤后沿用原来的 seq；手工构造的兼容 batch 默认为 0 |
| `items[i].request_id / generation` | 无 | 新增 | generation 在 `Scheduler.add_request` 时从单调递增计数器分配，preempt/resume 期间保持不变；手工构造的 Request 默认为 0 |
| `items[i].position` | 无 | 新增快照 | token position，与 loop depth 分开 |
| `items[i].loops_done` | 无 | 新增快照 | 单位是"已完成的轮数"。对 RECURRENT，它就是本轮要执行的 0-based depth；对 CODA，它就是 exit depth |
| `items[i].output_index` | 无 | 新增快照（仅 CODA） | 值等于 `num_scheduled_outputs` |
| 逻辑 block 信息 | 无 | R1 不做 | 属于 R2（M4/M5） |

**快照在 `_take` 时取，与 `ModelRunner.prepare()` 的快照（model_runner.py:326-328）等价。** 从 `_take` 到 `submit` 之间，只有 async CODA 的强制交付会修改 Request：它使 `generated_token_ids` 加 1、placeholder 减 1，而 `num_scheduled_outputs`、`position`、`loops_done` 都不变。Step 1 会用测试断言这一点。

**兼容性约束**：`ScheduledItem(r, 3, 4)` 这种按位置传参的构造方式（test_speculative.py:110）必须继续可用。

### 2.2 ModelRunnerOutput

`Progress` 字段说明这份输出能证明什么：

- `COMPLETED`：sync 路径。`execute()` 已返回，本 seq 的 host 结果可用；设备依赖与回收仍由 runner/stream 保证，不以此标记推断 GPU 完成。
- `SUBMITTED`：async 刚提交。本 seq 的值还拿不到；这份输出最多只带上更早 seq 中已解析出的信号。
- `DELIVERED`：async CODA 的 token 已到达 CPU；对应的 `SUBMITTED` 之前已经应用过。

GPU 完成不是 `Progress` 的一个取值；它只以不透明的完成证据形式出现（见下表 `completion` 字段）。

| 字段 | 含义 | sync `COMPLETED` | async `SUBMITTED` | async `DELIVERED` |
|---|---|---|---|---|
| `seq`、`stage`、`progress` | 标识本输出；`stage` 只用于校验，不用来推断其它字段的含义 | 本次的 seq | 本次的 seq | 原 CODA 提交的 seq |
| `prefill_ranges` | 与行对齐的 `(token_start, token_count)` | PREFILL | PREFILL | – |
| `completion` | 与行对齐的不透明完成证据 | `None`（单流顺序已保证） | `model_runner.events[rid]` | – |
| `exit_signals` | 元素为 `ExitSignal(request_id, generation, position, signal_depth, source_seq, score)`，每个元素自带身份 | RECURRENT：本 seq 的每一行都有（trace 模式为空） | RECURRENT：只包含本 batch 中持有 handle 的行，内容是这些行**上一轮**的信号 | – |
| `sampled_token_ids` | 与行对齐的 host token id | CODA | 不提供（token 仍在 device 上） | CODA |
| `speculative` | 与行对齐的 `(token_ids, accepted_count, draft_count)` | SPECULATIVE | – | – |

规则：

- **`signal_depth` 的单位**：产生该 score 的那一轮完成后的已完成轮数，即产生它的那一行的 `loops_done + 1`。这和今天 `_pending_exit_signals` 第 4 项、`_delayed_signal(signal_depth)` 的单位一致。
- **不含 device tensor**：输出中不放任何 device tensor。
- **结构**：每个 batch 一个对象，字段按列存放。
- **不新增 host 读取**：adapter 只包装今天已有的两处读取，即 `execute()` 里的 `.cpu().tolist()` 和 `collect()`。
- **不在 R1 范围**：PD 的 transfer 完成反馈留给 M9（R4），R1 不加。

### 2.3 唯一的结果应用入口

入口为 `Scheduler.update_from_output(scheduler_output, runner_output) -> SchedulerUpdate`，名字由 M3 定。它不调用 runner，也不会阻塞。

下表列出必须保持的现有行为；具体用哪些 helper 实现由 M3 决定。

| Stage | `COMPLETED`（sync） | `SUBMITTED`（async） | `DELIVERED`（async） |
|---|---|---|---|
| PREFILL | 推进 prefill 进度；调用 `publish_prefix(completion)`；prompt 全部完成则进入 CODA（`loops_done=total_ut_steps`），否则继续 PREFILL | 同左，但 completion 是 event | – |
| PRELUDE | 重置 loop 状态，进入 RECURRENT | 同左 | – |
| RECURRENT | `loops_done += 1`，用本轮信号按模式判定是否退出 | `loops_done += 1`；先按 trace 或 max 判定；未退出再用上一轮信号判定；返回 `retain_signal` | – |
| CODA | 追加 token 和深度；判断 EOS 和长度；若都未触发，进入 PRELUDE 或 SPECULATIVE | placeholder 加 1；若 `num_scheduled_outputs<max_tokens`，进入 PRELUDE 并记入 `next_prelude` | 做两项校验；placeholder 减 1；追加 token 和快照深度；判断 EOS 和长度 |
| SPECULATIVE | 逐个追加 token，判断 EOS 和长度；若都未触发，调用 `truncate_suffix`、设 `loops_done=0`，回到 SPECULATIVE | – | – |

EngineCore 拿到 `SchedulerUpdate` 后按字段执行下列动作，自己不做任何调度决策：

| 字段 | EngineCore 的动作 |
|---|---|
| `exited` | 调用 `finalize_many(...)`。sync 路径也改为调用它：`async_state is None` 时，它会逐个调用 `finalize`（model_runner.py:508-511），行为不变 |
| `finished`，元素为 `(request_id, generation, FinishReason)` | 保持今天 `_finish` 的顺序：`release`（等待最后一个 event）→ `poll_prefixes` → 丢弃 signal handle → `scheduler.finish` |
| `output_rows` | 在 finish 之后构造 `RequestOutput`，确保 finished 和 reason 正确 |
| `retain_signal` | 登记 handle，内容为 `(seq, ticket, row, generation, position, signal_depth)` |
| `next_prelude` | R1 的 shim：`input_token_tensor = ticket.device_values[row]` |
| `speculative_committed` | 更新 stats |

### 2.4 sync 的调用时机

```text
_step_sync()
  out = scheduler.schedule()                       # None 时按 RECEIVING 规则处理
  raw = model_runner.execute(out)                  # SPECULATIVE 时改为 speculative_runner.execute(out)
  upd = scheduler.update_from_output(out, adapt(raw, COMPLETED))   # 本步只调用这一次
  finalize_many → finish → stats → outputs
  出异常时：abort out 中仍在注册表里的行，然后重新抛出
```

### 2.5 async 的调用时机

```text
_step_async()
  退休已 ready 的 _inflight
  对每个已 ready（或被强制交付）的 CODA ticket：
      update_from_output(ticket.batch, adapt(collect(), DELIVERED))    # 调用点 A：CPU 交付
  out = schedule(prefer_recurrent=hint)
      out 为 None：若有 pending CODA，强制交付一个（走调用点 A）；否则按 RECEIVING 规则处理
      out 为 CODA：反复强制交付，直到本 batch 中没有待交付的行（走调用点 A）；再按 generation 过滤
  ticket = model_runner.submit(out)
  out 为 RECURRENT：对持有 handle 的行调用 collect()。此时等待的是第 r−1 轮，第 r 轮已经提交
  upd = update_from_output(out, adapt(ticket, SUBMITTED, 已解析的信号))  # 调用点 B：提交
  按 retain_signal 登记 handle；执行 next_prelude shim；finalize_many(exited)
  GPU 完成本身从不调用 update_from_output，只作为 M4/M5 的资源释放条件
```

示例设定：单个请求，`ouro_delayed`，`min_loops=1`，threshold 0.5，每轮 hazard 0.3。前 3 步依次是 PREFILL、CODA、PRELUDE，下面从第 4 步开始。

```text
sync                                       async
4  REC d0 → s@1，消费（累计 .30）          4  submit REC d0(seq4)；保留 s@1
5  REC d1 → s@2，消费（累计 .51）          5  submit REC d1(seq5)；collect(seq4)，消费 s@1；保留 s@2
   → pending=3
6  REC d2：loops_done == pending，退出     6  submit REC d2(seq6)；collect(seq5)，消费 s@2 → 在第 3 轮退出
   （不消费 s@3）→ finalize → CODA            → finalize_many → CODA
```

在 GPU 上，CODA 的交付可以晚于下一个 PRELUDE 或 RECURRENT 的提交。

---

## 3. 迁移步骤

测试集缩写：

- **G-CPU**：上面列出的 8 个测试文件（基线为 192 passed / 57 skipped），加上新增测试。
- **G-GPU**：在分配到的 GPU 上加 `--run-gpu` 运行。包括 test_async_state.py、其它文件中的 GPU 用例，以及 tests/test_cuda_graph.py；`tests/conftest.py` 要求分配环境提供 `CUDA_VISIBLE_DEVICES`。

**每步开始前需要确认的待定问题**（问题编号见第 6 节）：

- Step 1：Q3
- Step 2：Q12、Q13
- Step 3：Q1、Q6、Q7、Q16，以及 Q19 中 #60 的合入顺序
- Step 4：Q2、Q8、Q15，以及 Q19 中 #48、#52、#59 的合入顺序
- Step 5：Q4、Q5、Q9、Q17、Q18

### Step 0 — 完善本文并补行为固定测试（M2；只改文档和测试）

- 完善本文：补充调用链和由简到繁的时序图，结构沿用 M3 的 walkthrough。
- 新增测试。测试尽量只通过输出和 Request 字段观察行为：
  - T1：delayed 信号的消费次数。sync、async 各测两种情况：到 max 深度退出、由信号触发退出。观察方式是请求进入 CODA 时的 `remaining_probability`。
  - T2：async 下，请求持有 pending signal 时被 preempt，resume 后输出不变。
  - T3：async 下，请求持有 handle 时被 abort，之后复用同一个 id。
  - T4a：stale signal 应当报错。
  - T5：CODA 的 `output_index` 不符、或 placeholder 不等于 1 时，应当报错。
  - T6：sync 下，结果应用阶段抛错时，只 abort 该 batch 中的行。
  - T7：async 下，结果应用阶段抛错时，完整执行清理。
  - T8：后提交的 CODA ticket 先 ready 时，先交付它。
- T4a 和 T5 需要注入内部状态，属于结构性测试，以后随状态一起迁移。T4b 依赖 generation，放到 Step 2 再加。
- T9：改写 `test_engine_yields_while_waiting_for_remote_kv`。它现在通过 `object.__new__` 手工设置 4 个私有字段，再塞一个 Mock ticket（tests/test_pd.py:241-268）。改为用真实的 CPU `LLMEngine` 构造，断言保持不变，这样之后每一步都不必再改它。
- 必须通过：G-CPU、G-GPU。

### Step 1 — M3 前置契约（M3 的小 PR，M2 review）

- 新增 `Request.generation`、`SchedulerOutput.seq`，以及 2.1 中的快照字段。
- 必须通过：全部测试，外加以下新测试：
  - 复用 id 时 generation 会变化；
  - preempt 前后 generation 保持不变；
  - 新增的快照与 `prepare()` 的结果一致。

### Step 2 — 类型化输出与 adapter（M2；类型放在哪里由 M5 review）

- **不动的部分**：结果应用逻辑仍留在 engine 内。
- **改动**：
  - `_update`、`_deliver_coda`、`_update_speculative` 和 async RECURRENT 块，改为读取 `ModelRunnerOutput` 的字段，不再用 `result[index]`。
  - EngineCore 判断身份时改用 `(request_id, generation)`，涉及 :224、:371、:435 三处。
  - handle 记录新增 `generation` 和 `source_seq` 两个字段。
- **保留的适配**：新增 4 个 adapter，分别对应 sync execute、speculative execute、async submit、async delivery。它们是 R1 期间的临时接缝。
- **调用方**：只改 engine 内部。以下调用的位置和次数都不变，因为有测试对它们做 monkeypatch：
  - 被 patch 的调用：`model_runner.execute/submit`、`Submission.collect/ready`、`speculative_runner.execute`。
  - 相关测试位置：test_serving.py:53-63, 439-449；test_cdb_runtime.py:225-241, 373-383, 413-417；test_async_pipeline.py:275-286；test_speculative.py:338。
- **必须通过**：全部测试，且测试代码不做修改。另外补充：
  - T4b：旧 generation 的 handle 或 CODA 行会被丢弃；
  - adapter 单测；
  - `collect()` 调用次数与顺序的前后对照；
  - 用 benchmarks/cdb_runtime.py 对比每一步的 CPU 时间。

### Step 3 — PREFILL/PRELUDE 迁入 Scheduler（M3 写 scheduler 侧，M2 切换调用）

- **移动**：`_update` 的 PREFILL/PRELUDE 分支（:269-288），连同 `publish_prefix`。
- **保留的适配**：engine 内按 stage 临时分派：已迁移的 stage 走 `update_from_output`，其它仍走 `_update`。PRELUDE 行的 handle 清理（等价于 :287）留在 EngineCore。
- **调用方**：sync 的 :219-220；async 的 :456-457。
- **约束**：`num_prefilled_tokens` 等仍作为 Request 字段保留。原因是 PD 的 `prefill_step`（pd/worker.py:257-275）和 decode 激活（:331-342）会直接写这些字段。
- **必须通过**：全部测试。重点关注：
  - CPU：`test_chunk_limit_allows_short_prompt_before_long_prefill_finishes`、`test_prefix_reuse_matches_cold_generation…`、`test_incremental_growth…`、`test_padded_chunked_execution_matches_serial`、`test_prelude_consumes_device_sample…`。
  - GPU：`test_fa4_prefill_uva_prefix_growth_and_bank_reuse`、`test_pd_prefix_reuse_*`、test_async_state.py。

### Step 4 — CODA、SPECULATIVE、EOS/长度迁入 Scheduler

- **移动**：
  - :303-317：`_update` 的 CODA 分支。
  - :371-389：`_deliver_coda` 的校验与写入。
  - :450-455：CODA 提交后的 placeholder 更新与 enqueue。
  - :230-256：`_update_speculative` 的写入与 `truncate_suffix`。
- **留在 EngineCore**：
  - `_pending_coda` 和交付调度（:392-402, 419-420, 428-429）；
  - CODA 提交前的过滤；
  - `_overlap_boundary`；
  - `_finish` 的执行顺序（只负责执行，不再做判断）；
  - shim、stats 和输出构造。
- **新增**：`update_from_output` 单测，断言 EOS 的行只出现在 `finished` 中、不会入队。
- **必须通过**：全部测试。重点关注：
  - CPU：`test_eos_finishes…`、`test_sampled_generation_is_batch_invariant`、`test_large_coda_threshold…`、`test_delayed_delivery_uses_snapshot_depths…`、`test_delayed_eos_discards_next_work…`、`test_cancel_pending_coda_and_reuse_request_id`、`test_pending_coda_does_not_block_other_recurrent_work`、T9、test_speculative.py 的全部 CPU 用例、test_serving.py。
  - GPU：`test_completed_core_refills…`、`test_abort_speculative_core_then_reuse_id`、`test_cuda_async_matches_synchronous…`、`test_cuda_auto_memory_plan_and_abort_pending_work`。

### Step 5 — RECURRENT 与 exit policy 迁入 Scheduler，删除旧路径

- **移动**：
  - `_update` 的 RECURRENT 分支（:289-302）；
  - `_should_exit`（:320-329）、`_trace_exit`（:337-339）、`_delayed_exit`（:341-349）、`_delayed_signal`（:351-364）；
  - async 中 :463-494 的判定部分。
- **留在 EngineCore**：
  - `_pending_exit_signals`，其元素改为 handle 记录；
  - 提交后解析 handle，顺序仍是"先提交第 r 轮，再等第 r−1 轮"；
  - 按 `retain_signal` 登记 handle；
  - `finalize_many(exited)`。
- **必须通过**：全部测试。重点关注：
  - CPU：gate、threshold、refill、chunked 相关用例；test_cdb_runtime 中 delayed、trace、`ouro_delayed` 以及"先 submit 后 collect"的用例；三个 preemption 用例；T1–T4。
  - GPU：test_async_state.py 全部；`test_bounded_pipeline_survives_slow_gpu…`（它断言 `not engine._pending_exit_signals`）；`test_cuda_dynamic_async…`；`test_cuda_boundary_and_core_can_overlap`；`test_async_pressure_preemption_with_resident_state`；tests/test_cuda_graph.py。
- **性能**：本步改动了热路径，需按风险-A 做受控对比。

### `_should_exit` 与 engine 中 `_update` 的写入何时消失

- `_update` 中的写入分三步迁出：PREFILL/PRELUDE 在 Step 3，CODA 在 Step 4，RECURRENT 在 Step 5。Step 5 合入时，整个 `_update` 方法删除。
- `_should_exit`、`_trace_exit`、`_delayed_exit`、`_delayed_signal` 在 Step 5 删除，与 RECURRENT 结果应用的迁移放在同一个 PR 里；按 stage 临时分派的代码也在这时删掉。
- 在任何时刻，同一个 stage 的逻辑都不会在 engine 和 scheduler 中各保留一份。

### R1 完成判据

- `LLMEngine` 中不再有 exit policy、EOS/长度判断和请求的后继 stage 选择；按 `SchedulerOutput.stage` 分派 runner 的执行仍留在 engine。
- `LLMEngine` 不再调用 `scheduler.enqueue`；`PreemptionManager` 的 WAITING/resume 入队留给 #50 的队列迁移。
- `LLMEngine` 的结果路径对 Request 的写入只剩 `input_token_tensor` shim 这一处；preemption 的 CPU 快照/恢复保持原样。
- 1.1 和 1.2 两张表中的每一格都与迁移前一致。

### R1 之后仍保留的部分

以下每一项都已指定移除者：

- **M5（R3）**：adapter、shim、以 Request 为参数的 `finalize_many`、`items[i].request`。
- **M9（R4）**：PD 中重复实现的 PREFILL 结果应用，以及 decode 激活。
- **M1**：对 `engine.scheduler.requests` 与 `engine.cache_manager` 的直接依赖，以及类的拆分。当前上游已提供 profiling 的 `close()`，R1 保留它。

---

## 4. 场景覆盖

| 场景 | sync 预期 | async 预期 | 已有测试 | 新增 |
|---|---|---|---|---|
| 正常完成（EOS、长度） | 在 CODA 应用时判定 STOP 或 LENGTH；不会多 decode 一步；KV 全部归还 | 在 DELIVERED 时判定。已提交的下一个 prelude/core 作废；`release` 等最后一个 event；不再采样；最后一个 token 不入队 | `test_eos_finishes…`、`test_delayed_eos_discards_next_work…`（CPU/CUDA）、`test_delayed_delivery_uses_snapshot_depths…`、`test_eos_in_accepted_prefix…` | Step 4 的 `update_from_output` 单测 |
| 取消 | 从所有队列中移除，释放 KV | pending CODA 在交付时跳过；handle 在 abort 时丢弃；`release` 等 event | `test_abort_removes_all_queued_work…`、`test_cancel_pending_coda_and_reuse_request_id`；GPU：`test_abort_speculative_core_then_reuse_id`、`test_cuda_auto_memory_plan_and_abort_pending_work` | T3 |
| 执行异常 | 只 abort 失败 batch 中仍注册的行，其它请求不受影响；`update_from_output` 抛错也算在内 | `synchronize()` 后 abort 全部，清空 in-flight，重新抛出；提交到一半失败也覆盖 | `test_execution_failure_releases_affected_requests`、`test_async_failed_submission_reclaims_all_affected_state`；GPU：`test_execution_error_retires_routing_readers…`；`test_dynamic_arrival_budget_and_failure_reclamation`；serving 的 `PauseAt(fail=True)` | T6、T7 |
| 迟到的结果 | 不会出现（host 会阻塞等待）；保留 generation 检查作为防御 | 旧 generation 的 CODA 交付会被跳过；handle 在 abort/finish 时丢弃；即使旧信号到达，generation 不符也会被丢弃，不报错 | `test_cancel_pending_coda_and_reuse_request_id` | T4b |
| 复用 request-id | 复用后分配新的 generation | 同 sync；M5 的 AsyncState 另有 owner 检查 | GPU：`test_resident_state_trace_refill_abort_and_slot_reuse`；`test_prefix_replay_incremental_growth_cancel_and_reuse` | T3 |
| CODA 乱序交付 | 不适用 | 同一请求内乱序会抛 `"out-of-order coda delivery"` 或 `"invalid pending output count"`；不同请求之间可以乱序，因为每个请求最多只有一个未交付输出 | 无直接测试 | T5、T8 |
| stale lookahead signal | 不适用 | generation 相同但 position/depth 不符时，抛 `"stale lookahead signal"`；generation 是旧的则直接丢弃 | 无 | T4a |
| 空调度且有请求处于 RECEIVING | 返回 `[]`，不调用 synchronize；若存在非 RECEIVING 请求，抛 `"scheduler made no progress"` | 先交付 pending CODA；之后同 sync | `test_engine_yields_while_waiting_for_remote_kv`（sync/async） | T9（断言不变） |
| preemption 时有 pending signal | – | resume 后信号仍被消费，exit depth 不变 | 无（已在基线验证，见附录 A） | T2 |
| delayed 信号的消费次数 | 在 max 深度退出时消费 1..M−1 | 消费 1..M−2 | 无（已在基线验证，见附录 A） | T1 |

---

## 5. 保持不变的语义

### 5.1 退出公式

- **`ouro`（仅 sync）**：每一轮都先执行 `remaining *= 1−σ(g)`，低于 `min_loops` 的轮次也计入。满足以下任一条件时退出：
  - `loops_done >= max_loops`（`max_loops` 默认为 `total_ut_steps`）；
  - `exit_threshold<1`，且 `loops_done >= min_loops`，且 `1−remaining >= exit_threshold`。
- **`ouro_delayed`**：在 k 处消费 hazard。当 k ≥ `min_loops` 且累计值 ≥ threshold 时触发，在第 k+1 轮退出。
- **`random_lookahead`**：score 直接与 threshold 比较，且要求 k+1 ≥ `min_loops`。触发后在第 k+1 轮退出。
- **所有模式共同的规则**：
  - threshold 为 1 时关闭自适应退出；
  - `max_loops` 永远优先；
  - trace 模式在 `loops_done >= exit_trace[num_scheduled_outputs]` 时退出；
  - prefill 和第一个 token 都跑满深度。

### 5.2 delayed 信号的消费

- 每个信号至多消费一次。
- 第 k 轮的信号触发后，在第 k+1 轮退出；最后一轮的信号从不被消费；不会为此多跑一轮投机 loop。
- sync 在第 n 轮完成时消费 s_n。若此时已到 max，或 `pending_exit_depth==n`，则不消费。
- async 在第 n 轮提交之后消费 s_{n−1}。若已到 max，则不消费。
- 由此，在 max 深度退出时，两条路径消费的信号数量相差 1。T1 会固定这一点，结构性 PR 不统一它。
- `ouro` 是"先乘 hazard 再判断"，delayed 模式是"先判断再消费"，两者也不统一。
- async 在第 r 轮提交之后才 `collect` 第 r−1 轮的结果，所 collect 的集合和顺序都不变。

### 5.3 CODA 只有一个 placeholder

- 每个请求最多只有一个未交付的输出。
- 如果 CODA batch 中有待交付的行，提交前先强制交付。每次调用最多阻塞一个，同时交付所有已 ready 的。
- 输出交付之前，下一个 prelude/core 可以先运行（即使该 token 可能是 EOS），但不会采样新的 token。
- 最后一个 token 不进入 PRELUDE。
- 记录的深度使用提交时的快照。
- 两个 RuntimeError 保留。
- `num_scheduled_outputs` 包含 placeholder。

### 5.4 device 上的 token

- CODA 的 `device_values` 直接作为下一个 PRELUDE 的输入；CUDA 下经过 routing token pool 传递。整条路径上没有 `.item()` 或 `.tolist()`。
- host 只通过非阻塞 readback 获取要交付的值。
- preemption 对 `input_token_tensor` 的快照和恢复逻辑不变。

### 5.5 speculative 支持同步 eager / graph

- 构造期校验保持当前上游语义：必须是 `last_exited`；必须是固定深度的 `ouro`；目标深度等于满深度；threshold 为 1；使用 refill；不支持 preemption、async、PD。#48 已加入同步 speculative graph 支持，R1 保留该路径与 runner 的 execution config。
- 只有 `COMPLETED` 一种结果应用。
- `truncate_suffix`、stats 以及 `target_loops` 深度的语义都不变。

### 5.6 资源释放条件

- **prefix 发布**：sync 立即发布；async 等 event 完成后发布。
- **finish 的顺序**：`release` → `poll_prefixes` → `free`。abort 的顺序本来就不调用 poll，也不改。
- **readback lease**：用完才释放；复用前先等 slot 的 event。
- **提交上限**：同时在飞的 submission event 最多 3 个。
- **依赖顺序**：`finalize` 等最后一个 core event；CODA 等 `finalize`。
- **transfer lease**：会推迟 free。

### 5.7 错误处理与空调度

与 1.1、1.2 的描述一致。

### 5.8 调度决策

- `prefer_recurrent` 的计算方式不变，batch 的组成不变。
- 过滤前的 async CODA batch 仍计入 `selected_request_ids` 和 `record_batch`。
- `last_schedule` 仍为过滤后的 batch；没有 batch 时为 `None`。

### 5.9 位置与 KV

- token position 与 loop depth 分开。
- LAST_EXITED 和 SHARED 的 finalize 行为不变。
- 遇到 RECEIVING 时让出执行。
- PD transfer 的语义不变。

### 5.10 对外接口

- `add_request/abort_request/step/has_unfinished_requests` 不变。
- `RequestOutput` 的字段不变，finish reason 仍为字符串。
- engine 上现有的公开属性全部保留。

---

## 6. 决策与剩余确认

以下是以 `d286233` 为基线的本地实现选择，供 M3/M5 review；下面各 Q 项保留了草案时的备选方案与理由。

| 问题 | 本地实现选择 |
|---|---|
| Q1–Q3 | `update_from_output(batch, result) -> SchedulerUpdate`；采用 F1，由 EngineCore 按 `release → poll_prefixes → scheduler.finish` 执行 finish。generation 在注册时分配，seq 在非空 `_take` 时分配，行快照在 `_make_scheduled_item` 时取得。 |
| Q4–Q5 | 采用 P；sync/async 分别按 `COMPLETED`/`SUBMITTED` 消费信号，async CODA 用 `DELIVERED` 追加 token。 |
| Q6–Q8 | prefix 发布与 speculative 截断由 Scheduler 调用 KV API；构造 Scheduler 时传入模型深度、exit mode、EOS 集合。`get_live(request_id, generation)` 查询身份；强制 CODA 交付通过 EngineCore 持有的 pending ticket 身份判断。 |
| Q10、Q12–Q18 | trace 输入校验仍在 Engine；类型位于 `worker/output.py`；四个临时 adapter 位于 `engine/output_adapter.py`。`_inflight`、device-token shim 和 speculative stats 仍由 EngineCore 持有；PREFILL completion 来自 `events[rid]`；退出后统一调用 `finalize_many`。 |
| Q9、Q11、Q19 | preemption、PD 重复结果路径和未合入 PR 的合并顺序未在 R1 本地变更；集成前需要对应 owner review。 |

### 6.1 与 M3 确认

- **Q1**：入口的签名；`SchedulerUpdate` 的 6 个字段。建议由 EngineCore 在 finish 之后构造 `RequestOutput`。
- **Q2**：finish 由谁执行，两个方案：
  - **F1**：scheduler 只返回决定，EngineCore 按 `release → poll_prefixes → scheduler.finish` 执行。
  - **F2**：scheduler 自己执行 finish，并通过回调先完成 release。
  - **依据**：`release` 会调用 `event.synchronize()`（model_runner.py:582-593），而 `Scheduler.finish` 会立即执行 `cache.free`（scheduler.py:89）。
  - **建议**：F1，它只用到现有的 `finish`。等 M4 的 deferred release 完成后可以再回头看。
- **Q3**：generation、seq、快照字段的定义与单位。generation 必须在 preemption 前后保持不变；否则 T2 中被保留的 handle 会被当作迟到结果丢弃，输出就会改变。
- **Q4**：delayed signal 如何拆分，有三个方案：
  - **P（建议）**：
    - 规则：只有在"下一次提交时一定会被消费"的情况下才保留 handle。具体条件是：delayed 模式、`exit_threshold<1`、继续 RECURRENT、且 `n+1 < max_loops`。
    - 好处：EngineCore 可以无条件解析所有 handle，不需要知道任何 policy。
    - 等价性：今天第 n 次提交时，collect 的条件是"第 n−1 轮登记过 handle，且 n < max"，这与 P 的登记条件完全相同。所以 collect 的集合、消费次数和 exit depth 都不变，handle 的寿命也不会比今天更长。
  - **L**：把 handle 作为惰性值交给 scheduler，由 scheduler 读取时阻塞。不建议，因为 scheduler 会因此等待设备。
  - **两段调用**：scheduler 先返回需要哪些信号，EngineCore 解析后再调用一次。缺点是在已经受 CPU 限制的路径上多一次调用。
  - 另外，"信号缺失时报错"会改变行为，需要单独提 PR。
- **Q5**：结果应用按 `Progress` 区分；保留 sync 与 async 在消费次数上的差异。
- **Q6**：`publish_prefix` 和 `truncate_suffix` 放进 scheduler 的 helper，需与 M4 一起确认。完成证据原样透传。
- **Q7**：scheduler 构造时需要的常量：
  - eos ids：engine 今天在 :233-234、:307-308、:383-384 三处把 int 和 list 统一成 list；
  - `total_ut_steps`；
  - exit mode；
  - `target_loops`。
- **Q8**：
  - EngineCore 需要一个只读查询，判断 `(request_id, generation)` 是否仍然存活。今天是直接读 `scheduler.requests`。
  - 强制交付的判断条件：用 placeholder，还是用 EngineCore 自己维护的待交付索引，需要二选一。在"每个请求最多一个未交付输出"的前提下，两者等价。
- **Q9**：与 PR #50 的合入顺序。preemption 必须继续排除带 placeholder 的请求（preemption.py:36），suspend 时也不能清掉 handle。
- **Q10**：`_exit_traces` 归谁。
- **Q11**：PD 的两处重复实现，需要与 M9 一起定。R1 先让 Request 字段继续充当共享状态。"P 端最后一个 chunk 直接 handoff、不进 CODA"是否需要单独的结果应用变体，留到 R4 决定。

### 6.2 与 M5 确认

- **Q12**：`ModelRunnerOutput` 放在哪里（建议 `vllm_rlt/worker/output.py`）；字段按列存放；性能预算。
- **Q13**：接缝的形状：
  - sync：`execute(out)` 返回 `COMPLETED`。
  - async：`submit(out)` 返回一个 handle，它提供 `ready()`、`delivered()`、`signal(row)` 和 `device_tokens`。
  - R3 时由 runner 直接返回这些，届时删除 adapter。
  - M2 不新建 Executor 类，这层 adapter 本身就是轻量的单进程入口。
- **Q14**：`_inflight` 最终归谁。
- **Q15**：device token 的 shim，以及 preemption 对 `input_token_tensor` 的快照（preemption.py:98-100, 135-137）。CUDA 下已有 AsyncState token pool（async_state.py:126-128）。
- **Q16**：完成证据用 `events[rid]` 还是 `ticket.event`。对 PREFILL 来说两者是同一个对象，需要确认这一点可以依赖。
- **Q17**：`finalize_many` 以 Request 为参数；sync 改用它之后是否等价。
- **Q18**：
  - 在方案 P 下，readback lease 的寿命不长于今天，`_readback_slot` 的同步行为也不变（model_runner.py:439-446）。
  - speculative stats 放在哪里。

### 6.3 跨模块

- **Q19**：几个未合入的 PR 与本计划冲突，必须约定合入顺序。原则是：R1 以 d286233 为基线，谁先合入，谁负责补上对应 stage 的契约。
  - **#60（WCPP，draft）**：改写了 `_update` 的 PREFILL 分支，新增 `prefill_task` 和 `advance_prefill_task`，并在 engine 中写 `hidden_state`。与 Step 3 冲突。
  - **#48 和 #52**：放宽了 speculative "只能同步 eager"的限制，并在 engine 上新增 `_spec_drafted/_spec_interleaved`。与 Step 4 和 5.5 冲突。
  - **#59**：新增 `BranchSpeculativeRunner`。

### 6.4 风险

- **风险-A：CPU 开销。**
  - 现状：PR #57 的 F11 显示，async eager 路径已经受 CPU 限制（GPU 忙碌时间约 32%）；每行多一个对象会继续拉长 host 时间。
  - 缓解：字段按列存放，并使用 `slots`。
  - 验收：用 Ouro-1.4B BF16，在 sync eager、async eager、async + graphs 三种配置下，比较 tokens/s、TTFT、ITL/TPOT 以及每步的 host 时间。`torch.cuda.set_sync_debug_mode("warn")` 统计到的同步次数不能增加。
- **风险-B：顺手把语义统一了。** 包括消费次数的差异、"先乘后判"与"先判后消费"的差异。由 T1 固定。
- **风险-C：测试依赖内部结构。** 涉及 test_pd.py:233-278；test_async_pipeline.py:82, 92, 142, 239, 250；test_cdb_runtime.py:386；以及所有读取 `last_schedule` 的地方。迁移时只改访问路径，不删断言。
- **风险-D：monkeypatch 的调用点。** `execute`、`submit`、`_execute`、`prepare`、`_sample_tensor`、`Submission.ready/collect`、`speculative_runner.execute` 必须仍然是真实的调用点。
- **风险-E：F1 方案的中间窗口。** 从返回决定到执行 finish 之间，请求不在任何队列中，但仍在注册表里。这段时间内若有异常，由两条异常处理路径兜底，T6/T7 覆盖这种情况。
- **风险-F：PD 的重复路径逐渐偏离。** 以 GPU 上的 PD 测试作为门禁，并在 M9 处登记。
- **风险-G：BF16 多 stream 下的数值差异**（docs/cdb_runtime.md:219-231）。正确性门禁用 CPU tiny 模型和 GPU FP32；BF16 只用来看性能和 exit depth。
- **风险-H：GPU 未验证。** CPU 测试无法证明 stream/event 的正确性，Step 2–5 每一步都要跑 G-GPU。

---

## 附录 A：基线上的运行验证

两项验证都在 `main @ d286233` 上用 CPU 完成，未修改任何源码。

**公共设置**

- 模型：先 `torch.manual_seed(123)`，再构造 `OuroForCausalLM(OuroConfig.tiny())`。
- gate：把 `model.model.early_exit_gate.weight` 置零，bias 设为 `logit(0.3)`，这样每轮 hazard 恒为 0.3。
- 配置：`ExitConfig("ouro_delayed")`。sync 用默认的 `ExecutionConfig()`，async 用 `ExecutionConfig(async_scheduling=True)`。

### A.1 delayed 信号的消费次数

- **请求**：prompt `[2, 3]`，`SamplingParams(max_tokens=3, min_loops=1, max_loops=4, exit_threshold=t, ignore_eos=True)`。
- **观察点**：
  - 包装 `scheduler.enqueue`，在已生成 token 的请求进入 CODA 时，记录 `remaining_probability`；
  - 包装 `engine._delayed_signal`，记录每次传入的 `signal_depth`。

| threshold | 路径 | exit_depths | 进入 CODA 时的 `remaining_probability` | 每个 decode token 消费的 `signal_depth` |
|---|---|---|---|---|
| 0.99 | sync | `[4, 4, 4]` | 0.343 | 1, 2, 3 |
| 0.99 | async | `[4, 4, 4]` | 0.49 | 1, 2 |
| 0.5 | sync | `[4, 3, 3]` | 0.49 | 1, 2 |
| 0.5 | async | `[4, 3, 3]` | 0.49 | 1, 2 |

### A.2 preemption 后 pending signal 仍然保留

- **设置**：在公共设置基础上，使用 async，另加 `CacheConfig(64, 2)` 和 `SchedulerConfig(enable_preemption=True)`。
- **请求**：prompt `[1, 2, 3]`，`SamplingParams(max_tokens=4, min_loops=1, exit_threshold=0.5, ignore_eos=True)`。
- **操作**：等请求满足以下三个条件：已经产出至少一个 token、处于 RECURRENT、并且在 `_pending_exit_signals` 中有条目。然后加入一个新请求，清空 `selected_request_ids`，手动调用 `preemption.preempt(新请求)`。
- **结果**：
  - `preempt` 返回 True，快照已保存；
  - `_pending_exit_signals` 中仍有该请求，此时 `loops_done=1`，stage 为 WAITING；
  - 该请求随后 resume 一次；
  - exit_depths 与不做 preemption 的运行一致，都是 `[4, 3, 3, 3]`；
  - 结束时 KV 全部归还。
