# Changelog

本项目采用 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/) 风格，
并遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

1.0 起,对 **STABLE** API 的破坏性变更只能出现在 major 版本,
并且必须在此文件中显式列出受影响的 Public API。

## [Unreleased]

## [6.11.0] — 2026-09-03

### Added

- **同轮并发**（design/86 修订）：同一轮里 ≥2 个 ``async_capable`` 工具调用并发执行
  （``AgentLoopConfig.tool_batch_concurrency``，默认 4；0/1 关）。模型面对「三张图」的直觉是同轮批量
  发调用而不是先起 background_run，之前逐个排队（conv-224 三张图 3 分钟）。不变量：TOOL_BEFORE 仍按
  原顺序先跑完（闸类 hook 语义不变，判 SKIP 的不起任务）；结果按原顺序回填；TOOL_AFTER / 事件 /
  落库串行；非 async_capable 工具永远串行；取消 / HumanInputRequired / TOOL_AFTER BREAK 会取消未完成
  的并发任务。

## [6.10.0] — 2026-09-02

### Added

- **上下文三旋钮解耦**（此前折叠预算借用 ``max_tokens``——那是每次请求的**输出上限**，两个概念混用）：
  - ``context_budget_tokens``：独立的折叠预算（投影前缀估算 token ≥ 它 × trigger_ratio 时折叠）；
    None 回退 ``max_tokens``（兼容）。
  - ``context_checkpoint_tokens``：轮边界按**上一轮真实 prompt_tokens**（= 当前上下文真实大小）判，
    达到 → COMPLETE_DECIDE 收尾窗口 → 新终态 ``context_checkpoint`` → 正常投影 → 宿主续接。
    与 ``max_tokens_per_run``（累计费用，随轮数平方增长）正交。
  - ``insend_distill_tokens`` / ``insend_distill_batch`` / ``insend_distill_hot_tail``：send 内保险丝——
    轮边界按**上一轮真实 prompt_tokens**判，达到阈值就把当前 send **最早的 batch 条**尚未蒸馏的
    工具结果在内存里替换成投影行（同一套 ``ToolDefinition.project`` 蒸馏 + ``recall_send(send_index,
    seq)`` 坐标），最近 ``hot_tail`` 条永不动；下一轮仍超就再蒸馏下一批（逐轮递进）。
    不落盘、不改 pl_messages、不改行数。
- 新终态 ``context_checkpoint``（LoopStatus）；workflow 叶子的 ContinuationPolicy/retry 与
  子代理 LIMIT 事件把它按 ``hit_round_limit`` 同等处理。

## [6.9.0] — 2026-09-02

### Fixed

- **投影模式：中断的 send 现场自愈投影**（真实事故）。投影只在 end-of-send 写入；一个被进程
  重启/崩溃杀在半途的 send 永远没有投影行，装配上下文时走「逐字回退」——它的全部原始行
  （含 tool 协议行）从此逐字带进之后**每一个** send（一个 80 轮 send = 182K 字符 ≈ 115K tokens，
  之后每轮 2–3 分钟，agent 看起来像死了；四个长会话全部如此，`pl_compactions` 一次没触发——
  因为膨胀的不是投影前缀而是这段原始行，折叠触发器量不到它）。现在
  `_run_loop` 装配时对**没有当前版本投影行**的过去 send 调 `_heal_send_projection`：确定性
  `project_send` + 幂等落库（短锁、无 LLM、不触发 fold），再按投影渲染。版本不匹配/未迁移/
  迁移失败同样自愈；逐字回退只剩「投影本身抛异常」一种情形。

### Changed

- 行为变更（测试契约随之更新）：缺投影行/旧版本投影行的过去 send 不再逐字进入上下文。
  内容不丢（蒸馏文本可读、`recall_send` 可回取原文），只是不再以 tool 协议行形态出现。

## [6.8.1] — 2026-09-02

### Added

- **foreach 迭代号进叶子 metadata**（`workflow_iteration`）：body 各迭代共享一个 node_id，
  观测端没有迭代号就把 N 路并发画成同一个节点反复亮灭——被用户读成「循环不是并行」
  （真实反馈）。宿主活动面板据此把 foreach 铺开成逐迭代格子/×N 计数。

## [6.8.0] — 2026-09-02

### Added

- **工具异步化（长工具都是任务，不是调用）**：
  - `ToolDefinition.async_capable: bool = False`——标记无副作用、可安全并发/重跑的长耗时
    工具（生成图像、抓网页）。
  - `background_run` 新增 `action=tool`：`(tool=<name>, args={…})` 把一个 async_capable
    工具作为后台任务在本进程事件循环上跑，立即返回 task_id；结果持久化进同一张
    background 任务表（`command` 以 `tool:` 前缀区分），`action=check` 取结果。
    contextvars 随 task 创建自动拷贝（PEP 567）——宿主的计费/活动上下文天然跟着走。
    并发上限 8/会话；`background_run`/`workflow`/`spawn_agent` 拒绝后台化（防递归）。
  - `register_tool_task_callback(cb)`：后台工具任务完成时回调 `(session_id, task_id,
    status)`，宿主决定要不要唤醒已睡的 agent（在忙的 session 由既有的
    `BackgroundRuntimeProjector` 在下一轮开轮时注入更新，无需回调介入）。
  - `ToolRegistry.to_openai_tools()`：async_capable 工具的描述自动追加「⏳ 可异步」用法
    后缀——仅当 `background_run` 同在工具集里才追加（不教模型调不存在的入口）。

## [6.7.0] — 2026-09-02

### Added

- **`ContinuationPolicy`（耗尽续跑）**：`AgentNode.continuation = {max_continuations, extra_rounds,
  gate}`。叶子以 `hit_round_limit` 落地且门条件成立（缺省 `gate="todo_remaining"`：该叶子
  session 自己的 todo 还有未完成项）时，引擎在**原会话**上 `follow_up` 补 `extra_rounds` 轮
  接着做——不起新会话、不从头重跑；至多续 `max_continuations` 次，全程受 run 共享预算钳制，
  续跑轮的 rounds/usage 折算进节点结果。续命提示自动附上该叶子的剩余 todo 清单。
  executor 可通过可选的 `continue_agent(session_id, input, *, parent_loop, extra_rounds,
  stop_event)` 方法自定义续跑（缺省实现走 `InProcessExecutor`：child-run guards +
  `parent_loop.follow_up`）。
- **`journal.amend_step`**：宿主在 run **终局后**显式修正某节点的记录（status/text/usage，
  带 `amended` 审计标记；run 级 status 不动）。与 `record_step` 互补——那条路对终局 run 冻结。
  用途：宿主对某叶子原会话补轮（nudge）后回写新结果，否则日后 `resume` 会按旧的
  hit_round_limit/failed 记录把该叶子从头重跑，清掉续跑成果。

### Changed

- **`retry.on` 语义拆分**：`failed` 触发器**不再**涵盖 `hit_round_limit`（耗尽不是「没跑好」
  是「没跑完」——新会话重跑只会同预算再耗尽一次）。`RETRY_TRIGGERS` 新增 `hit_round_limit`
  取值，需要旧行为（耗尽也从头重跑）就显式写上它。

### Fixed

- **run 级终态漏计截断叶子**：此前只有 `status=="failed"` 的叶子会让 run 落 failed，
  `hit_round_limit` 的叶子让 run 谎报 completed。现在任何非 completed、非 tolerated
  （continue_on_error）的叶子都会把 run 判为 failed。

## [6.6.1] — 2026-08-31

### Fixed

- **截断提示的位置**：6.6.0 引入的「工具调用 arguments 被截断」提示，被追加在
  `assistant(tool_calls)` 与它的 `tool` 结果**之间**，造出非法历史
  `assistant(tool_calls) → user → tool`。下一次请求供应商直接回 400
  （"An assistant message with 'tool_calls' must be followed by tool messages responding to
  each 'tool_call_id'"）→ 重试耗尽 → 整个 run 降级。
  真实事故：一个会话在**第一次发交互卡片**时就死在这（提示内容是对的，位置错了）。
  现在提示补在**该轮所有 tool 结果之后**。
  同一条不变量在 `TOOL_AFTER` BREAK 分支本来就守着（被跳过的工具也要补 tool 结果），
  6.6.0 在新分支上漏了。新增单测直接钉住**消息序列合法性**本身，而不是「提示在不在」。

## [6.6.0] — 2026-08-30

### 修复

- **被 max_tokens 截断的一轮不再当成「供应商打嗝」原样重试。** 真实事故
  （DeepTalk conv-213，glm-5.3-flash）：模型想在一轮里写一个 25KB 的 CSS 文件，输出打到
  `max_tokens=20000` 被切在工具调用的 JSON 中间 → 解析不出 `tool_calls`、正文也是空的
  （内容全在那段 JSON 里）→ 命中「空响应 = 打嗝」的重试路径 → **原样重试** → 同一个
  prompt、同一个模型、写出同样长的东西、同样被切断。两轮各约 8 分钟、产出为零，
  用户那边看到的是 16 分钟沉默。

  区分两者的信号一直都在：provider 的 `finish_reason`（截断是 `length` / `max_tokens`）。
  现在按它判定（取不到时以 `completion_tokens` 打满 `max_tokens` 兜底），
  处置不是重试而是**改变输入**——把「你上一轮被从中间截断了，把它拆小再来」作为一条
  user 消息落进历史。输入变了，模型才可能给出不一样的输出。最多提示 2 次。

- **截断的第二种表现也说实话**：工具调用在、但它的 `arguments` JSON 断在半路时，
  参数会被降成 `{}`，必填校验于是报「缺参数」——模型据此以为自己忘了填，原样再写一遍、
  再被截断（conv-213 实测：一条 `missing required parameter` 背后是
  `completion_tokens=20000`）。现在同样补一句实话。

  🔴 **这里刻意不做 JSON 修复**：把截断的 `{"path":"a.css","content":"body{co` 补成合法
  JSON，`content` 就是那半个文件——`write_file` 会当成功写下去、agent 继续往前走，
  交付一份残缺的稿子。静默损坏比报错严重得多。（结构化输出那条路的
  `runtime/structured._try_repair_json` 不受影响：补全一个只读的 JSON 结果无害。）

### 内部

- `_sanitize_tool_calls` 返回 `(calls, 有参数解析不了)`。标志走返回值而不是塞进 call 里：
  那些 dict 会原样进 assistant 消息、下一轮发回给供应商，多一个非标准字段可能把请求打挂。


## [6.5.0] — 2026-08-28

### Fixed

* 🔴 **`.webp` 在某些镜像里猜不出 MIME，会拖垮整个会话**。`mimetypes` 读的是系统的
  mime.types，不同镜像装的不一样：宿主 Python 认得 `.webp`，DeepTalk 的生产容器里
  `guess_type(".webp")` 返回 `None` → 落到 `application/octet-stream` → 渲染层判定
  「这不是图片」抛 `ModelCapabilityError` → 重试耗尽 → run 降级终止（真实事故 conv-201：
  agent 刚把配图处理成 webp、正要看一眼，会话就断在那里，用户说「继续」才接上）。
  现在 `_guess_mime_type` 在系统表认不出时查一张**内置**扩展名表兜底
  （webp/avif/heic/heif/png/jpg/gif/bmp/tiff/svg/pdf）。本地全绿、生产炸掉的典型，
  只能靠内置表挡。
* **认不出类型的附件降级，不再抛**。原先 `prepare_attachment` 对非 image/pdf 抛
  `ModelCapabilityError`，理由是「占位读起来像文件已经被读过」——那对含糊的占位成立，对
  现在这句不成立：它明说模型没拿到内容并给出回取坐标。而抛出去的代价是整个会话不可用
  （历史里躺着一个认不出类型的附件，之后每一次 send 都失败）。

* **图片文件读不到时降级，不再抛异常**。`_render_image_attachment` 过去让 `FileNotFoundError`
  穿出渲染层，代价与「模型不支持看图」那一路完全一样：一次 render 同时渲染历史与本轮输入，
  一个读不到的附件会让**每一次** send 都失败，重试耗尽后整个 run 终止。DeepTalk conv-198 真实
  发生——宿主把相对路径交给渲染器，两张刚生成的图按进程 cwd 解析而找不到，agent 就此停在半路，
  图既没进上下文也没发给用户。现在与能力不匹配同款处理：降级成占位文本，**说清模型没有看到**，
  并带上 `ref` 坐标供找回。宿主的路径 bug 该自己修，但渲染器不该把「一个附件读不到」放大成
  「会话不可用」。
* `_downscale_to_data_url` 遇到 `OSError` 不再吞掉后重试读同一个文件——那会打出
  "could not downscale …; sending it at original size" 这句**误导性**日志（毛病在路径，不在
  图像解码），现在直接上抛交由降级处理。

## [6.4.0] — 2026-08-28

### Added

* **`queue_images_for_next_round()`：一批图入队成 ONE user 消息**（一个说明 + 每张一个
  attachment 块）。逐张调 `queue_image_for_next_round` 会产生 N 条独立 user 轮次、**每条都
  重复一遍同样的说明**——真实会话里「三张截图配一个问题」就在 transcript 里留下三份那个问题。
  一批一条也正是 provider API 期望的形状。


## [6.3.0] — 2026-08-28

### Changed

* **图片注入默认改为 DURABLE**（落库成真实 ``user`` 行），`queue_image_for_next_round(durable=False)`
  保留原来的「只活一轮」语义。默认之所以反转：第一版选 ephemeral 是怕「回取三次就永久带三张图」，
  实测把这个顾虑推翻了——provider 的 prefix cache 对稳定前缀命中率约 99%，图待在前缀里每轮只花
  约十分之一价；而语义上 durable 明显更自然：看完一张 UI 图要基于它写十几轮代码，图该一直在
  眼前，而不是看一眼就消失、想再看得重新调一次。跨 send 由投影蒸馏成
  ``[image: shot.png · file_uuid=…]``，不会无界累积。

  ``drain_queued_images()`` 相应返回 ``(durable, ephemeral)`` 两组（**签名变更**，但这是 6.1.0
  才引入的内部面，未进 ``STABLE_API``）。注入位置在 hook 自己的 ``persist_messages`` **之后**——
  一张图应该落在宣告它的那条工具结果下面，这正是模型预期它出现的地方。


## [6.2.0] — 2026-08-28

### Added

* **`LlmCallCompletedPayload` 带上 prompt-cache 拆分**（`prompt_cached_tokens` /
  `prompt_cache_miss_tokens`）。transport 早就把这两个数从 provider 响应里解析进
  `LLMTokenUsage` 了，但**逐次调用的事件把它们丢掉了**：宿主只能看到「这一轮花了 44k prompt
  token」，**无从判断**那是 44k 全价、还是 99% 命中缓存只按十分之一计。
  没有这个拆分，任何关于上下文成本的判断都是猜——它决定了「精简历史」是一次大胜还是 20 倍的
  亏损（改动历史中段会让该点之后的缓存全部失效）。累计口径的 `USAGE_UPDATED.usage` 早有
  `cache_read_tokens`；这里补的是**逐轮精度**，用来定位是哪一轮击穿了缓存。
  `None` 表示 provider 没报（与真实的 0 区分开）。


## [6.1.1] — 2026-08-28

### Fixed

* 投影/重放读取行 ``meta`` 时改用 ``getattr(..., None)``。``Representation`` 是公开 seam——
  宿主与测试可以喂任何行对象进来，缺一个属性不该让整个 send 的投影炸掉（6.1.0 引入 meta 读取
  时漏了这层防御）。


## [6.1.0] — 2026-08-28

### Fixed

* 🔴 **换到看不了图的模型会让整个会话崩掉。** 一次 render 同时渲染**历史**与本轮输入，所以
  6.0.0 那条「未声明发图直接抛」在 `VerbatimRepresentation` 下会被历史里的图触发：定义换模型
  后，每一个 send 都抛 `ModelCapabilityError`，会话彻底不可用。而历史是既成事实，不是调用方
  的错。现在渲染层**降级**而不抛。

  降级不等于回到 5.x 那个静默毛病——区别全在文案。旧实现塞的是一句含糊的 "The current model
  does not support image input"，混在附件描述里，模型照样按「我看过这张图」的语气编答案。新的
  占位必须做到两件事：**说清模型没有看到**，并**给出把图找回来的坐标**。
  想要「发图给看不了图的模型就报错」的调用方，用 `capabilities.require_image_input()` 自查。

### Added

* **`AttachmentRef.ref`：宿主给的回取坐标**，由 `create_attachment_ref(path, ref=…)` 传入，
  跟着这张图走进**蒸馏行**（`[image: shot.png · file_uuid=…]`）与**降级占位**。换到看不了图的
  模型之后，这行文本是模型唯一能据以找回原图的东西——DeepTalk 放 `file_uuid=…`，可直接喂给
  see_image。`recall_send` 与 `queue_image_for_next_round(ref=…)` 全程携带。


## [6.0.0] — 2026-08-28

### Added

* **`ModelCapabilities.max_image_edge`：图片长边上限，在渲染的唯一汇点生效。** 因此「新发的图」
  「recall 回取放回眼前的图」「宿主自己塞的图」一律受限——不靠每条入口各自记得裁一次。
  已在限内的文件**原样直通**（不解码、不重编码），所以常见情况零开销。Pillow 是可选依赖
  （`power-loop[images]`，已并入 `[all]`）；缺席时按原尺寸发送并告警一次，而不是把图丢掉。
  为什么是这个旋钮：图片按**像素**计费而非字节——787KB 的噪点图与 1.8KB 的同尺寸纯色图 token
  完全相同（实测），所以降 JPEG 质量一个 token 都不省，缩边长省 43%。

* **按需图片回取**：`recall_send(send_index, seq=…)` 命中一行含图片的记录时，把图放回模型眼前
  **一轮**（`runtime/image_recall.py`）。图不能从工具返回——OpenAI 兼容协议的 `tool` 消息
  是纯文本类型——所以它作为独立 user 消息进入本轮请求，工具返回值只说明「图已放到你眼前」。
  刻意是 **ephemeral**：只进请求、不落库、不进投影，否则回取三次就永久携带三张图，正是投影
  要避免的无界增长。

* 顶层导出 `create_attachment_ref()`：构造多模态输入的必需件，宿主不该为此 import `_vendor`
  内部路径。（未进 `STABLE_API`——多模态输入面仍在演进。）

### Changed — BREAKING

* **模型能力改为「声明」，不再从模型名推断。** 原先 `resolve_model_capabilities()` 用一张约 15 条
  厂商正则的表按**模型名**猜 `supports_image_input` 等能力，猜不中就判定为不支持，并**静默**把图片
  换成一句「当前模型不支持图片输入」再照常发出去。真实后果：`deepseek-v4-flash-vision-exp`（一个
  实测能吃 `image_url` 的模型）不在表里，发给它的每张图都被悄悄丢掉，模型照样给出通顺答案，调用方
  完全看不出这个答案是在没看见图的情况下编的——绿灯罩着一个从未发生过的能力。

  现在：能力是 `LLMProviderConfig(capabilities={"supports_image_input": True})` 上的**配置**，
  三态（`True` 支持 / `False` 明确不支持 / `None` 未声明，且 `None` **不等于** `False`）。
  未声明或声明为不支持时发图 → 抛 `ModelCapabilityError`，绝不降级。

  **受影响的 Public API：**
  - 删除 `resolve_model_capabilities()`、`capability_overrides_from_env()`、`PROVIDER_DEFAULTS`、
    `MODEL_PATTERNS`、`CAPABILITY_OVERRIDE_ENV_MAP`。
  - 删除环境变量 `POWER_LOOP_SUPPORTS_*` / `OPENAI_COMPAT_SUPPORTS_*`（进程级作用域无法表达
    「同一进程里这个定义的模型能看图、那个不能」，而这正是多 agent 宿主的常态）。
  - `LLMProviderConfig.capability_overrides` → `LLMProviderConfig.capabilities`；
    `OpenAICompatibleChatConfig.capability_overrides` → `.capabilities`。
  - `ModelCapabilities` 只保留 `model` 与 `supports_image_input`。删除 `provider` / `api_family` /
    `supports_tools` / `supports_stream` / `supports_data_url` / `supports_pdf_input_chat` /
    `supports_pdf_input_responses`——它们**没有任何代码读取**，是纯装饰。传入这些键现在直接 `ValueError`，
    以免一份配置声称拥有一个永远不会被兑现的能力。
  - 新增导出 `ModelCapabilities` / `ModelCapabilityError`。

* **`LLMRequest.to_messages()` 不传 capabilities 不再等于「跳过渲染」。** 以前 `capabilities=None`
  会整个跳过多模态渲染，把 `{"type": "attachment"}` 原样塞进 provider 请求体；现在等同「什么都没声明」，
  同样抛 `ModelCapabilityError`。

* **PDF 一律走文本抽取。** 任何 transport 都没有实现原生 PDF 传输，所以 `supports_pdf_input_*` 是一个
  没人兑现的承诺，直接删掉而不是留着当摆设。文本 PDF 抽取是忠实路径（内容确实到达模型），不抛异常；
  **抽不出文本**的 PDF（扫描件 / 纯图导出 / 加密）改为抛异常——塞一句「未能读取内容」给模型，会复现
  刚刚被消灭的那种「无中生有的答案」。不支持的附件类型同理，不再降级成占位文字。

### Fixed

* **`recall_send` 会把序列化的多模态记录原样吐给模型。** 它直接返回 `content` 文本列，而多模态
  行存的是 JSON——内联 data URL 的话就是整个 base64。现在文本侧走与投影同一个蒸馏
  （`[image: shot.png]`），图片侧改为放回眼前（见 Added）。纯文本记录不受影响。

* **steering 会丢掉图片。** 进程内 follow-up 队列保留的是原始对象，但 `merge_follow_up_inputs`
  用 `json.dumps` 把 content 压成一坨文本——图片只以「序列化后的块」形式活下来，模型看不见。
  同一张用户照片，会话恰好空闲时看得见、恰好在忙时看不见。现在文本合进 `<follow_up>` 信封，
  **非文本块（图片）作为独立内容块带过去**。跨进程队列是 TEXT 列、图片本就无法穿越，但
  `follow_up_text()` 也不再把整个 data URL 粘进去（改为 `[image_url]` 标记）。

* **verbatim 重放不还原结构化内容（多模态静默失效）。** `runtime/representation.py` 的
  `_row_to_loop_dict` docstring 声称 "mirrors `stateful_loop._row_to_loop_message`"，却恰恰没有镜像
  其中的 JSON 还原那段——`CONTENT_ENCODING_META_KEY` 在整个 `representation.py` 里一次都没出现过。
  于是 `VerbatimRepresentation` 模式下，一条多模态消息重放时 content 是**字面 JSON 字符串**而不是
  数组：模型收到一段 prose，图片彻底失效，全程无报错。
  常量与新的 `decode_row_content()` 移到 `runtime/store/types.py`（中立位置，避免 runtime→agent
  循环导入），写入侧与读取侧共用同一份定义。`power_loop.agent.sink` 继续 re-export，下游 import 不变。

* **投影会把内联 base64 当文本逐字保留。** `project_send` 对 send 的输入侧刻意 verbatim（注释理由：
  "it is short relative to tool output"）——对文本成立，对 `data:` URL 崩塌：图片变成不可读的文本，
  却在**之后每一个 send** 上被重复计费。新增 `distill_multimodal_text()` 把每个内容块蒸馏为一行引用
  （`attachment` → `[image: shot.png]`，保留可回取的文件名；内联 data URL → `[image]`），并以
  `_strip_data_urls()` 兜底：无论哪条路径（含未打编码标记的老行）塞进来的 data URL 都不会进投影。
  send 输入与 send 中途注入（`__user__`）两个入口都已接上。

* **Anthropic transport 会丢弃 `attachment` 块。** 该 transport 自己翻译消息、不走
  `LLMRequest.to_messages()`，因此从未执行过多模态渲染：附件块原封不动到达 `_non_text_blocks`，
  被当作 `[unsupported content block dropped: attachment]` 扔掉。现在它先跑同一套渲染与能力判定，
  再把 `image_url` 翻成原生 Anthropic image block。`AnthropicChatConfig` 随之新增 `capabilities`
  字段（此前它根本收不到任何能力信息）。

## [5.4.0] — 2026-08-27

### Added

* 投影条目带**行坐标**：`ProjectedRepresentation.project_send` 给每个工具条目记 `seq`（结果行的
  pl_messages seq）与 `call_seq`（发起调用的 assistant 行），mid-send `__user__` 注入也带 `seq`。
  宿主渲染时每行可以自带 `recall_send(send_index=N, seq=S)` 坐标——模型不必跨 send 找。
* `recall_send(send_index, seq=None)`：带 `seq` 时只回**那一行原文**（上限 `RECALL_SEND_ROW_CHARS`
  = 40000 字符，超出部分计数），并附上发起它的 assistant 调用；不带 seq 仍是整 send 列表
  （每行 2000 字）。起因：整 send 回取对 155 行的 send 是几十万字符，而一份 14KB 的技能文件
  被截到 2000 字——「回取原文」名不副实。

## [5.3.0] — 2026-08-27

### Added

* `tools.command_policy`：`bash` / `background_run` 之上的**类目级命令策略**（规则层）。把命令按
  `package_install` / `download` / `pipe_to_shell` / `daemon` 分类，宿主通过
  `RuntimeEnv.blocked_command_categories` 决定哪些类目被拒；`pipe_to_shell`（`curl … | sh`）永远拒。
  库默认只拒 `pipe_to_shell`，向后兼容。拒绝是**工具结果**（带出口文案），不是异常。
  起因：DeepTalk 会话 194 的设计师 agent 在沙箱里 `npm i playwright-core` + 下载 chromium 绕了 20 分钟
  ——沙箱本身没被打穿，但"装软件"应当是运营授予的能力而不是 egress 白名单的副作用。
  `RuntimeEnv` 新字段 `blocked_command_categories: frozenset[str]`（默认空）。
* `OpenAICompatibleChatConfig.request_extra`：配置级请求参数，合并到每次请求的 `extra` 之下（请求优先，
  `extra_body` 逐键合并）；`LLMProviderConfig.extra["request_extra"]` 直通。用途：宿主按模型开
  DashScope `extra_body.enable_thinking` 这类开关，而不必碰 pipeline 的每次 LLMRequest。

## [5.2.2] — 2026-08-19

### Fixed

* 带工具调用的回合现在也写 `usage_rounds` 逐轮明细行。此前 `sink.on_round_ended(usage=…)`
  只在无工具回合的收尾路径上发——agent 会话里几乎每轮都带工具，于是一个 34 轮的叶子在表里
  只剩最后一轮那一行（2026-08-19 线上实锤）。**总量从来没错**（`session_stats` 用 send 结束时
  的内存聚合 bump），丢的是这张表存在的唯一意义：逐轮成本明细与 `prune_usage_rounds` 的保留语义。

## [5.2.1] — 2026-08-15

### Added

* `RetryPolicy.backoff_factor`（默认 1.0 = 固定间隔）。第 N 次重试等
  `backoff_s * backoff_factor**(N-1)`，单次等待封顶 `MAX_BACKOFF_S`（60s）——provider 限流时
  才真正需要指数退避；等太久等于把整个 run 挂在那儿。

### Fixed

* `foreach` 现在把当前迭代序号放进 body 的 env（保留键 `ITERATION_ENV_KEY = "__iteration__"`），
  宿主的 `WorkflowFileIO.output_path(..., iteration=)` 才拿得到它。**不修的话 N 个并发迭代会
  追加进同一个产出文件，互相搅乱**——5.2.0 的文件产出端口对 foreach 实际上是坏的。

## [5.2.0] — 2026-08-14

### Added

* **叶子级错误语义**（`AgentNode`）。容器节点的 `on_error` 管的是「兄弟分支要不要被取消」，
  这三个管的是「这个叶子失败了怎么办」——两者正交：
  * `retry: {max_attempts, on: [failed|empty], backoff_s}` —— 重试**起新的叶子会话**（失败会话
    的上下文可能已被半截的工具调用污染）。`idempotency_key`（`run_id:node_id`）跨 attempt
    **保持不变**（变了工具就没法去重），attempt 号单独进 leaf metadata 供工具区分第几次。
    `max_attempts` 上限 5（重试烧的是真钱，无上限是个跑飞入口）；取消与预算耗尽不触发重试。
  * `continue_on_error: bool` —— 本叶子失败不算 run 失败。
  * `fallback: <node>` —— 所有 attempt 用尽后跑的替代节点；成功则**顶替**主节点的结果供下游
    `inputs_from` 引用（下游引用的是主节点 id）。兜底节点不得再带 `fallback`，解析期拒绝。
* **叶子文件产出端口** `WorkflowFileIO`（`output_path` / `render_ref` / `before_attempt`）+
  `AgentNode.output_file`。引擎自己不碰文件系统：它只决定「哪个节点有产出文件」「输入里的引用
  占位符在哪」，读写与文案渲染由 host 实现。输入里的 `@@FILEREF:<path>[<slice>]@@`
  （`FILEREF_RE`）在派发前就地渲染；重试前调 `before_attempt`，host 可归档上一次的产出，
  免得重跑内容追加到旧证据后面分不清哪次。

### Changed

* **BREAKING（行为）**：任一叶子 `failed` 现在会让 run 终态变成 `failed`。此前只要没抛异常
  就报 `completed`——哪怕汇总节点整个失败了。要保留旧的容忍行为，给该节点加
  `continue_on_error: true`。（叶子上限触顶时仍以 "leaf ceiling" 作为根因诊断。）
* 上游节点失败时，下游 `inputs_from` 拿到的不再是空字符串，而是一段**显式的失败说明**——
  空字符串会让下游模型把「没说话」读成「没意见」，进而在缺证据的情况下下结论。

### Notes

* 新字段全部参与 `to_dict()` 序列化：resume 从 journal 的 spec 重建，不序列化就等于恢复后
  悄悄丢掉重试/兜底/产出文件配置。


## [5.1.0] — 2026-08-11

**Minor：`send()` 支持 per-send 的 `response_format`（结构化输出不再只能在构造期定死）。**

### Added
- **`StatefulAgentLoop.send(..., response_format=...)`** —— OpenAI 兼容的结构化输出规格
  （`{"type": "json_object"}`，或 `StructuredOutputSpec.to_openai_response_format()` 产出的
  json_schema），**只作用于这一次 run**。

  为什么必须是 per-send：`response_format` 原本只有 `AgentLoopConfig` 一条路（构造期），而
  一个 loop 常常被**多个调用方共用** —— 宿主按 agent 定义缓存一个 loop，很多会话同时跑在
  上面。想要 JSON 的那一次 send 如果去改 loop 的 config，会把所有并发的 send 一起翻进 JSON
  模式。实现沿用既有的 per-call 覆盖机制（和 `system_prompt` / `max_rounds` 同一条路：
  `dataclasses.replace` 出一份 per-run config，绝不改动 `self.config`）。

  不传时行为完全不变：配置里的 `response_format` 照常生效。

## [5.0.1] — 2026-07-24

**Patch：workflow 叶子会话不再随 run 结束被抹掉（审计留档）。**

### Fixed
- **`WorkflowEngine.run()` 的收尾不再级联删除 LINKED 叶子会话**：原实现在 finally 里
  `close_session(driver, cascade=True)`，把每个角色叶子的逐工具 transcript 连同 driver
  一起删掉——run 一结束，叶子做过什么就无从审计（真实的对抗验证 run 因此不可回放）。
  现在只关 driver（`cascade=False`），LINKED 叶子存活并被重新挂到 NULL 父节点，
  按 metadata（`workflow_run_id` / `workflow_node_id` / `spec_name`）仍可精确定位。
  `close_driver=False` 语义不变（driver 也保留）。
- **`close_session(cascade=False)` 语义补齐**（此前无人使用）：只删指定会话，
  所有直接子会话（不限 lifecycle）重新挂到 NULL——不再留悬空 parent 引用。
  async store 与 legacy 同步 store 行为一致。


## [5.0.0] — 2026-07-23

**Major：黑板工具合并（破坏性）。** 与 4.0.0 的内置工具合并同一思路的收尾。

### Changed（破坏性）
- **`board_read` / `board_post` / `board_update` / `board_remove` 合并为单个 `board` 工具**：
  `board(action=post|read|update|remove, text?, kind?, status?, entry_id?)`。
  `register_blackboard_tools(registry, kinds=..., statuses=..., default_kind=..., overwrite=...)`
  签名不变，但现在只注册一个 `board` 工具；按旧四名注册/勾选/过滤的宿主需同步
  （DeepTalk 侧：loop_cache 勾选名、board 注入头文案、admin_tool_catalog、投影蒸馏已同步）。

### Migration
- 工具调用侧：`board_post(text=...)` → `board(action="post", text=...)`，其余同理。
- 宿主按名过滤/取消注册 `board_*` 的逻辑改为单名 `board`。

## [4.0.0] — 2026-07-23

**Major：内置工具合并（破坏性）。** 默认工具集的工具名/参数契约变化；未升级的 `include=[...]`
列表会在 `get_tool_definitions` 处直接 `ValueError`，宿主需同步更新（DeepTalk 侧：agent 定义
include、admin_tool_catalog、投影蒸馏、文档一并更新）。

### Changed（破坏性）
- **`background_run` 吸收 `check_background`**：`background_run(action="run", command=...)` 启动，
  `background_run(action="check", task_id?)` 查询/列表。`check_background` 工具名删除。
  `DEFAULT_REQUIRED_PARAMS["background_run"]` 由 `("command",)` 改为 `("action",)`。
- **`schedule_wakeup` 吸收 `list_wakeups` / `cancel_wakeup`**：
  `schedule_wakeup(action="schedule"|"list"|"cancel", delay_seconds?, note?, every_seconds?, timer_id?)`。
  `list_wakeups`、`cancel_wakeup` 工具名删除。
- **`run_agent` meta-tool 删除，仅保留 `spawn_agent`**：`spawn_agent(task, name?, system_prompt?,
  tools?, max_rounds?)` —— 原 run_agent 的唯一实际增量（自定义 system_prompt）并入 spawn_agent；
  完整声明式 spec 场景请在宿主代码直接调 `run_agent_spec()`（Python API 不变）。
  受影响 Public API：`RUN_AGENT_DEFINITION` 符号删除；`register_spawn_agent` 移除
  `include_run_agent` 参数（现在只注册 spawn_agent 一个工具）。
- `python -m power_loop` 内置 system prompt 的 tool_guide、各工具确认/错误文案中的旧工具名
  同步为新调用形态。

### Migration
- `include`/`exclude` 列表：`check_background`→删；`list_wakeups`/`cancel_wakeup`→删
  （功能并入 `background_run` / `schedule_wakeup`）；`run_agent`→删（用 `spawn_agent`）。
- 工具调用侧：给 `background_run`、`schedule_wakeup` 补 `action` 参数。

## [3.24.0] — 2026-07-23

Feature release。Tool catalog UI now supports category fold/select-all, and the built-in note tools were consolidated
into a single parameterized `note` tool. No `STABLE` API break.

### Added
- Tool catalog categories can be collapsed and selected in one shot.
- `note_add` / `note_update` / `note_delete` / `note_list` were merged into `note(action=...)`.

## [3.23.0] — 2026-07-22

Additive, backward-compatible (minor)。无 schema 变更,无 STABLE API 破坏(AgentLoopConfig 新增带默认值字段)。

### Added
- **`AgentLoopConfig.max_context_rows`（默认 300,投影模式）——最近行数上下文上限。** 投影模式的
  history 原本从 fold 压缩点一路铺到最新消息;fold 迟迟不触发或 send 细碎众多时无上界。现在组装
  history 时最多保留 N 行:压缩摘要行(如有)**始终保留**在最前,当前 in-flight send 始终完整保留,
  更早的内容从最旧端**整块**丢弃(先 legacy 前缀,再整个旧 send)直到装下——绝不切开一个块,
  verbatim-fallback send 的 tool 协议对不会被拆散。`None`/`<=0` 关闭。verbatim 模式不受影响
  (其窗口由就地 compactor 约束)。

## [3.22.0] — 2026-07-21

Additive, backward-compatible (minor)。无 schema 变更,无 STABLE API 破坏(校验放宽,原有 spec 仍合法)。

### Changed
- **`AgentSpec.max_rounds` 去掉 50 的上限**:校验由 `1 <= max_rounds <= 50` 放宽为 `>= 1`(下限保留,
  0/负仍拒)。子 agent / workflow 叶子可申请超过 50 轮。`spawn_agent` 工具的 max_rounds 描述同步(min 1)。

## [3.21.0] — 2026-07-21

Additive, backward-compatible (minor)。无 schema 变更，无 STABLE API 破坏
（`run_detached` / `spawn_background` / `resume_detached` / `Workflow.start` 均非 STABLE）。

### Added
- **detached workflow 的完成唤醒可插拔:`on_complete` host seam。** `run_detached` /
  `resume_detached` / `spawn_background` / `Workflow.start(detached=True)` 新增 kw-only
  `on_complete: Callable[[WorkflowCompletion], Awaitable[None]] | None = None`。**提供时它接管父
  唤醒**——内置的 durable timer 与 `eager_wake` 快路**双双跳过**,转而把一个新的 `WorkflowCompletion`
  (`parent_session_id` / `run_id` / `status` / `note`)交给回调。用于**进程内、无定时器**的唤醒
  (host 通常据此重解析父的当前 loop 再 `follow_up`),与进程共存亡:回调抛错只记 `warning` 并吞掉
  (run 已 journal 定稿,丢失的唤醒由调用方重发),**绝不**回退到定时器、**绝不**让后台任务崩溃。
  `on_complete=None`(默认)行为完全不变。新导出 `WorkflowCompletion`
  (`power_loop.workflow.WorkflowCompletion`)。

## [3.20.0] — 2026-07-21

Behavior fix, backward-compatible (minor)。无 schema 变更，无 STABLE API 变更。

### Fixed
- **空 LLM 响应不再被当作「完成」（严重）**：一轮 LLM 返回**既无文本又无工具调用**时，
  `AgentPipeline.run()` 过去直接 `_finalize("completed", final_text="")` —— 把一次 provider
  空返回（实测 deepseek 偶发）当成了「我做完了」，整个 send 的成果塌成空白终稿。现在纯空返回
  被视为 provider 抖动：有界重试（`_EMPTY_RESPONSE_MAX_RETRIES=3`，每次重试推进 round，故仍受
  `max_rounds` 封顶），重试期间打 `logger.warning`；连续空超过上限才收尾（final_text 仍为空，
  但此时是「重试耗尽后的显式放弃」而非「首次空即静默完成」）。有文本或有工具调用的轮次一律
  清零 streak，正常完成路径完全不受影响。与子 agent 层「final 空时从 transcript 捞回最后非空
  assistant 文本」的 fallback 串联：pipeline 层先治抖动，子 agent 层再兜底。

## [3.19.0] — 2026-07-19

Fix + additive API, backward-compatible (minor)。**含 store schema v6 → v7 迁移**（新增两张表，
无数据变更；AUTO_CREATE 自动应用）。

### Fixed
- **每 session 的锁不再依赖 loop 对象身份（严重）**：`_locks` / `_follow_up_queues` /
  `_follow_up_queue_locks` 三个 **实例** 字典改为按 `session_id` 键控的**进程级**注册表。
  旧实现下，同一个 `session_id` 被两个 `StatefulAgentLoop` 对象驱动时会拿到**两把不同的锁**，
  互斥完全失效——而宿主替换缓存中的 loop（例如配置编辑触发重建）是完全合法的操作，从库这边
  不可见。「发往同一 session 的 send 串行执行」是 API 承诺，不应取决于宿主持有几个对象。

  折叠队列必须一同全局化：只全局化锁会让第二个 loop 把 steering 塞进一个**没有任何人排水**
  的队列，把并发损坏换成消息静默丢失。

  实测影响（DeepTalk 会话 119）：两个 loop 在同一 workspace 上交替编辑同一批文件约 6 分钟、
  互相撤销对方的修改；`last_dispatched_seq` 由后完成者写入，导致投影的 send 区间重叠、
  历史被重复灌进后续 send 的上下文。

### Added
- **跨进程 session 租约（`distributed_sessions`，默认关闭）**：进程内的锁只能串行化**一个**
  解释器。多个进程共享同一个 store 时，`{prefix}session_leases` 行成为共同裁判：`send` 先取
  租约、运行期间按 TTL/3 后台续约、结束释放；抢不到则抛 `SessionBusy`。
  - 输给竞争的一方不会丢消息也不会另起一轮：`follow_up` 把 steering 存进
    `{prefix}follow_up_queue`，由持有者在下一个 round 边界排水——**折叠语义跨进程保留**。
  - 崩溃的持有者停止续约，租约到期后可被接管；`fence` 列单调递增，为将来的写路径围栏预留。
  - 默认关闭：单进程宿主得不到收益，却要为每次 send 多付一次租约写、每轮一次续约，并引入
    「持有者卡顿导致租约过期」这一新失败模式。仅服务端 store 有意义（SQLite 本就是单机）。
  - 新配置：`AgentLoopConfig.distributed_sessions`、`session_lease_ttl_s`（默认 90s）。
  - 新 STABLE 符号：`SessionBusy`。
  - 新 store API：`acquire_session_lease` / `renew_session_lease` / `release_session_lease` /
    `session_lease_holder` / `enqueue_follow_up` / `drain_follow_up_queue` /
    `pending_follow_up_depth`。

### Notes
- 排水的「领取即删除」是方言接缝（`Dialect.claim_follow_ups`）。首版实现是 `SELECT` 后
  `DELETE`，在 SQLite 上全绿，但在 PostgreSQL 的 READ COMMITTED 下两个并发排水会读到同一批行、
  各自把同一条用户消息投给自己的模型。SQLite 的串行写掩盖了它——因此新增
  `tests/unit/test_session_leases_pg.py`，这类性质必须在服务端后端上验证。

## [3.18.0] — 2026-07-17

Fix + additive API, backward-compatible (minor).

### Fixed
- **hook BREAK 硬停不再悬空 follow-up steering**：round 边界的 follow-up 队列排水从
  ROUND_START hooks **之后**挪到**之前**。旧顺序下，任何 ROUND_START hook 以 BREAK 收尾
  （宿主典型场景：pass_turn 硬停）会绕过排水，把上一轮期间折叠进来的 steering 静默丢在
  进程内队列里（DeepTalk 会话 117 事故：用户卡片提交被吞、agent 永久沉默）。新顺序下
  BREAK 判定发生在排水之后，排进来的消息已入 history。

### Added
- **`RoundStartCtx.drained_follow_ups`**（additive 字段）：本轮边界排入的 steering 条数。
  break-deciding hook（如宿主的 pass_turn 硬停）可据此发现"沉默决定已过期"，撤销 BREAK。
- **`StatefulAgentLoop.pending_follow_up_count(session_id)`**：空闲会话上仍滞留的
  follow-up 条数（终态微窗口内被接受的 steering）。
- **`StatefulAgentLoop.flush_follow_ups(session_id, ...)`**：把滞留在空闲会话上的
  steering 合并成一条 `<follow_up>` 消息并以 `send` 跑掉；队列空或会话忙时返回 `None`。
  宿主在 owner run 返回后循环调用直至 `None`，即可关闭"接受 vs 终态"竞态窗口。

## [3.17.1] — 2026-07-17

Fix, backward-compatible (patch).

### Fixed
- **token 预算耗尽现在进入 `COMPLETE_DECIDE` 收尾边界**：一次 run 在干净轮边界超过
  `max_tokens_per_run` 时，先以 `reason="budget_exceeded"` 咨询 hook；无注入时仍原样返回
  `budget_exceeded`，有注入时则把 durable user 收尾提示放进同一 send，并允许其在正常 token
  预算已经耗尽后运行明确授予的 `extra_rounds`。该收尾窗口是精确、有限的，不会继承主循环尚未
  使用的轮数。`max_tokens_per_run=None` 或 `<=0` 统一表示不限制。

## [3.17.0] — 2026-07-15

Feature + schema migration, backward-compatible (minor; the store migrates itself on open).

### Fixed
- **`usage_rounds` 逐轮用量不再被后续 send 覆盖**：主键
  `(session_id, round_index)` → `(session_id, send_index, round_index)`。
  `round_index` 每个 send 从 0 重置,旧键使得同一会话每次新 send 都覆盖上一次 send
  的逐轮明细——账面只剩最后一次 send。存量行回填 `send_index=0`(历史明细已不可恢复)。
  异步 store schema v5→v6(SQLite 重建表;PostgreSQL/MySQL 原地换 PK,PG 语句已在
  生产克隆表实测);旧版同步 SQLite store `user_version` 1→2 同款重建。
  `record_usage(...)` 新增可选 `send_index`(缺省 None→0,旧调用方为兼容语义);
  `SQLiteSink.send_index` 属性由 `StatefulAgentLoop` 在 send/resume/submit_input
  的收敛点自动盖章——托管用法无需任何改动即获得按 send 的逐轮明细。

### Added
- **detached workflow 实时心跳（`live` / `live_nodes`）**：单叶 detached 运行期间
  `workflow_status` 此前只有 `running` + 空 `steps`(steps 只在节点**完成**时落账),
  一次 2 小时的真实走查被主 agent 和用户一致误判为卡死。现在:引擎新增
  `on_node_start` 观察者(仅真实执行触发;replay/预算拒绝不触发),detached runner
  将其journal 为 `live[node_id]=started_at_ms`,节点落账时清除;`get_workflow` 对
  running 运行返回 `elapsed_s` + `live_nodes[{node_id, running_for_s}]` 与一行
  「有 live_nodes 且时长在涨 = 在干活」的说明。resume(同步/detached)两路同样接线。
  `WorkflowEngine` 新增可选构造参数 `on_node_start`(默认 None,行为不变)。

## [3.16.1] — 2026-07-11

Fix, backward-compatible (patch — validation now rejects specs that were silently broken).

### Fixed
- **`inputs_from` 引用不存在的节点现在会被校验拒绝**（与 `items_from` / `branch.on`
  的存在性检查对齐）。此前这类引用通过校验、运行时被引擎静默丢弃（只拼接
  `_results` 里存在的 ref）——节点在缺上下文的情况下照跑，无人知晓。由
  `validate_workflow` 的真机冒烟发现。

## [3.16.0] — 2026-07-11

Feature, backward-compatible (minor).

### Added
- **`validate_workflow` 工具（dry-run 校验，不执行）**：与 `create_workflow` 完全相同的
  检查链——严格解析/校验 + 宿主 `spec_transform` 策略钳（有装则跑）——但零执行、零会话；
  INVALID 时一次性列出全部问题（含 `INVALID (platform policy)` 前缀区分平台拒绝），
  VALID 时返回结构摘要（节点形状统计、叶子 id、声明了结构化输出的节点、spec 预算）。
  「validate 通过 = create 必收」是保证（运行期失败另论）。`register_workflow_tools`
  三件套捆绑注册；`create_workflow` 描述补一句「大/detached spec 先 dry-run」。
  导出 `VALIDATE_WORKFLOW_DEFINITION`。

## [3.15.0] — 2026-07-11

Feature, backward-compatible (minor).

### Changed
- **`create_workflow` 默认描述重写为完整的模型作者手册**：五种节点完整语法、
  **AgentSpec 字段表（由 `dataclasses.fields(AgentSpec)` 派生渲染）**、数据流规则
  （items_from/branch.on 必须配 output_schema、id 全局唯一、不可引用并行兄弟 /
  foreach body 内节点）、detached 唤醒协议、完整示例、聚合报错提示。原一行压缩
  语法与「校验为 LLM 作者设计」的哲学不符——校验为模型优化了，说明书却没有。
  防漂移守卫测试（`test_workflow_tool_description.py`）：`_AGENT_SPEC_FIELD_DOCS`
  必须与 AgentSpec 字段集精确相等、描述必须覆盖每种节点类型与关键规则 token ——
  加字段/节点忘改说明书直接红。`workflow_status` 描述补「被唤醒后先调它」。

### Added
- **`register_workflow_tools(description_suffix=...)`**：追加到 `create_workflow`
  模型描述尾部的宿主段。默认描述只陈述 power-loop 自己的事实；宿主的
  ``spec_transform`` 策略上限、自有唤醒协议措辞属于宿主事实，从这里注入
  （宿主删码指引：不再需要整段覆盖描述——DeepTalk 从 78 行全量覆盖缩成一段后缀）。

## [3.14.0] — 2026-07-11

Feature, backward-compatible (minor). 「宿主接缝」主题：五条 host seams（设计见
`docs/design/host-seams-3.14.md`），全部可加性——不带新参数时行为不变。新符号以
**PROVISIONAL** 发布（不进 `STABLE_API`），稳定一个迭代后再提升。

### Added
- **S1 — child-run guards**（`StatefulAgentLoop.register_child_run_guard` /
  `remove_child_run_guard`，类型别名 `ChildRunGuard`）：宿主注册的 context-manager 工厂，
  由 `run_agent_spec` 在**每次内联子运行**（spawn_agent / run_agent 委派、in-process
  workflow 叶子）外围按注册序进入、逆序退出（异常亦然），并**传导给子 loop**（孙辈同样进入）。
  用途：子运行与父共享 hooks 对象和 task-local 状态，宿主的 per-send hook 状态（提醒计数、
  turn 标志、同 send finalize 认领）在子运行期间挂起、结束后恢复。
  宿主删码指引：DeepTalk `agent/app/tools/subagent.py::_run_spec_isolated` 整体可删，
  改为在 loop 构建处注册三个 guard。
- **S2 — per-send 工具集传导**（`get_effective_tools`；`run_agent_spec(inherit_send_filter=True)`）：
  `send/follow_up(tools=...)` 的本次运行有效工具集现在发布到 agent context（innermost-run
  语义：无 `tools=` 的运行重置为无限制），子 agent 的 registry 默认与之**求交**——沉默父的
  孩子不能替它说话、被禁 bash 的父的孩子不能跑 bash；`inherit_send_filter=False` 为显式逃生门。
  workflow 在**提交时捕获**该集合（detached 下 contextvar 不可靠），引擎在 **spec 层** clamp
  每个叶子（对任意 Executor 生效，含 subprocess），并记入 run journal（`allowed_tools` 键，
  旧 journal 缺省 = 不限制），resume 时原样重施——resume 不会静默放宽权限。
  宿主删码指引：DeepTalk `subagent.py::_parent_allowed` contextvar 全套可删（`tools=` 已在传）。
- **S3 — `AgentLoopConfig.subagent_config_factory`**：`(AgentSpec, 默认子配置) -> AgentLoopConfig`
  的宿主工厂；`run_agent_spec` 构建默认极简子配置后过它，产物按原样使用。让宿主按叶子分流
  上下文策略（representation / fold / microcompact），不再需要 fork 建 loop 段。工厂在建
  child session **之前**执行，抛错不泄漏会话。
- **S4 — `register_workflow_tools` 宿主注入点**（`executor_factory` / `budget_factory` /
  `spec_transform`，均按**每次工具调用**以 `(loop, parent_session_id)` 求值）：宿主可注入
  能力钳制 Executor、按配置的 `SharedBudget`、策略性 spec 改写（`spec_transform` 抛
  `WorkflowSpecError` = 聚合问题回给模型修复）。不传 = 旧行为；不再需要 fork tool handler。
- **S5 — `TimerRunner(delivery=...)`**（类型别名 `TimerDelivery`）+
  `power_loop.workflow.claim_wake` / `parse_workflow_wake` 公开：只替换 firing 的最后
  「注入会话」一步——scan / CAS claim / heartbeat / stale 恢复 / TIMER_FIRE hook（veto/去重点）
  全部照旧先行，故 `register_wake_guard` 的精确一次去重对自定义投递**免费生效**；投递抛错 →
  重臂 +30s 重投（at-least-once）。宿主可把 wake 路由进自己的运行管道（DeepTalk：经 api 造
  `agent.trigger`）。`claim_wake` 仅服务完全绕开 TimerRunner 自轮询 `due_timers()` 的宿主；
  `parse_workflow_wake(note)` 从 timer note 判别/提取 workflow run id。
  注意：`eager_wake=True` 直调 `loop.follow_up` 不经 TimerRunner——装了自定义 delivery 的
  宿主应保持其为 False（docstring 已明示）。

## [3.13.1] — 2026-07-05

Fix, backward-compatible (patch).

### Fixed
- **Fold-trigger token estimate now matches the live hot/cold render** — in projection mode
  `_plan_and_run_projection_fold` rendered its trigger snapshot WITHOUT stamping recency, so the
  kept-recent sends were estimated at COLD size while the real prompt renders them HOT — the
  estimate under-counted and the fold fired late (risking context overflow before folding). It now
  stamps the snapshot (recency keyed on the live cursor) before the estimate render, so the trigger
  fires against the true prompt size. The isolated fold-summary render is unaffected (its span is
  all old → cold regardless).

## [3.13.0] — 2026-07-05

Feature, backward-compatible (minor).

### Added
- **`ProjectedRepresentation.stamp_render_context(rows, current_send_index)` + `hot_window`** — a
  recency-aware projection seam. The projection render is invoked PER-SEND (each send's rows in
  isolation), so a `render_project_row` can't see recency or other sends on its own. Call this ONCE
  over the full ordered project-row set before rendering: it stamps each row with a transient
  `.recency` (0 = newest, keyed on ABSOLUTE `send_index` vs the cursor — so the fold's isolated
  old-span render still classifies old sends as cold) and `.render_ctx` (`latest_key` = the newest
  send per dedup key `k`, for cross-send read dedup; + `hot_window`). Rows without the stamp read as
  cold (safe default). `stateful_loop` calls it in the projection context build; the base
  `render_project_row` ignores the attrs (only a recency-aware override consumes them). `hot_window`
  should be set == the fold's `keep_last_sends` so the recency and fold boundaries coincide.

## [3.12.0] — 2026-07-05

Feature/fix, backward-compatible (minor).

### Changed
- **`ProjectedRepresentation` keeps mid-send user rows in chronological position** — durable
  mid-send injections (LLM_BEFORE `persist_messages` reminders, COMPLETE_DECIDE finalize prompts,
  drained steering follow-ups) used to be folded into the send's single `{"input": [...]}` list,
  which rewrote history: the NEXT send's projected context showed every mid-run reminder as if it
  arrived before the work started. Now only user rows BEFORE the first assistant turn form the
  input; later user rows appear in the `tools` timeline as `{"name": "__user__", "text": ...}`
  entries at their real position (rendered as `[user] …` lines). Hosts with custom save/render
  sources see the new entries flow through the tools list; hosts aligning tool entries by index
  must skip `__user__` entries. Existing stored project rows are unaffected (additive; only new
  sends produce interleaved entries). The deprecated 2.x `DefaultDeterministicProjector` is
  unchanged.

## [3.11.1] — 2026-07-05

Fix, backward-compatible (patch).

### Fixed
- **Sub-agent loops inherit the parent's `retry_policy`** — `run_agent_spec` built the child
  `AgentLoopConfig` without one, so child LLM calls ran on the NO-RETRY path. The child reuses
  the parent's LLM client (same httpx connection pool), and a delegation typically fires right
  after a long tool phase — exactly when pooled keep-alive connections have gone stale — so the
  child's FIRST streaming call could die on a bare `httpx.ReadError` (empty message), killing
  the whole delegation with an opaque "Error: ". Observed twice in production before diagnosis.

## [3.11.0] — 2026-07-05

Feature, backward-compatible (minor). No STABLE API break.

### Added
- **`HookPoint.COMPLETE_DECIDE` + `CompleteDecideCtx`** — a hook consulted at a send's two
  terminal boundaries: natural completion (the model produced a round with no tool calls,
  `reason="completed"`) and round-budget exhaustion (`reason="hit_round_limit"`, before the
  forced wrap-up call). A handler that SHORT_CIRCUITs with a non-empty `ctx.inject` gets that
  text appended as a **durable user message in the SAME send**, and the loop keeps running with
  `ctx.extra_rounds` more rounds. Use it for same-send finalize turns ("before you stop, update
  memory / todo / deliver") that previously had to be a separate `follow_up()` send.
  `ctx.fire_count` tells the handler how many injections already happened this send — handlers
  MUST bound themselves with it. Internally the round loop now runs against a dynamic
  `_round_limit` instead of a fixed `range(max_rounds)`; behavior without a registered
  COMPLETE_DECIDE handler is unchanged.

### Fixed
- **`run_agent_spec` empty-answer fallback** — a sub-agent that delivered its answer through
  tools (send_message, board posts) and ended its last round with a blank assistant text used
  to return `final_text=""` (surfacing as "(sub-agent returned no text)" to hosts). On a
  `completed` child with empty final_text, the last non-empty assistant text is now recovered
  from the child transcript before the ephemeral cleanup deletes it.
- **`apply_patch` / `edit_file` warn on broken JSON** — editing a `.json` file into invalid
  syntax (dangling comma after a removed line, etc.) used to succeed silently; the agent then
  burned rounds diagnosing a downstream "not valid JSON" error. The tool result now carries a
  loud positioned warning (the write is kept — the edit may be one step of a multi-edit).

## [3.10.0] — 2026-07-04

Feature, backward-compatible (minor). No STABLE API break.

### Added
- **Per-call `max_rounds` override on `send()` / `follow_up()`** — run a single continuation with
  a different round budget than `config.max_rounds`, without mutating the shared loop config
  (applied via a per-run `dataclasses.replace(config, max_rounds=…)`, like the existing
  `system_prompt` override). Use it for a short, bounded "finalize" turn (e.g. "before you stop,
  update your memory / todo") separate from the main loop's budget. On `follow_up`'s STEERED path
  (an in-flight loop) it is ignored — the running loop's own budget governs the drained message.
  Adding a keyword-only param with a default to the STABLE `send`/`follow_up` is backward-compatible.

## [3.9.0] — 2026-07-04

Feature, backward-compatible (minor). No STABLE API break.

### Added
- **`LlmBeforeCtx.persist_messages`** — an `LLM_BEFORE` hook can now inject a *durable* turn, not
  just an ephemeral one. Any message a handler appends to `ctx.persist_messages` becomes a REAL
  history + store row (stamped with the round's `send_index`, via the loop's own append path) AND
  is added to that round's request tail. This is the persisted counterpart to the request-only
  edits to `ctx.messages` (which never touch history). Use it for injected turns that must survive
  the send — e.g. a periodic "you haven't called tool X in N rounds" reminder. Adding a field with
  a default to the STABLE `LlmBeforeCtx` dataclass is backward-compatible.

Bugfix, backward-compatible (patch). No STABLE API change.

### Fixed
- **`read_file` no longer rejects valid UTF-8 / CJK text as "binary".** `_looks_binary` only
  counted ASCII bytes (32–126), so a non-Latin UTF-8 file — e.g. a Chinese Markdown doc, whose
  bytes are mostly >126 (3 per CJK char) — fell below the 0.70 text ratio and was refused with
  "Refusing to read binary-looking file as text". It now keeps the ASCII fast-path, then accepts
  non-Latin UTF-8 that decodes cleanly (few U+FFFD / control chars); genuine binary (NUL bytes,
  control-char-heavy) is still rejected. Affects `read_file` and the attach/preview path.

## [3.8.0] — 2026-06-28

Additive, backward-compatible (minor bump): a new store schema version (4 → 5) with an automatic,
idempotent `ALTER … ADD COLUMN` migration; no STABLE API removed or renamed.

### Added
- **Cached (prompt cache-read) token accounting.** `cached_tokens` is now captured from each LLM
  call's usage (`cache_read_tokens`) and persisted to the store's `usage_rounds` (per round) and
  `session_stats` (cumulative). Surfaced on `SessionStatsRow.cached_tokens` (default `0`).
  `Store.record_usage` gained an optional `cached_tokens` parameter. The store schema bumps to v5;
  opening an existing store runs a single `ALTER TABLE … ADD COLUMN cached_tokens` on `usage_rounds`
  + `session_stats` across SQLite / PostgreSQL / MySQL. Real-LLM-verified (`tests/real`).

## [3.7.0] — 2026-06-28

Additive, backward-compatible (minor bump): no STABLE API removed/renamed, no schema change.

### Changed

- **`max_chars <= 0` now means "no truncation" (unlimited)** for `ProjectedRepresentation` and
  `DefaultDeterministicProjector` (and the shared `_truncate` helper). Previously `max_chars <= 0` was
  rejected at construction (`ValueError`). It is now accepted and disables the library's per-field
  truncation entirely — so a host that does its own (e.g. a whitelist-aware, per-tool) limiting can
  turn the library cap off cleanly instead of passing a large sentinel. Positive `max_chars` is
  unchanged. Purely permissive: code that passed a positive value behaves identically.

## [3.6.0] — 2026-06-26

Additive, backward-compatible (minor bump): no STABLE API removed/renamed, no schema change.

### Added

- **`note_list` default tool.** A read/list counterpart to `note_add` / `note_update` / `note_delete`:
  returns the agent's persistent notes rendered with their `#id`, `[pinned]` flag, and content
  (`run_note_list` → `store.list_notes` + `render_notes`). This lets an agent read its own memory —
  and obtain the `#id`s needed for `note_update` / `note_delete` — **via an explicit, transcript-visible
  tool call**, instead of depending on the (optional) `MemoryRecallHook` auto-injecting "YOUR NOTES"
  every turn. Registered in `DEFAULT_TOOL_DEFINITIONS` / `DEFAULT_TOOL_HANDLERS` (so it's in the `full`
  preset and selectable via `include=[...]`); no params. The `note_add` description no longer asserts
  notes are "shown at the start of every turn" (that only holds when the recall hook is enabled) and
  `note_update` / `note_delete` now point at `note_list` for the `#id`. The recall hook itself is
  unchanged (still auto-registered when `memory` is set + `builtin_memory_hook=True`).

## [3.5.0] — 2026-06-25

Systematic-review remediation (BUG_REVIEW_3.4.md): 7/7 HIGH + 14/15 MEDIUM + 30/32 LOW findings
fixed or documented, with regression tests (unit suite 1016 green; verified against live MySQL +
the live LLM). Backward-compatible: no STABLE API removed/renamed (minor bump). The store schema
advances **v3 → v4** — existing MySQL stores migrate (TEXT → LONGTEXT) automatically on open under
AUTO_CREATE; SQLite/Postgres get a no-op version bump. 3 Anthropic-transport findings deferred
(extended-thinking signatures, ModelCapabilities, tool_result images).

### Fixed — systematic-review HIGH findings (BUG_REVIEW_3.4.md)

- **Projection mode dropped the submitted answer on `submit_input()` / `resume()` / `abort_pending()`
  (H1).** Out-of-band tool rows were persisted with `send_index=NULL`, so projection-mode rendering
  partitioned them into the legacy prefix *before* their own assistant `tool_call` — `align_tool_calls`
  then dropped them as orphans and the model answered blind. These rows now carry the in-flight
  send's index, keeping the tool result paired in the active send. Verbatim/default mode was never
  affected.
- **MySQL `TEXT` (64 KiB) columns hard-failed large content (H2).** Every free-text/JSON column in the
  MySQL dialect is now `LONGTEXT` (SQLite/Postgres `TEXT` is already unbounded). A >64 KiB tool
  result / system prompt / `tool_calls_json` write that raised `DataError(1406)` now succeeds.
  **Schema v3 → v4**: existing MySQL stores are migrated with `ALTER … MODIFY … LONGTEXT` on open
  (AUTO_CREATE); SQLite/Postgres get a no-op version bump.
- **Multimodal content was stringified on persist and never parsed back (H6).** Structured (list/dict)
  message content is now JSON-encoded with a `meta.content_encoding` marker and reconstructed on
  reload, so vision/multimodal messages reach the model as the original structure instead of a literal
  JSON string (it broke on the very first send, since history is rebuilt from the store).
- **`trim_history` could emit a dangling `assistant(tool_calls)` (H5).** The body front-fill now drops
  a trailing assistant whose tool results didn't fit, so the public `trim_history` never produces a
  tool_call without its result (both OpenAI and Anthropic 400 on that).
- **`metrics_sink` backend exceptions aborted the agent loop (H7).** A raising backend (StatsD socket
  error, Prometheus label error) on a non-suppressing bus (the default) propagated out of `publish()`.
  The dispatch is now wrapped in a log-and-swallow guard, matching `otel_sink`.
- **Workflow `foreach`/`parallel` fanout was unbounded (H3).** `max_concurrency` (≤ 64) and literal
  `items` length (≤ 4096) are now capped at spec-validation time; a dynamic `items_from` list is
  capped at runtime before tasks are created; and a per-run leaf ceiling (10 000) fail-closes nested
  / programmatic fanout independent of the optional budget.
- **A leaked grandchild process could hang a workflow leaf forever (H4).** The subprocess worker now
  runs in its own process group (`start_new_session`), terminate signals the whole group
  (`killpg`, pgid captured at spawn), and every `communicate()` drain is bounded — a grandchild that
  inherited the worker's stdout/stderr can no longer keep the pipe open and stall the run.

### Fixed — systematic-review MEDIUM findings (BUG_REVIEW_3.4.md)

- **Round-limit wrap-up summary now persisted** to the transcript (success branch), so the next
  send's history isn't a dangling "summarize…" prompt with no answer.
- **Balanced stream events on LLM retry:** `STREAM_STARTED`/`STREAM_COMPLETED` are now emitted
  per-attempt (the terminal used to fire once for N attempts).
- **Projection recall hint restored for migration-seeded folds** (`compact_from_send==0`): the
  `recall_send` note now shows whenever a fold covers ≥1 real send.
- **`read_file` size cap enforced even with a `limit`** — paging a multi-GB file no longer loads the
  whole file into memory.
- **Workflow driver + linked leaf sessions are cleaned up** after `run()` (new `close_driver=True`
  default; set `False` to retain for inspection) — no more session-row leak per run.
- **`foreach` `as` validated as an identifier** so the body's `{{var}}` actually binds.
- **Workflow reference reachability validated:** forward refs, parallel-sibling refs, and
  cross-branch-case refs are now rejected at parse time instead of failing/silently-dropping at run.
- **`budget_exceeded` is a terminal journal status**, freezing the run against orphaned late writes.
- **`POWER_LOOP_SUPPORTS_*` capability overrides are now wired** from env into the OpenAI-compatible
  transport (previously parsed but never applied).
- **`TimerRunner.stop()` aborts an in-flight timer-fired run** (via an owned cancellation token)
  instead of blocking for the whole agent run to finish.
- **One bad `SKILL.md` no longer takes down the loader:** non-UTF-8 files are read with
  `errors="replace"` / skipped, not raised.
- **`parse_structured` tolerates an odd quote count in the LLM's prose preamble** (string-state is no
  longer carried across prose into object detection).
- **`otel_sink` span creation / `set_attribute` are guarded** (log-and-swallow), upholding its
  "never break the loop" contract.
- **Opt-in value-secret redaction** for the logging/JSONL sinks (`redact_value_secrets=True`) scrubs
  secret-shaped substrings (Bearer/sk-/AKIA/JWT/…) inside string values; the key-name-only default is
  now documented.

### Fixed — systematic-review LOW findings (BUG_REVIEW_3.4.md)

_Loop / events:_ round-limit + no-tools-drain rounds now emit `ROUND_COMPLETED`/usage and the
`@phase` decorator pairs start/end on every error path; pending `assistant_seq` uses the durable DB
seq, not the in-memory history index.
_Projection / compaction:_ `ProjectionRenderConfig` coerces JSON bools, `render_user_row` tolerates a
non-list `input`, and a malformed `CONTEXT_COMPACT_THRESHOLD` is fail-soft.
_Tools / sandbox:_ `BackgroundManager` task cache + persistent-bash output are now bounded; the bash
home-scope guard matches on path boundaries (no superstring false-positives).
_Workflow:_ `output_schema` shape validated; resume refuses a still-live in-process run (`force` to
override); `reap_runs` documents its mtime-liveness hazard; EPHEMERAL sub-agent sessions are cleaned
up on a raised child run; sub-agent blackboard author is the spec name.
_Store:_ MySQL `table_prefix` length capped; `upsert_background_task`/`mark_background_seen`
serialized on the state-row lock; legacy WAL `checkpoint` logs a BUSY result; legacy-vs-new
schema-version namespaces documented as non-portable.
_Misc:_ timer `heartbeat_interval_s` floored (no busy loop); `MemoryRecallHook` memoizes per-session;
`StructuredOutputSpec.examples` folded into the schema description; tool-role messages always emit
`tool_call_id`; `JsonlSink` with `backup_count=0` no longer truncates on rotation; the OpenAI
streaming accumulator no longer merges ambiguous (no id/index) sequential tool calls; the
`MessageSink` raising contract and a length-preserving-history-swap compaction edge are documented.

### Deferred

- Anthropic extended-thinking signature blocks are still dropped (review finding M10), and the
  Anthropic transport's `ModelCapabilities`/tool_result-image handling (LOW: llm-transport-2/4) — a
  correct fix round-trips the thinking-block signature through `LLMResponse` + persistence + reload
  and the Anthropic multimodal path, and needs a real Anthropic-thinking endpoint to validate.
  Tracked for a focused follow-up.

## [3.4.0] — 2026-06-25

### Added — `ProjectedRepresentation` render is now a first-class extension point

Projection-mode rendering (stored rows → LLM messages) was a single monolithic `render()` you could
only customize by copy-pasting the whole method. It is now extensible two ways, and the **defaults
reproduce the previous output byte-for-byte**:

- **Config (`ProjectionRenderConfig`, provisional export):** a dataclass of pure-scalar format knobs
  — `user_tag` / `project_tag` (with a `{n}` send_index placeholder), `tools_header`, `tool_sep`,
  `tool_arg_sep`, `include_tools`, `include_final_text`, `empty_project`, `fold_note` (with a
  `{range}` placeholder). Pass via `ProjectedRepresentation(render_config=…)`; a plain dict is coerced
  (`ProjectionRenderConfig.from_dict`, unknown keys ignored), so the whole config round-trips through
  JSON and a host can surface it in a UI and retune the rendered context live.
- **Subclass override:** `render()` now delegates to small per-row methods — `render_row` →
  `render_user_row` / `render_project_row` / `render_compact_row`, plus `_render_project` /
  `_render_tool` / `_send_tag` — so a subclass overrides exactly the one shape it wants. A row whose
  kind has no renderer is skipped (unchanged behavior).

## [3.3.0] — 2026-06-25

### Changed — projection keeps the input turn VERBATIM; user-row key `human` → `input`

`ProjectedRepresentation` (projection mode) now renders a send's INPUT/user turn **verbatim** instead
of truncating it to `max_chars`. The input is the actual conversation content — short relative to tool
output and high-value — so truncating it dropped context the model genuinely needs; only the
assistant's WORK (tool args/results + `final_text`) is still compressed, which is where the token
savings actually are.

- The user row's content key is renamed `{"human": [...]}` → `{"input": [...]}`. The input turn is
  NOT necessarily a human — a multi-agent host feeds another agent's message there — so `input` is the
  accurate, neutral name. `render()` reads BOTH keys, so pre-3.3 projection rows render correctly
  after upgrade (no migration / no data rewrite needed). The legacy 2.x `history_projector` path is
  unchanged.

## [3.2.0] — 2026-06-24

### Added — hook-injected context audit log (`pl_hook_events`, schema v3)

The ephemeral context that `LLM_BEFORE` hooks inject per round (e.g. recalled memory) used to
vanish after the LLM call, recorded nowhere. A new **opt-in audit table** `{prefix}hook_events`
captures it for observability — linked to the round's assistant message — **without ever re-entering
history or the LLM request** (so context construction and prefix-caching are unchanged). The audit is
written only onto the sink copy of the assistant message, exactly like `send_index`.

- **Added** `AgentLoopConfig.record_hook_events`: `"off"` (default — zero overhead), `"metadata"`
  (name/source/char-count/position per injected item), or `"full"` (also the injected text).
- **Added** the `{prefix}hook_events` store table (schema **v2→v3**, an idempotent `CREATE TABLE`
  migration — no `ALTER` on the hot `messages` table) with a per-session monotonic `event_id`,
  `message_seq` link, `hook_point`/`hook`/`position`/`kind`, and a JSON `payload`
  (`{v, items:[{role,name,source,chars,content?}], item_count, total_chars}`).
- **Added** `SessionStore.list_hook_events(session_id, *, message_seq=None)` and the `HookEventRow`
  type (PROVISIONAL — not in `STABLE_API`). Rows are deleted with the session
  (`close_session_tree`). The table is NOT exported by `export_session` (audit-only).
- Capture is an identity-diff of the post-`LLM_BEFORE` message list against a pre-hook snapshot, so it
  records both tail- and front-positioned injection. Nothing is written to `pl_project_messages`.
- Scope/limitations: captures only **appended** injection (not in-place edits of existing messages);
  one row **per round** (the canonical degraded / `LLM_AFTER`-BREAK rounds are recorded too; an
  `LLM_BEFORE`-BREAK round writes no assistant message so it records nothing). A hook that replaces
  all or most of `ctx.messages` with fresh copies yields a small `inject_unresolved` marker (no
  content) instead of mislabeling the conversation.

## [3.1.0] — 2026-06-23

### Changed — memory recall is now a built-in, overridable hook (ephemeral tail injection)

Memory recall moved from a hardcoded pipeline step that spliced recalled messages
into `self.history` at the **front** (index 0) to a **built-in `LLM_BEFORE` hook**
(`MemoryRecallHook`) that injects them **ephemerally at the request tail** — never
into `self.history`, the store, or the window cache.

Why: front injection put volatile content (notes change as the agent writes them)
ahead of all history, so any change invalidated the entire prompt prefix for
provider prefix-caching. Tail injection keeps `system + prior history` byte-stable
and prefix-cacheable; only the small memory block + new turn are uncached. It also
removes the `self.history`↔seq realignment footgun (no more `on_messages_inserted`).

- **Added** `AgentHooks.replace(...)`, `.remove(name)`, `.has(name)`, and `name=` /
  `replace=` kwargs on `.register(...)` so built-in (`builtin.*`) hooks can be
  overridden or disabled by hosts.
- **Added** `MemoryRecallHook` (public) — the built-in recall hook. Auto-registered
  by `StatefulAgentLoop` when `config.memory` is set (skipped if the host already
  registered one under `MemoryRecallHook.NAME`, or `config.builtin_memory_hook=False`).
  Recall runs **once per send** (memoized on the first round; re-injected each round
  so the within-send tail stays stable) — same cadence as before.
- **Added** `AgentLoopConfig.memory_position` (`"tail"` default | `"front"`),
  `AgentLoopConfig.builtin_memory_hook` (default `True`), and
  `AgentLoopConfig.effective_context_budget()` — the fold/compaction trigger now
  reserves `memory_budget_tokens` of headroom (the tail memory isn't counted by the
  fold trigger, so fold a little earlier to keep `history + memory` within budget).
- **Added** `session_id` to `LlmBeforeCtx`.
- **Changed (behavior)** default memory injection position is now the **tail** (was
  the front). Output prompts for hosts using `config.memory` change accordingly.
- **Renamed** `SQLiteNoteMemory` → `NoteMemory` (it was always backend-agnostic — it
  reads from whatever `SessionStore` is passed: SQLite / Postgres / MySQL). The old
  name is kept as a back-compat alias.
- **Removed** `MessageSink.on_messages_inserted` (Protocol + `NullSink` + `SQLiteSink`):
  it existed only to realign the index↔seq map after the front `self.history` splice,
  which no longer happens. **Breaking for external `MessageSink` implementers** (a
  niche, non-STABLE surface). The `on_compaction` `None`-placeholder handling and
  `list[int | None]` seq maps are unchanged — still load-bearing for projection mode
  and corrupt-history repair.

Note: prefix-stability only converts to real cost savings where the transport hits a
prompt cache. OpenAI-compatible transports auto-cache prefixes (benefits immediately);
Anthropic requires explicit `cache_control` breakpoints, which power-loop does not yet
emit — a separate follow-up.

### Changed — microcompact is now opt-in (default OFF) + fully configurable

`microcompact` (the cheap per-round mechanism that spills OLD oversized tool outputs to
disk and leaves a short pointer, distinct from the LLM-summary fold) is now **OFF by
default** and exposed through `AgentLoopConfig` instead of env-only knobs.

- **Changed (behavior)** microcompact no longer runs unless `microcompact_enabled=True`.
  It only helps when those old outputs are never needed again; otherwise the pointer
  just trades for a re-read. Projection mode (it's verbatim-only anyway) + fold + provider
  prefix-caching already cover the context budget. Long verbatim sessions that read many
  large files and rarely revisit them can opt back in.
- **Added** `AgentLoopConfig.microcompact_enabled` (default `False`),
  `microcompact_size_limit` (default from `CONTEXT_MICRO_SIZE_LIMIT`, else 1000),
  `microcompact_hot_tail` (default from `CONTEXT_MICRO_HOT_TAIL`, else 10), and
  `microcompact_spill_dir` (default `None` → the runtime home's `.cache`). Config takes
  precedence; the `CONTEXT_MICRO_*` env vars remain as defaults for back-compat.
- `ContextManager.microcompact()` gained optional `size_limit` / `hot_tail` / `spill_dir`
  kwargs (fall back to the instance's env-defaulted fields, so direct callers are
  unaffected). Object-store / custom spill backends (`SpillSink`) are a deferred follow-up.

### Fixed — system-prompt assembly deduplicated (preview can no longer drift from the live prompt)

The runtime system-prompt assembly (`base → tool catalog → skill section`) was
**duplicated** in `AgentPipeline.__init__` (the live prompt) and
`StatefulAgentLoop.resolve_system_prompt` (the preview, whose contract is "exactly
what the LLM will see"). Editing one without the other would silently diverge.

- Extracted into one shared `resolve_runtime_system_prompt(...)` helper (in
  `agent/system_prompt.py`, alongside `build_skill_section`); both sites now call it,
  so the preview is byte-identical to the live prompt by construction. Pure refactor,
  no behavior change. New regression test asserts `resolve_system_prompt == ` the
  prompt the LLM actually receives.

## [3.0.2] — 2026-06-23

### Docs (no code/API change)

- Refreshed README (en + zh), the user guide, and examples for the 3.0 two-axis context model:
  `representation` (`VerbatimRepresentation` / `ProjectedRepresentation`) × `fold_strategy`
  (`LLMSummaryFold` / `AgenticFold`) presented as the primary, orthogonal API. Removed the
  now-incorrect "projection and compaction are mutually exclusive" claim and the removed
  deterministic-fold / `max_compact_chars` material; documented the `Representation` / `FoldStrategy`
  protocols, `recall_send` `#N` tags, and `migrate_history_on_switch`, with legacy/deprecated notes
  for the 2.x `compactor=` / `history_projector=` kwargs. Corrected stale facts (example count, LOC,
  test counts). No runtime behavior change from 3.0.1.

## [3.0.1] — 2026-06-22

> Bug-fix release from a deep adversarial review (16 confirmed findings; see `BUG_REVIEW_3.0.md`).
> No public API changes. Each fix has a red-before/green-after regression test
> (`tests/unit/test_deep_review_3_0_fixes.py`, `tests/unit/test_deep_review_3_0_server_fixes.py`).

### Fixed

- **[high] Postgres v1→v2 `send_index` migration is now schema-scoped.** `_column_exists` probed
  `information_schema.columns` across ALL schemas, so a same-named `pl_messages` (with `send_index`)
  in another schema made the `ALTER … ADD COLUMN` get skipped while v2 was stamped → every
  `append_message` crashed. Now scoped to `current_schema()` (mirrors `_table_exists`).
- **[high] Workflow wake-guard claims atomically.** `make_wake_guard` did a bare
  `get_runtime_state`→mutate→`set_runtime_state`, clobbering a concurrent `journal` write on the same
  run key and risking a double-wake. Now routed through the row-locked `mutate_runtime_state`
  (tolerates a stale timer on a deleted session).
- **[high] `abort_pending` primes the sink's tool_calls** so a crash mid-abort persists a
  *consistent* intermediate pending (was `{tool_call_ids:[…], tool_calls:[]}`); and `resume()` /
  `_execute_pending` now self-heals an ids-only pending instead of returning `completed` while the
  session stays permanently stranded.
- **[high] Projection migration no longer drops history on a fold soft-fail.** When the one-time
  switch-to-projection fold timed out / errored, the fallback wrote a `compact` whose `to_send`
  *covered* sends it never merged (reader excluded them = silent permanent data loss), or dropped the
  fold sends while poisoning the done-marker. It now never claims an unfolded range — it preserves
  the would-be-folded sends as individual project rows (a later end-of-send fold compresses them).
- **[medium] `recall_send` is usable under `ProjectedRepresentation`.** `render` now tags each
  projected send with its `#N` send-index (and the folded compact with its covered range), matching
  the tool docstring / host note that tell the model to call `recall_send(send_index=N)`.
- **[medium] `POWER_LOOP_HOME` bash scope guard is default-deny.** It only blocked ~9 hardcoded
  read/write verbs and fell through to *allow* for everything else (`awk`/`base64`/`od`/`python -c`/
  `dd of=`/`truncate`/`ln -s` reached agent-home undetected). A command that references
  un-allowlisted home is now refused regardless of verb.
- **[medium] Legacy `IdentityProjector` no longer silently drops folded history.** It had no `kind`,
  so it was mis-routed onto the projection-fold path and its `render` returned `[]` for compact rows.
  It now declares `kind="verbatim"` (routes to the safe in-place path, never folds) and its `render`
  handles `kind=="compact"`.
- **[low] Legacy projector `keep_last_sends=0`** is no longer silently coerced to 4 (`0 or 4`); a
  verbatim never-fold projector maps to never-fold, a projection one folds at the floor of 1.
- **[low] MySQL `background_tasks` index** renamed to the canonical
  `{prefix}idx_background_tasks_session_status` (was `idx_bgtasks_…`), matching SQLite/PG.

### Known / deferred (documented in `BUG_REVIEW_3.0.md`)

- [medium] Verbatim `keep_last_sends` is still counted as `keep_last_n` exchanges (a fix needs
  `send_index` threaded into the in-place compactor; no correctness/data-loss — atomic tool pairs are
  preserved). [low] projection→verbatim degrade rendering, resume-before-send `send_index=NULL` rows,
  verbatim-fallback prefix seq mapping — narrow, non-default.

## [3.0.0] — 2026-06-22

> **MAJOR — orthogonal context axes.** Message **representation** (how each finished send is
> recorded/rendered) and **fold strategy** (how older history is compacted once over budget) are now
> two independent, fully config-driven axes. Any representation composes with any fold strategy, and
> both accept custom implementations of their Protocol. The 2.x `compactor` / `history_projector`
> kwargs still work (mapped onto the new axes with a `DeprecationWarning`); a future major drops them.

### BREAKING — Public API (STABLE)

- **Removed public exports** `HistoryProjector`, `IdentityProjector`, `DefaultDeterministicProjector`,
  `ProjectedCompact`. Their roles are replaced by the new `Representation` Protocol and its
  implementations. The `power_loop.runtime.history_projector` module still exists for one transition
  release (deep-import only; not in `power_loop.__all__`).
- **Removed the deterministic / no-LLM fold.** Concatenation/truncation is not compaction; folds are
  now always LLM-backed. Projection-mode compaction is therefore LLM-backed too, consistent with
  verbatim mode (previously projection folded via a deterministic concat).

### Added — STABLE

- **`Representation` axis**: `VerbatimRepresentation` (full messages, byte-identical history) and
  `ProjectedRepresentation` (per-send terse projection; original detail kept in `pl_messages`,
  `recall_send` re-expands). Custom representations implement the `Representation` Protocol
  (`kind` / `version` / `project_send` / `render`). Helper types `ProjectedSend`, `ProjectedRow`.
- **`FoldStrategy` axis**: `LLMSummaryFold` (default — one summary call, no side effects) and
  `AgenticFold` (LLM + a bounded tool loop that persists durable facts as notes). Custom strategies
  implement the `FoldStrategy` Protocol; helper types `FoldContext`, `FoldResult`, `NoteOp`. Trigger
  (`trigger_ratio`) and keep-recent floor (`keep_last_sends`) live on the strategy; a fold always
  keeps whole sends (never splits an atomic tool-call/result pair).
- **`AgentLoopConfig` fields**: `representation`, `fold_strategy`, `fold_timeout_s` (default 120s;
  the fold runs OUTSIDE the store lock, soft-fails on timeout), `migrate_history_on_switch`
  (default True — fold prior history into the new form once on a representation/fold change),
  `repair_corrupt_history` (default False — durably deactivate the orphan tool rows the always-on
  `align_tool_calls` sanitizer drops).

### Changed

- `AgentLoopConfig.compactor` / `history_projector` / `migrate_history_on_projection_switch` are
  **deprecated** and mapped onto `representation` / `fold_strategy` / `migrate_history_on_switch` in
  `__post_init__` (a legacy projector's `keep_last_sends` / `trigger_ratio` seed the mapped fold; a
  legacy `compactor`, including `None` = no compaction, is preserved exactly under verbatim). Emits
  `DeprecationWarning`. No behavior change for existing call sites.

### Added — "Build your own tools" guide, example & parity tests (no API change)

- **New guide [Build your own tools](docs/en/user-guide/build-your-own-tools.md)** (+ 中文) —
  recreates every "special" built-in (background task / sub-agent / mini-workflow / durable timer /
  human-input / blackboard / memory) as a plain custom tool using ONLY public primitives, with the
  exact seams, parity, and honest gaps per feature. Grounded in a per-feature design+adversarial-verify
  pass over the built-in source.
- **Example `42_build_your_own_tools.py`** — all seven reimplementations in one runnable file
  (`run_agent_spec`, `HumanInputRequired`, a custom `RuntimeProjector`, a `MemoryProvider`,
  `get/set_runtime_state`, `add_note`), with a real-LLM demo (verified) using the custom
  remember/board/delegate as drop-ins.
- **`tests/unit/test_byo_tools.py`** — deterministic parity tests for all seven (scripted-LLM),
  loading the canonical code from the example so docs/example/test never drift; plus the real
  `test_example_42`. Surfaces the gotcha that a custom `RuntimeProjector` must *replace* the default.

### Added — Async orchestration docs, example & cross-cutting tests (no API change)

- **New guide [Async orchestration](docs/en/user-guide/async-orchestration.md)** (+ 中文) — the
  cohesive model the per-feature pages (background tasks / sub-agents / workflows / timers) assume:
  power-loop is **host-driven** (no daemon); the `send` / `resume` / `submit_input` / `follow_up`
  wake API and when to use each; how every async result re-enters (background → `RuntimeProjector`
  `<background_updates>`; timer → `follow_up`; sub-agent → inline; pending → `resume`; input →
  `submit_input`); persistence, `send_index` & crash recovery (`SessionPendingError` →
  `resume`/`abort_pending`/`heal_pending`); the projection/compaction interaction; a copy-pasteable
  **custom async-wake tool recipe** (public seams only); and a troubleshooting section.
- **Example `41_custom_async_tool.py`** — a custom tool that starts async work, returns immediately,
  and wakes the agent via `schedule_timer` + `TimerRunner` + `follow_up` (verified against a live LLM).
- **`tests/unit/test_async_projection_interaction.py`** — covers the previously-untested cross-cut:
  a `follow_up` on an idle session opens its own projected send (own `send_index`); background
  updates re-enter once via `BackgroundRuntimeProjector` and are then marked seen (`mark_seen=False`
  keeps re-injecting).

## [2.2.0] — 2026-06-21

### Added — Agentic memory-aware compactor (opt-in)

- **`AgenticMemoryCompactor`** (`power_loop.runtime.compact`) — an opt-in `Compactor` that runs a
  **bounded, memory-aware agent loop** at the compaction boundary: it lets the model use memory
  tools (by default the existing `note_add` / `note_update`) to persist durable facts into the
  **current session's notes** *before* folding the rest into the `compact_note` summary. Subclasses
  `DefaultCompactor` (reuses its trigger + span selection unchanged); only the summarize step is
  agentic. **Default behavior is unchanged** — construct it and pass it as
  `AgentLoopConfig.compactor` to opt in.
  - Safe: the loop is a FLAT, bounded (`max_rounds`, default 4) tool-use loop — not a nested
    `StatefulAgentLoop`, so it can never recurse into another compaction. On ANY failure (no tool
    support, malformed output, exception) it **falls back to the plain single-call summary**, so it
    never blocks a fold.
  - Customizable: `memory_tools` (a `ToolRegistry`; defaults to the note tools), `system_prompt`
    (defaults to the planned `DEFAULT_COMPACTION_AGENT_PROMPT`), `max_rounds`.

### Added — Send-context projection (opt-in; PROVISIONAL public API)

A new way to control what each send feeds the LLM, separate from in-place compaction. By
default nothing changes (`history_projector=None` → verbatim history + the existing compactor).

- **`AgentLoopConfig.history_projector: HistoryProjector | None`** (default `None`). When set,
  the loop feeds the LLM a per-send **plain-text projection of FINISHED sends** plus the
  in-flight send verbatim, instead of the full verbatim history. Mutually exclusive with
  `compactor` (set `compactor=None`; enforced at construction **and** re-validated on
  post-construction reassignment) — the projection layer replaces in-place compaction.
- **`HistoryProjector` Protocol** + two implementations (new module `power_loop.runtime.history_projector`):
  - `IdentityProjector` — stores/renders each send verbatim (history identical to the default;
    useful to verify the seam introduces no change).
  - `DefaultDeterministicProjector` — generic, no-LLM structured summary: each tool call is
    summarized via the tool's optional `project()` hook else a truncating fallback, rendered to
    terse plain text with **no OpenAI tool-call protocol fields** (so a projected past send can't
    dangle a tool pair and is provider-agnostic). Older sends fold into an append-only `compact`
    row **once the rendered prefix reaches `max_tokens × trigger_ratio`** (token-driven, mirroring
    `DefaultCompactor`'s policy); `keep_last_sends` is the always-kept-recent floor.
- **`ToolDefinition.project`** — optional `(args, result) -> dict|str` self-projection hook so
  each tool decides how it appears in projected history (`compare=False`, doesn't affect equality).
- **`recall_send(send_index)` default tool** — re-expand one finished send's FULL `pl_messages`
  detail (original tool calls + results) on demand.
- **Store: new `{prefix}project_messages` table (schema v1 → v2)** — the derived projection layer
  (`pl_messages` stays an immutable, append-only audit log; this table is rebuildable and excluded
  from session export). `SessionStore.upsert_project_message` / `load_project_messages` /
  `latest_project_compact`; new row type `ProjectMessageRow`. The v1→v2 migration is idempotent
  (`CREATE TABLE IF NOT EXISTS`) and runs under the provisioning lock on SQLite/PostgreSQL/MySQL.
- **`pl_messages.send_index` column** — a monotonic, never-resetting per-session send index
  written on every row (a real, queryable column — NULL on pre-v2 rows; never sent to the LLM);
  the authoritative send boundary the projection layer and transcript tooling use. The v1→v2
  migration adds it via a guarded `ALTER TABLE … ADD COLUMN`.

New public exports: `HistoryProjector`, `IdentityProjector`, `DefaultDeterministicProjector`,
`ProjectedSend`, `ProjectedRow`, `ProjectedCompact`, `ProjectMessageRow` (PROVISIONAL — in
`__all__`, not yet in `STABLE_API`). Example: `examples/40_send_context_projection.py`. No
breaking changes to the STABLE API.

### Fixed — projection pre-release hardening (deep-review remediation)

A multi-agent review of the projection layer surfaced these; all fixed with regression tests
(the feature is still unreleased, so these refine the additions above rather than ship a patch):

- **Pre-projection / legacy rows are no longer silently dropped from context.** Rows written
  before projection was enabled — or before v2, or restored via export→import — carry
  `send_index = NULL`. The reader now renders them **verbatim as a temporally-first prefix**
  (instead of excluding everything that doesn't equal the current send index), and a session
  resumed in projection mode with no allocated send index fails loudly rather than feeding all
  rows as one pseudo-send. The `send_index` coercion is `>= 1`-explicit (no longer treats `0` as
  unset by accident).
- **Tool results pair correctly under duplicate / missing / empty `tool_call_id`.** Results are
  matched to assistant tool-calls via an order-preserving multimap (duplicate ids no longer
  collapse onto one result), and a missing result (no tool row) is rendered as `<missing>`,
  distinct from a produced-but-empty `""`. `ToolDefinition.project`'s `result` is now `str | None`
  (`None` = no result produced). Malformed `tool_calls` (a non-dict `function`) no longer raise in
  the projector, `recall_send`, or `SessionPendingError`.
- **Atomic, concurrency-safe projection write.** A finished send's projection rows and any
  compaction fold are written in **one transaction under the session-state row lock** (new
  internal `SessionStore.write_send_projection_locked`), so a crash can't leave a half-projected
  send and two `StatefulAgentLoop`s sharing a store can't double-write a fold.
- **`send_index` survives export/import** (added to the messages export columns) so a v2 session
  round-trips with projection intact.
- **`HistoryProjector` Protocol now declares `trigger_ratio`** (the token-fold fraction), matching
  what the loop reads and the built-ins provide.
- **MySQL migration failure is actionable** — the error now surfaces the actual migration steps
  (including the `ALTER … ADD COLUMN`, which `provisioning_ddl` omits) and explains the
  DDL-auto-commit / re-run-with-AUTO_CREATE recovery.
- **`recall_send`** truncates the message body before appending the `[tool_calls: …]` summary, so
  a long message no longer hides that it made tool calls.

A second review round (config/validation, unsafe mutations, max-length, fallbacks) added:

- **A missing or stale-version projection no longer drops a send from context.** If a send's
  end-of-send projection write failed/crashed (it is best-effort), or its rows were written by a
  **different `projector.version`** (the user changed the projector), the reader now renders that
  send **verbatim from the immutable `pl_messages`** instead of silently omitting it. As a side
  benefit, an imported session (projection excluded from export) renders correctly and re-folds on
  the next send. A stale-version `compact` row likewise falls back to its covered range verbatim.
- **A misbehaving projector degrades instead of losing the send.** The fold decision
  (`render`/`compact`/trigger) is wrapped: an exception skips the fold (**rows still commit**)
  rather than aborting the whole locked write. The tool `project()` hook was already exception-guarded.
- **Projector params are validated at construction.** `IdentityProjector` /
  `DefaultDeterministicProjector` now reject `trigger_ratio ∉ (0, 1]` (incl. `NaN`, which would
  otherwise crash `int(max_tokens × NaN)`), `keep_last_sends < 0`, `version < 1`, `max_chars ≤ 0`,
  and `max_compact_chars < 0`. `AgentLoopConfig` rejects `max_tokens ≤ 0` when a projector is set,
  and a rejected post-construction reassignment now **rolls back** (no half-applied invalid config).
- **`DefaultDeterministicProjector` bounds the folded `compact` row** via a new
  **`max_compact_chars` (default 4000; `0` = unbounded)**. The no-LLM projector concatenates, so
  without a cap the compact — and the rendered context — grew without bound over a long session;
  it now keeps the most-recent tail plus a drop marker (dropped detail stays in `pl_messages`,
  recoverable via `recall_send`).
- **Per-session projector/compactor exclusion.** A session with in-place compaction history
  (`compact_note` rows) is refused in projection mode (a cross-run mode switch the config-level
  check can't catch). `send_index` coercion is also exception-guarded against a corrupted
  `runtime_state` value (non-numeric / inf / nan → treated as unallocated, never a crash).
- **`DefaultDeterministicProjector.max_chars` default raised 200 → 300** (per-field projection
  truncation budget; it was already a configurable field).

A third round (mode-switch robustness + corruption self-heal) made the loop never brick a session:

- **Switching a session's history mode never throws.** The previous round *refused* (raised) when a
  projection-mode session had in-place compaction history, and when `resume()` ran before any
  `send()`. Both now **degrade to a best-effort verbatim render and log a warning** instead. The
  session's **original mode is recorded** (`runtime_state["history_mode"]`) on first run, for
  inspection and switch detection. (`history_projector`/`compactor` remain mutually exclusive *per
  config*; this is the cross-run, per-session story.)
- **Self-healing malformed history (new `power_loop.runtime.history_sanitize.align_tool_calls`).**
  A corrupt row in `pl_messages` (a crash between an assistant tool-call row and its result, a bad
  import, a manual edit) would make the provider reject the whole prompt and **repeat on every load
  — bricking the session forever**. The assembled prompt is now realigned before **every** LLM call:
  an orphan tool result is dropped, a mid-history unanswered call gets a synthesized placeholder
  result, and a trailing pending call is left untouched. Always-on, mode-agnostic, a no-op on a
  healthy history; each repair logs a warning.
- **Opt-in durable repair: `AgentLoopConfig.repair_corrupt_history` (default `False`).** When `True`,
  the orphan rows the sanitizer drops are also deactivated in the store (new `MessageState.DROPPED`
  + `SessionStore.deactivate_messages`) so they aren't re-sanitized every load — kept in the full
  audit, excluded from the active history.
- **The original history mode + projector config is recorded in `SessionRow.metadata`** (`history_mode`,
  `projector_version`, `projector_trigger_ratio`, `projector_keep_last_sends`) on first run — inspectable,
  and the baseline the switch-warning compares against (new `SessionStore.merge_session_metadata`).
- **One-time history migration on a mode switch** (`AgentLoopConfig.migrate_history_on_projection_switch`,
  default `True`). When a session with prior NON-projection history is first opened in projection mode,
  its prior history is folded into the projection table once — a `compact` (seeded from an in-place
  `compact_note` if present) plus the most-recent `keep_last_sends` as project rows — so the session
  becomes projection-native instead of rendering prior sends verbatim forever. Best-effort (falls back
  to verbatim on failure), idempotent (`projection_migrated` marker in metadata), runs only when the
  projection table is empty, and folds via `projector.compact()`. New `SessionStore.write_projection_migration`.

## [2.1.0] — 2026-06-17

A correctness-and-robustness release: it resolves every confirmed and contested finding from
`BUG_REVIEW_2.0.md` for the 2.0 pluggable store. No breaking changes to the STABLE API; two
small backward-compatible public additions.

### Added

- `StatefulAgentLoop.ensure_store()` — public accessor that opens the lazily-bound store and
  returns it. Host integrations that need the store before the first `send` (e.g. wiring up a
  `SqliteBlackboard`) should `await loop.ensure_store()` instead of reading `loop.store`,
  which is `None` until the store is first opened.
- `SessionStore.close_session_tree()` — like `close_session`, but returns the id of every
  deleted session (the named session plus any cascaded LINKED descendants).

### Fixed

The remaining confirmed findings from `BUG_REVIEW_2.0.md` (G9–G18) and all six contested
items (C1–C6). Each has a regression test; the backend-specific ones run against the real
PG/MySQL test servers.

- **G9** (`store.py`) — `create_timer` / `add_note` allocated the per-session id with an
  unlocked `SELECT MAX(id)+1`, so two concurrent allocators on PG/MySQL could collide on
  the composite PK. They now take the `session_state` row lock (`lock_state` → `FOR UPDATE`)
  first, like `append_message`; a missing state row is tolerated (legacy behaviour).
- **G10** (`schema.py`) — the printed `provisioning_ddl` version stamp is now the
  dialect-aware idempotent form (`ON CONFLICT DO NOTHING` / `INSERT IGNORE`), so re-applying
  the script (Terraform/Ansible/CI) no longer fails on a duplicate-key error.
- **G11 / C2** (`backends/sqlite.py`) — a failed `ROLLBACK` no longer masks the caller's
  original exception, and a failed `COMMIT` now rolls back instead of leaving the shared
  connection wedged in an open transaction for the rest of the process.
- **G12** (`backends/sqlite.py`) — `PRAGMA auto_vacuum=INCREMENTAL` now runs *before*
  `journal_mode=WAL`, so it actually takes (it was a silent no-op, leaving incremental
  `vacuum()` unable to reclaim space). DBs created by an older build need a one-time full
  `VACUUM` to switch the mode.
- **G13** (`store.py`) — `set_pending({})` now normalizes an empty dict to SQL `NULL`
  (round-trips to `None`), matching the legacy oracle.
- **G14** (`schema.py`) — concurrent first-boot `AUTO_CREATE` now serializes across
  processes with a cross-process provisioning lock (PG `pg_advisory_xact_lock`, MySQL named
  `GET_LOCK`, SQLite no-op), so N instances racing on a fresh DB converge instead of all but
  one crashing with a raw duplicate-object error (verified: 7/8 fail without it).
- **G15 / G16** (`schema.py`) — `VERIFY` probes the catalog instead of swallowing every
  exception, so a transient connection/permission failure surfaces as itself rather than as
  "schema not initialized"; and it now checks every data table exists, not just the version
  row, so a partially-dropped schema fails at open time rather than on the first write.
- **G17** (`stateful_loop.py`) — sync `close()` called inside a running loop keeps a strong
  reference to the scheduled `store.close()` task so it can't be GC'd mid-flight.
- **G18 / C6** (`backends/mysql.py`, `workflow/runner.py`) — MySQL `_args` always returns a
  tuple so the driver's `%`-collapse pass runs even for parameterless statements (a literal
  `%` no longer leaks as `%%`); and `spawn_background` takes the caller's resolved store
  instead of reading the lazily-opened `loop.store` (which is `None` until first async use).
- **Loop** (`stateful_loop.py`, `examples/29_shared_blackboard.py`) — added a public
  `StatefulAgentLoop.ensure_store()` accessor; the shared-blackboard example used
  `loop.store` (now `None` until lazily opened) and crashed at startup. Host integrations
  that need the store up front should `await loop.ensure_store()`.
- **C1** (`schema.py`, `store.py`) — `table_prefix` is now validated (`[A-Za-z_]\w*` or
  empty) at `SessionStore`/`ensure_schema`, since it is concatenated into SQL identifiers
  without quoting; a tenant/config-derived prefix can no longer inject DDL or silently
  produce malformed SQL.
- **C3** (`backends/sqlite.py`) — `transaction()` uses `BEGIN IMMEDIATE` instead of a
  DEFERRED `BEGIN`. Every store transaction is a read-modify-write, so taking the RESERVED
  lock up front lets `busy_timeout` serialize cross-process writers instead of deadlocking on
  a lock upgrade (`database is locked`). The multi-writer correctness the docs advertise now
  actually holds for the default backend.
- **C4** (`store.py`, `stateful_loop.py`) — new `SessionStore.close_session_tree()` returns
  every deleted session id (named + cascaded LINKED descendants); the loop now drops the
  cache/lock/queue bookkeeping for each, not just the directly-closed session, so a long-lived
  loop that closes subtrees no longer leaks per-descendant state.
- **C5** (`backends/sqlite.py`) — `checkpoint`/`vacuum`/`backup` raise a clear
  "operation on a closed SQLite store" error instead of an opaque driver `ProgrammingError`
  when called after `close()`.

## [2.0.0] — 2026-06-16

**Pluggable storage + an async, stateless, resumable loop.** The store became a
backend-neutral **async** facade (SQLite by default; PostgreSQL/MySQL by DSN) and the
public API went fully async. These are breaking changes to the **STABLE** surface — hence
the major bump. Single-file SQLite stays the zero-infrastructure default; nothing else is
required to upgrade beyond `await`-ing the now-async calls.

### Breaking (STABLE API)

- **The public API is async.** `StatefulAgentLoop` session-management methods that were
  synchronous are now coroutines and must be `await`ed: `new_session`, `close_session`,
  `get_messages`, `get_pending`, `resolve_system_prompt`, `abort_pending`, `schedule_timer`,
  `cancel_timer`, `list_timers`, `get_session_stats`, `list_session_stats` (plus the new
  `prewarm`). Every `SessionStore` method is a coroutine, and `SessionStore.open(...)` is now
  `await SessionStore.open(...)`. `send` / `follow_up` / `resume` / `submit_input` were already
  async; the `send_sync` / `follow_up_sync` wrappers stay synchronous. STABLE symbol *names*
  are unchanged (the SemVer name-guard still passes); the break is the sync→async signatures.
- **Store schema changed — no on-disk migration from 1.x.** Tables now carry a configurable
  prefix (`pl_` by default); the `session_runtime_state` / `shared_state` `key` column was
  renamed to `state_key` (a MySQL reserved word); the SQLite `PRAGMA user_version` ladder was
  replaced by a portable, backend-neutral `pl_schema_migrations` version table. A pre-2.0 `.db`
  is not read by 2.0 — start fresh or `export_session` → `import_session`.
- **Removed the read-only WAL connection pool (`read_pool_size`).** The async SQLite backend
  offloads each statement to a worker thread under a single writer lock; PG/MySQL are natively
  async. Scale by choosing a server backend or sharding SQLite files across processes.

### Added

- **Pluggable storage backends.** One store written once against a tiny async
  `Database` / `Dialect` port, with **SQLite** (default, zero-dependency), **PostgreSQL**
  (`power-loop[postgres]`, asyncpg) and **MySQL** (`power-loop[mysql]`, aiomysql). `open_store(dsn)`
  and `StatefulAgentLoop(dsn=, table_prefix=, schema=)` select the backend by DSN scheme; PG/MySQL
  are real multi-writer servers (per-session seq via `SELECT … FOR UPDATE`). A backend-agnostic
  conformance suite runs against all three. New top-level exports: `open_store`, `SchemaPolicy`,
  `StoreSchemaError`.
- **`SchemaPolicy` provisioning.** `AUTO_CREATE` (default) creates tables if missing; `VERIFY`
  only checks and raises `StoreSchemaError` — whose `.ddl` carries (and prints) the complete
  per-backend provisioning script. `create_schema: bool` kept as a deprecated alias.
- **Stateless, resumable loop + per-session window cache.** The loop holds no authoritative
  session state, so a cold/fresh loop resumes any session from `dsn` + `session_id`;
  `await loop.prewarm(session_id)` pre-loads the active window; an LRU active-window cache
  (`session_cache_size`, default 256, `0` disables; `loop.cache_stats`) accelerates hot paths
  as a pure, validated accelerator that never changes what the model sees.
- New example `39_pluggable_backends_and_resume.py`; a **Storage backends** user-guide page
  (EN + ZH) with the exact per-backend DDL.

### Fixed

- Window cache could serve a stale, row-missing window after an out-of-band durable append
  (`resume` / `submit_input` / `abort_pending` / `heal_pending`, or a second loop sharing the
  store) — fixed with a contiguity guard in the cache; covered by warm-vs-cold regression tests
  (caught by an adversarial code-review pass).
- Carries the post-1.0 deep-review hardening merged on `main` (sandbox/import/workflow-cluster
  fixes + latent-finding guards) and the async `_bind_handler` runtime-env fix + restored
  `max_spawn_depth` validation surfaced during the swap.

### Docs

- README / README.zh rewritten around **loop engineering**, pluggable storage, and
  statelessness; the full EN/ZH user-guide + tutorials swept to the async API; a new Storage
  backends page; an async-API + storage migration note.

## [1.0.0] — 2026-06-16

**First stable release.** The `STABLE` public API is now under SemVer: a break to it
requires a major bump (`2.0.0`), enforced by the frozen-baseline guard. The post-0.14.1
hardening roadmap is complete — durability (0.15), scale (0.16), observability (0.17),
ecosystem (0.18) — and the release-readiness audit's blockers are resolved.

### Changed

- **STABLE API — construction closure (the 1.0 gate).** Promoted the LLM contract into
  `STABLE_API` (frozen): `LLMService`, `LLMRequest`, `LLMResponse`, `LLMStreamChunk`,
  `LLMProviderConfig`, and `create_llm_service_from_env` / `create_llm_service_from_config`.
  The flagship `StatefulAgentLoop` can now be **built, used, and given a custom provider
  using STABLE-only symbols** — previously its mandatory `llm=` collaborator was Provisional,
  making the freeze hollow. Error `.code` strings are now a frozen contract too
  (`test_stable_error_codes_are_frozen`).
- **Post-1.0 SemVer everywhere.** Classifier → `Production/Stable`; CONTRIBUTING / CHANGELOG /
  SECURITY / README / API-reference prose updated from the 0.x "breaks-in-a-minor" model to
  "a STABLE break needs a major bump". Stale doc counts (hooks 18→17, events 24→30) and the
  last "depends only on certifi" onboarding line (the core is zero-dependency) corrected.

### Added

- **Real MCP server test + example.** `StdioMCPClient` is now validated end-to-end against
  a live FastMCP stdio server (`test_mcp_real_server.py` — not just a fake source), and
  `examples/38_mcp_tools.py` shows an agent calling a real MCP server's tool. `mcp` added to
  the dev/CI extras so this runs in CI.

## [0.18.0] — 2026-06-16

迈向 1.0 的硬化路线图**第四阶段(收官):生态/供应链/治理**。MCP 工具适配器、vendored
`llm_client` 溯源、**核心零运行时依赖**、扩展工具手册、`SECURITY.md`、可复现发布流程。
**破坏性变更见 Changed**(移除 `certifi` 基础依赖)。至此 0.15.0(持久化)→ 0.16.0(扩展)→
0.17.0(可观测性)→ 0.18.0(生态)四阶段全部落地。

### Changed

- **（ECO-3)核心零运行时依赖。** 删除 vendored 死代码 `qwen_image.py` + `web_search.py`
  (power-loop 从未导入;`qwen_image` 是 `certifi` 的**唯一**导入者),从 `dependencies` 移除
  `certifi` → 基础依赖集为空。受支持的用法不受影响(transports 经 extras 自带 CA 处理);已用
  import-without-extras + 真机 HTTPS 冒烟验证。
- **（ECO-4)修正过期覆盖率目标。** `--cov=llm_client` → 仅 `--cov=power_loop`(vendored 子包
  随之被统计),ci.yml 与 pyproject 同步;覆盖率门槛 74% > 70%。

### Added

- **（ECO-6)扩展工具手册 + 示例。** `docs/{en,zh}/user-guide/extending-tools.md`:自定义工具配方
  (`ToolDefinition`+handler+`register`)、按调用白名单、把 MCP 作为外部连接器路径、以及"为什么
  不捆绑连接器"的内核理念;新增示例 `37_custom_retrieval_tool`(进程内确定性检索工具)。
- **（ECO-5)`SECURITY.md`。** 支持版本、私密漏洞上报渠道(尽力而为、无 SLA),以及"编排而非隔离"
  的安全模型(内置 bash/file 工具**不是**沙箱 → 用 `ShellBackend`/`SubprocessExecutor`;密钥留在
  编排层;磁盘 SQLite 为明文)。
- **（ECO-7)bus-factor surrogates。** CONTRIBUTING 增加可复现的 `Releasing` 流程;README 增加
  "Used by" 与 "Project status & governance"(单维护者、MIT、可分叉、机器化 API 稳定性守卫)。

- **（ECO-1)MCP 工具适配器。** `contrib/mcp`:无依赖的 `MCPToolSource` Protocol +
  `register_mcp_tools`——把 MCP 工具的 `inputSchema` 直接映射成 `ToolDefinition`(`required`
  驱动缺参校验),注册的异步 handler 把调用代理给 source;外加 `[mcp]` extra 之后惰性导入 `mcp`
  SDK 的默认 `StdioMCPClient`(stdio MCP server)。映射本身可用假 source 测试,无需 `mcp` 依赖。
- **（ECO-2)vendored llm_client 溯源。** `_vendor/llm_client/VENDOR.md`(来源、MIT 许可、
  vendored 日期、本地修改含 0.18.0 的删除)+ `scripts/sync_vendor.sh`(重新 vendor:拷贝、
  裁剪未用模块、改写 import 到 vendored 路径)。

## [0.17.0] — 2026-06-15

迈向 1.0 的硬化路线图**第三阶段:可观测性**。事件信封序列化 + 单调时钟、持久可回放的 JSONL
事件流、事件总线背压、指标(Prometheus/StatsD)与 OpenTelemetry span 桥。全部增量;新后端在
可选 extras 之后惰性导入,核心仍 SDK-free。

### Added

- **（OBS-1）事件信封序列化。** `AgentEvent.to_dict()/from_dict()` 携带 `ts`/`seq`/`mono`,
  作为持久化与外部导出的基础;`from_dict` 对时序字段做**存在性检查**(非真值检查)——序列化的
  `seq` 权威保留,既不重新盖章也不推进进程级 `_event_seq` 计数器。`logging_sink` 现在也输出
  信封序号(`seq`/`ts`),日志行可排序、可与持久事件流对账(此前丢弃信封)。
- **（OBS-6）单调时钟字段。** `AgentEvent.mono`(`perf_counter` 秒,进程相对)用于跨事件延迟/
  span 计算,不受 NTP/墙钟回拨影响(`ts` 仍为可读可导出的墙钟时间)。
- **（OBS-2）持久化 JSONL 事件 sink + 回放。** `attach_jsonl_sink(bus, path, …)` 把完整信封
  (经脱敏/截断)按行写入大小轮转文件;`replay(path)` 跨轮转按 `seq` 顺序还原成 `AgentEvent`。
  脱敏策略抽到共享的 `contrib/_redact`(logging 与 jsonl 复用)。
- **（OBS-3）事件总线背压。** 文档化硬契约:同步订阅者必须快、不可阻塞(否则卡住 agent 循环)。
  新增 opt-in `AgentEventBus(sync_dispatch="thread", queue_maxsize=…, on_overflow=…)`:同步订阅者
  改由后台线程经有界队列消费,`publish()` 立即返回,慢订阅者不再卡循环;队列满按 `on_overflow`
  (`drop_newest`/`drop_oldest`/`block`)处理并计入 `bus.dropped`;`shutdown()` 先排空再停线程。
  默认 `inline`,行为不变。异步订阅者仍调度到事件循环(不下放到无 loop 的线程)。
- **（OBS-4)指标 sink。** `contrib/metrics_sink`:无依赖的 `MetricsBackend` Protocol +
  事件→指标映射(llm 调用/重试、工具调用成败、轮次、错误、token 用量),映射本身不依赖任何
  第三方库(可用假后端测试);出厂 `PrometheusBackend`(`[prometheus]`)与 `StatsDBackend`
  (`[statsd]`)惰性导入各自客户端。
- **（OBS-5)OpenTelemetry span 桥。** `contrib/otel_sink`:把成对的 `*_STARTED`/`*_COMPLETED`
  事件映射成 session→round→llm/tool 的 span 树,接入任意 OTel 后端;在 `[otel]` extra 之后,
  `opentelemetry` 惰性导入(无依赖也可 import 本模块)。`close()` 结束所有未闭合 span。

### Changed

- 新增可选 extras:`prometheus` / `statsd` / `otel`,并并入 `all`;`dev` 增加
  `prometheus-client` + `opentelemetry-sdk` 以便 CI 跑 OBS-4/5 后端测试。

## [0.16.0] — 2026-06-15

迈向 1.0 的硬化路线图**第二阶段:扩展性**。把"推理出的"单进程上限变成"测出来的"(自带 `bench/`
压测台),并在读路径去瓶颈:只读 WAL 连接池、把每次 send 的历史读卸载出事件循环、压缩触发的
token 估算从每轮 O(history) 降为 O(1)。**全部增量,无破坏性变更。**

### Added

- **（SCALE-1）基准/压测台。** 新增**不随 wheel 发布**的 `bench/` 包:确定性 `FakeLLM`(可调人工
  延迟,不打真实 provider)驱动真实 `StatefulAgentLoop`+`SessionStore`,三个场景(FANOUT 并发会话 /
  BIG-HISTORY 大历史 / THROUGHPUT 持续吞吐)产出 JSON 报告(sessions/sec、p50/p99 读写延迟)。
  `python -m bench [--smoke]` 运行;`tests/bench/test_bench_smoke.py` 烟囱测试 + 非阻塞 CI
  (`.github/workflows/bench.yml`)。把"推理出的"单进程上限变成"测出来的",并已暴露 BIG-HISTORY
  的 O(history) 每轮成本(SCALE-4 的目标)。
- **（SCALE-3）卸载每次 send 的活动历史读取。** `_run_loop` 里 `load_active_messages`(O(active-history)
  的 SQLite 读 + 逻辑重排)改走 `asyncio.to_thread`,大会话加载不再卡住事件循环上的其它任务。
- **（SCALE-4）压缩触发的 token 估算从每轮 O(history) 降为 O(1)。** pipeline 维护一个自失效的
  增量 token 估算(append 增量更新;fold/recall/hook 替换历史时失效并重算,**永远等于全量重算**),
  经新增的 `CompactionContext.current_tokens` 交给 compactor 做触发判定,避免每轮重扫全历史
  (实测 5ms@1万 / 26ms@5万 每轮 → O(1))。对自定义 compactor 完全向后兼容(不传则照旧全扫)。

- **（SCALE-2）只读 WAL 连接池(opt-in)。** `SessionStore.open(read_pool_size=N)` 开 N 个额外的
  只读连接(`query_only=ON`),读操作(`load_active_messages`/`load_all_messages`)从池中取连接、
  与唯一写连接并发执行,不再排队等写锁——读密集 fan-out 下显著降低读延迟。写入仍由单写连接+RLock
  串行(正确性不变);WAL 保证池读看到读开始前已提交的全部写入。默认关闭;`:memory:` 自动回退
  (内存库连接不可共享)。含持写锁时池读不被阻塞的并发回归测试。

- **（SCALE-5）扩展性文档 + 示例。** 新增 `docs/{en,zh}/user-guide/scaling.md`:单写模型、读连接池、
  保留/VACUUM、多进程(一文件一进程)模式、调优旋钮,全部基于 `bench/` 实测数据(fan-out 在
  ~1000 sessions/sec 见顶;大历史每次 send 成本随历史线性增长——压缩使其平坦),并诚实标注上限。
  新增示例 `34_durability_lifecycle`(裁剪/VACUUM/导出导入/优雅停机)与
  `35_scaling_and_read_pool`(读池 + 并发会话 + 压测台)。

### Changed

- `CompactionContext` 新增可选字段 `current_tokens`(增量 token 估算提示;附加、向后兼容)。
- `SessionStore.open` / 构造器新增可选 `read_pool_size`(默认 0,行为不变)。
- `bench` fanout 场景支持 `db_path` + `read_pool_size`,以测量 SCALE-2 的读并发收益。

## [0.15.0] — 2026-06-15

迈向 1.0 的硬化路线图（见 `ROADMAP_1.0.md`）**第一阶段:持久化**。把长期存活的磁盘
`SessionStore`(本库的核心卖点)做成生产可用:可随版本升级、可回收空间、可导出归档、可优雅停机。
绝大多数为**纯增量** API;存储层的破坏性变更集中于本版本一次(见 Changed)。真机端到端验证见
`tests/real/test_real_durability.py`。

### Changed

- **（破坏性·存储）schema 版本网关。** `SessionStore.open()` 现在用 `PRAGMA user_version`
  门控:**拒绝打开版本高于本构建的 `.db`** 并给出清晰报错(对手改库者是行为变更)。这是后续一切
  schema 变更的前提,确保 ≤0.14.1 的旧库(`user_version=0`)被识别为 legacy 并经迁移步骤 1 升级,
  而非静默保持旧结构。所有存储层变更都集中在本版本这一次"存储形态拐点"。
- **（破坏性·存储)新建库默认 `auto_vacuum=INCREMENTAL`**(仅影响新建文件的空闲页行为;既有文件
  保持原样,不在 open 时做阻塞式全量 VACUUM)。

### Added

- **（OPS-1）SessionStore schema 版本网关 + 迁移阶梯。** 引入 `PRAGMA user_version` 门控与
  有序、幂等、纯增量的 `MIGRATIONS` 阶梯（`CURRENT_SCHEMA_VERSION` + `_apply_migrations`）。
  `open()` 在建表前探测是否全新库:全新库直接盖章到当前版本,既有库按 `target > user_version`
  顺序跑迁移(单事务,失败回滚则不前进版本),**版本高于本构建的库直接拒绝打开**并给出清晰报错。
  原先硬编码、仅针对 `timers` 的 `_micro_migrate` 收编为迁移步骤 1(对 legacy `user_version=0`
  幂等升级)。这是后续一切 schema 变更的**前提**——没有它,既有 `.db` 在升级时会静默保持旧结构
  (`CREATE TABLE IF NOT EXISTS` 永不改表)。回归测试见 `tests/unit/test_session_store_migrations.py`。
- **（OPS-2）按需保留/裁剪。** `SessionStore` 新增**调用方驱动、绝不隐式**的清理方法:
  `prune_compacted_messages`(删折叠出的 `compacted_out` 原文,保留 `compact_note`/active,
  支持 `older_than_ms`/`keep_recent`,**不可逆**)、`prune_usage_rounds`、`prune_timers`(仅终态)。
- **（OPS-3）空间回收。** 新建库默认 `auto_vacuum=INCREMENTAL`;新增 `vacuum(incremental=…)`
  与 `checkpoint(mode=…)`——配合 OPS-2/`close_session` 真正缩小磁盘文件、回收 WAL。
- **（OPS-4）会话导出/导入 + 整库备份。** `export_session` 把单会话全部持久态序列化为带
  `schema_version` 的 JSON,`import_session` 落到新 id(拒绝更高版本/已存在 id);`backup()`
  走 SQLite 在线备份 API 产出可直接打开的整库快照。支持「先归档再裁剪」与跨库迁移。
- **（OPS-5）优雅异步停机。** `StatefulAgentLoop.aclose()` + `async with`:先拒收新 send,
  再逐个获取 per-session 锁等待在飞 send 落盘完成(修复 `close()` 与 `to_thread` 写竞争导致的
  `ProgrammingError`),drain 待决异步事件订阅者(`AgentEventBus.drain()`),checkpoint 后关库。
  同步 `close()` 保留但标注为非优雅。回归:`test_session_store_retention.py` /
  `test_session_store_export.py` / `test_stateful_loop_aclose.py` + 真机 `tests/real/test_real_durability.py`
  端到端走通「压缩→裁剪→VACUUM→导出→aclose→重开(迁移网关)→导入」全链路。

## [0.14.1] — 2026-06-15

修复一个在 0.14.0 中发现的**高危持久化损坏**（C1 的同进程二次压缩遗漏分支）。纯 bug 修复，
无 Public API 破坏。

### Fixed

- **同一次运行内的第二次压缩会损坏持久化状态（C1 续）。** `compact_note` 被分配一个全新的**高**
  身份 `seq`（来自 `next_seq`），却被放在内存历史的**低**逻辑位置，使 `SQLiteSink._history_seqs`
  这个 index→seq 映射**非单调**。后果有二：
  - **数据损坏**：同一次运行内的第二次折叠把折叠边界经非单调映射翻译后，以
    `from_seq > to_seq` 调用 `record_compaction` → `UPDATE … WHERE seq BETWEEN from_seq AND to_seq`
    在 DB 里**一行都不标记**（内存却照折）→ 内存历史与持久化 active 集**分叉**，并向
    `compactions` 审计表写入倒置的 `(from,to)` 行。
  - **重排错误**：即使只折叠一次，note 因高 `seq` 在重载（`load_active_messages ORDER BY seq`）
    时**沉到 kept 尾部之后**，旧轮次的摘要出现在较新消息之后。
  - **修复**：把**身份**（高 `seq`，保持 append-only 与 recall 语义）与**逻辑位置**解耦。
    `record_compaction` 现在按**显式 seq 集合**（`seq IN (…)`，不再用 BETWEEN 区间，对非单调映射
    免疫）标记折叠行，并把 note 的逻辑位置存为 `meta['ord']`，审计区间用 min/max（不再倒置）。
    `SQLiteSink` 新增并行的 `_history_ord`（逐槽逻辑位置）；`load_active_messages` 改按**逻辑序**
    返回（`compact_note` 按 `meta['ord']` 排，否则按 `seq`）。这条直接修正了 0.14.0 的
    `HARDENING_PLAN` 验收声明“no compacted_out mis-map under recall+compaction”遗漏的同进程二次
    压缩场景——该场景此前**完全无测试**。
  - **回归测试**（`tests/unit/test_compaction_double_fold.py`，11 例）：同一次运行内二/三次折叠
    的内存↔DB 一致性、无倒置审计行、单次/中段折叠后 note 重载位置、reload-then-fold 恢复链路、
    显式集合标记、纯占位符折叠的映射对齐，以及一个 200 例 hypothesis 随机 append/fold 调度不变式
    （reload 必须逐字复现内存 active 历史），外加一个端到端「一次 send 内折叠两次」的真实管线用例。

## [0.14.0] — 2026-06-15

硬化计划 `HARDENING_PLAN.md`（0.13.1 → 1.0）的一次大批量推进：**全部已确认正确性 bug
C1–C8 修复**(各配红前/绿后回归测试) + 新增 H7 压缩联动轨道 + H2 测试/CI 加固(并由此发现并
修复一个真实安全漏报) + H3 打包到 1.0(featherweight 核心、vendored `llm_client`、稳定 API
机器化守卫) + H4 可观测性(每调用 LLM 事件、错误码、日志卫生) + H5 注入接缝修复。
**破坏性变更见 Changed**(`llm_client` 收编 / 安装方式 / 核心依赖)。

### Added

- **（H4.4）机器可读错误码**：每个 `PowerLoopError` 子类带稳定的类级 `code`(点分串,如
  `llm.timeout` / `session.pending` / `tool.not_found` / `spec.invalid`),调用方可按
  `exc.code` 分支而非类身份——重构友好、便于日志/翻译。
- **（H4.5）日志卫生**：`import power_loop` 给包根 logger 挂 `NullHandler`(无 handler 噪声、
  应用未配置即不输出);两处硬编码 logger 名改 `getLogger(__name__)`(全树归于 `power_loop.*`);
  `attach_logging_sink` 新增 `redact_keys`——默认对 `api_key`/`authorization`/`secret`/
  `password`/`*_token` 等密钥名的值脱敏为 `***`(故意不含裸 `token`,避免误伤 `*_tokens` 计数),
  可传 `()` 关闭或自定义。
- **（H4.1）每次 LLM 调用的观测事件**：`call_llm` 现在每个 attempt 发
  `LLM_CALL_STARTED` / `LLM_CALL_COMPLETED`(按 `call_id` 配对),带 round/attempt/model、
  `duration_ms`(perf_counter)、成功/失败 + `error_type`、以及**本次调用**的
  token usage(区别于 `USAGE_UPDATED` 的累计值)——重试因此逐次可见。新增两个 payload
  与枚举值,顶层 re-export。OTel 桥接的基石(配合 H4.2 的 `ts`/`seq`)。
- **（H3.4）顶层 re-export LLM 契约**：`LLMService` / `LLMRequest` / `LLMResponse` /
  `LLMStreamChunk` / `LLMTokenUsage` / `OpenAICompatibleChatConfig` / `AnthropicChatConfig`
  现可 `from power_loop import …`(PROVISIONAL),写 `llm.*` hook 或自定义 `LLMService` 不必再
  伸进内部 transport 包。
- **（H3.6）`STABLE_API` 成为稳定层的单一事实源 + SemVer 守卫**：docstring 不再重复罗列(消除
  三方漂移),`FollowUpQueued` 归入 STABLE;新增测试校验 STABLE_API 与 `__all__`/模块属性一致,
  并冻结 v0 基线——未升 major 删除/改名 STABLE 符号即测试失败。
- **（H2 测试/CI 加固）覆盖率门禁 + 严格 marker + 示例冒烟 + 免-extras 导入腿**：
  CI 的 pytest 现在跑 `--cov=power_loop --cov=llm_client --cov-fail-under=70`(当前 72.6%);
  pytest `addopts` 加 `--strict-markers --strict-config`(typo marker / 未知 ini key 直接报错);
  新增 `tests/unit/test_examples_smoke.py`——逐个 import 全部 35 个 `examples/NN_*.py`,public API
  改名会让 CI 立刻红(语义校验仍留夜间真实 LLM);新增 CI job `import-without-extras`——只装核心
  (不装 `[openai]`/`[anthropic]`)后 `import power_loop` + 跑惰性导入测试,守住「零 SDK 可导入」。
  dev 依赖加 `pytest-cov` / `hypothesis`。
- **（H7 Phase 2）`Compactor` 协议加可选 `CompactionContext`——折叠前可联动记忆**：
  `maybe_compact` 现在可**选**接收 `context: CompactionContext`(暴露注入的
  `MemoryProvider` + `session_id` + 只读 `fetch_messages`),自定义压缩器可在折叠前把要点
  `remember` 进记忆,跨 session 留存。**向后兼容**:pipeline 按签名判断,只对接受 `context`
  的压缩器传(老签名压缩器照常工作);`DefaultCompactor` 忽略它,行为不变。新增
  `power_loop.runtime.compact.CompactionContext`(PROVISIONAL)。单测:签名内省门、
  context-aware 压缩器收到完整 context、**老签名压缩器仍可用**(两方向红前/绿后);真实 LLM
  示例 `examples/33_coordinating_compactor.py`(折叠时捕获的事实跨新 session 经 recall 存活)。
- **（H7 Phase 1）`recall_compacted` 工具——按需取回被压缩折叠的细节**：压缩把旧消息折叠成
  `compact_note` 并标 `compacted_out`,但原文**没删**(仍在 store 里)。新默认工具
  `recall_compacted(query?, from_seq?, to_seq?, limit?)` 让 agent 在摘要缺具体细节时把原文
  捞回来——**只读、仅当前会话**、按关键词/seq 过滤、按 `limit` 取最近若干条。属 `full` preset
  (也可 `include=["recall_compacted"]` 单挑)。8 个单测(过滤/空/会话隔离/排除 active/截断)+
  真实 LLM 示例 `examples/32_recall_compacted.py`(把编码埋进被折叠的轮次,用极小 summary 预算
  逼出工具调用)+ en/zh 文档。设计见 `docs/compaction-coordination-design.md`(H7 轨道)。
- **（H4.2）`AgentEvent` 增加 `ts` + 单调 `seq` 信封字段**：每个事件自动盖上墙钟时间与
  进程内单调序号（`itertools.count`，CPython 原子），从而可时间戳化、可全序化——这是
  OTel span 桥接与重建交错子代理/工作流事件流的基石。两字段排除在相等性之外，不定义事件身份。
- **（H3.3）发布 PEP 561 `py.typed` 标记**：下游 mypy/pyright 现在能看到 power-loop 的类型
  注解（此前整套带注解的 Public API 对类型检查器不可见）。
- 新增回归测试：`reap_runs` 并发 unlink、eager_wake 失败重挂、journal 终态冻结、
  AGENT_ERROR 终结事件、事件 `ts/seq`、provider 惰性导入、py.typed 装车、以及
  **默认 OpenAI 流式 transport 的单元测试**（`tests/unit/test_openai_transport.py`，
  把默认 provider 从仅夜间真实 LLM 覆盖提升为每-PR 覆盖，H2.2）。

### Changed

- **（H3.2，打包）`llm_client` 收编进 `power_loop._vendor`**：wheel 不再发布一个裸的顶层
  `llm_client` 包(消除与他人 PyPI 包/本地模块的命名抢注/冲突风险)——`top_level.txt` 现在只有
  `power_loop`。内部引用改走 `power_loop._vendor.llm_client.*`(包内仍用相对导入,无需动)。
  **若你此前直接 `from llm_client.interface import …`**:改用顶层 re-export
  `from power_loop import LLMRequest, LLMResponse, …`(H3.4);工厂类属内部,改用
  `create_llm_service_from_config` / `create_llm_service_from_env`。
- **（H3.5，安装方式）核心依赖瘦身为仅 `certifi`**：`socksio`(从未直接 import,httpx 按需
  传递)移除;`python-dotenv`(仅 examples/tests)移入 dev;`pyyaml` → `[skills]` extra
  (缺失时 `load_skill` 优雅降级、不报错)、`pypdf` → `[pdf]` extra(PDF 输入,懒加载);
  `[all]` 现含两家 transport + skills + pdf。删除 `requirements.txt`(pyproject 为单一事实源)。
  classifier 升 `4 - Beta`(H3.7)。
- **（H3.1）transport 惰性导入 + 可选 extras**：`anthropic` / `openai` 从硬依赖移入
  `[project.optional-dependencies]`；`power_loop.runtime.provider` 仅在真正构造对应 provider
  时才导入其 SDK。`import power_loop` 现在零 SDK 即可成功（featherweight 名副其实）。
  **安装方式变化**：请改用 `pip install 'power-loop[openai]'` / `[anthropic]` / `[all]`；
  缺失所选 SDK 时构造 provider 会抛出带安装提示的清晰 `ImportError`。README 同步更新。

### Fixed

- **（H5.1）绑定默认工具注册表会遮蔽外层注入的 `ShellBackend`/`Blackboard`**：`bind=True` 时
  `_bind_handler` 在调用期把 runtime_env 重置为「仅路径」快照(`shell_backend=None`),悄悄
  defeats 宿主在 `runtime_env_context(shell_backend=sandbox, …)` 里注入的沙箱/board。改为调用期
  **合并**:绑定的是路径(workspace/home/skills),而 ShellBackend/Blackboard 继承外层上下文
  (注册表若显式设置则其优先);`create_default_tool_registry` 新增 `shell_backend`/`blackboard`/
  `blackboard_id` 参数。
- **（H1.10 / C12）`close_session` 不清理 per-session 内存锁**：长生命周期 loop 轮换大量 session
  会按 session id 泄漏 `asyncio.Lock`。`close_session` 现在 pop 掉 `_locks` /
  `_follow_up_queue_locks` / `_follow_up_queues` 三个字典的对应键。
- **（H2.4 / C14，安全）`bash` 危险命令守卫漏过 `rm -rf /<系统目录>`**：写安全分支测试时发现
  `_dangerous_command_reason` 的 rm 正则只匹配**裸**根/家目录(`/` / `~` / `$HOME` 且后接
  空白或行尾),`rm -rf /etc`、`rm -rf /usr/local`、`rm -rf /var/lib` 这类**子路径删除全部漏过**
  (真实 false-negative)。改正则:阻断根/家目录及 `/(bin|boot|dev|etc|home|lib|opt|proc|root|run|sbin|srv|sys|usr|var)`
  系统目录(含子路径与 `~/…` / `$HOME/…`),同时仍放行 `/tmp` 与相对路径;flag 组可重复(`rm -r -f /x` 也拦)。
  新增 `tests/unit/test_bash_guards.py`(37 例):各阻断/放行命令 + `_validate_bash_command_scope`
  的家目录读写/allowlist/越界全覆盖(此前零覆盖)。
- **（H1.9 / C8）同步 SQLite 写阻塞事件循环 → 多会话互相拖死**：pipeline 的写路径
  store/sink 调用同步执行,某会话一次有竞争的写(`busy_timeout` 最高 5s)会卡住整个
  事件循环、拖住其它所有会话。修复:把**写路径** sink/store 调用(`on_message_appended` /
  `on_compaction` / `on_round_started` / `on_assistant_tool_calls` / `on_round_ended` /
  `bump_session_stats`)用 `asyncio.to_thread` 下放到线程(RLock 已保证线程安全);**读保持内联**
  (快、少竞争)。`NullSink` 无 I/O,跳过线程跳转零开销。新增 `tests/unit/test_store_offload.py`:
  阻塞写期间并发 ticker 仍推进(红前 ticker≈停摆 / 绿后照常);`StatefulAgentLoop` 并发文档
  同步更正。
- **（H1.7 / C6）同步 `publish()` 静默吞掉 async 订阅者异常**：有运行中的 loop 时,async
  handler 被 `loop.create_task` fire-and-forget,`suppress_subscriber_errors=False` 的
  re-raise 发生在脱离的 task 里,只剩一条 "Task exception was never retrieved" 的 GC 告警;
  且 task 未被引用,可被 GC。修复:保留 task(防 GC)+ done-callback 取回异常——未抑制时
  在 ERROR 级别大声记录(async 订阅者要内联处理异常请用 `publish_async`)。新增
  `tests/unit/test_event_bus_async.py`(保留→排空、抑制吞掉、未抑制记 ERROR;红前/绿后)。
- **（H1.2 / C2）`parallel`/`foreach` 在 `on_error="halt"` 下不取消在飞的兄弟分支**：
  `asyncio.gather(return_exceptions=False)` 首个失败即 re-raise,但**不取消**其它仍在跑的
  分支——它们继续烧真实 LLM 调用,迟到的 `record_step` 还能污染已 finalize 的 journal。新增
  `WorkflowEngine._gather_branches`:halt 时首个失败即 `task.cancel()` 其余兄弟、置
  `self._cancelled`、best-effort 翻 `self._cancel`,排空后再 re-raise;`continue` 行为不变。
  (journal 污染那半已由 H1.3 终态冻结堵住。)新增 `tests/unit/test_workflow_fanout.py`:
  parallel/foreach halt 取消兄弟 + 无遗留任务 + continue 仍收集全部错误(红前/绿后)。
- **（H1.1 / C1，最高严重度）记忆召回与压缩的 `_history_seqs` 错位 → 压缩标错 DB 行**：
  `_maybe_recall` 把 `memory_*` 消息直接插进 `pipeline.history`（绕过 sink），使
  `sink._history_seqs` 与 history 错位 `len(recalled)`，随后 `on_compaction` →
  `record_compaction` 按错位索引把**错误的行**标 `compacted_out`（静默、持久、会级联）。
  修复：新增 `sink.on_messages_inserted`，召回时为每条非持久化消息插入占位以保持
  index↔seq 对齐；并给 `on_compaction` 加 `expected_history_len` 对齐安全网——一旦
  映射失准（如 `SESSION_START` hook 整体替换历史，C9），**跳过本次压缩持久化**而非
  标错行（内存折叠照常，active 行不动，resume 仍正确）。新增 `examples/31_memory_with_compaction.py`
  （真实 LLM）+ 跨「有/无召回」等价性回归测试。
- **（H1.5）未捕获异常逃逸时既无 `SESSION_ENDED` 也无错误事件**：`pipeline.run()` 中
  raise 的 hook / sink / store I/O 直接抛出，看过 `SESSION_STARTED` 的订阅者被悬挂，且
  「文档声称会发」的 `AGENT_ERROR` 通道实为死代码。现在在调用点捕获 → 发 `AGENT_ERROR` +
  终结 `_finalize("error")`（`SESSION_ENDED`）→ 原样 re-raise；`_finalize` 改为幂等。
- **（H1.4）`eager_wake` 触发未跟踪的 follow_up 任务，可被 GC → 永久丢失父唤醒**：claim
  woke 后用裸 `create_task` 触发、句柄丢弃（CPython 仅持弱引用），且 woke 已 claim 会压制
  durable timer。现在保留任务引用，失败/取消时经 done-callback 重开 woke 并重挂 durable
  timer，父代理仍恰好被唤醒一次。
- **（H1.3）finalized journal 的迟到写回退**：孤儿叶子（`on_error="halt"`，见 H1.2）在 run
  终态后的 `record_step`/`update` 会用陈旧整 blob 把 `status` 退回 `running`、`result` 置空。
  journal 达到终态后冻结 status/result/steps（`record_step` 在写前重读最新 blob 并合并）；
  唤醒/resume 等正当写入用 `allow_terminal=True` 显式放行。
- **（H1.6）`reap_runs` 遇并发 unlink 会中止整轮 GC**：未保护的 `f.stat()` 在 worker /
  `delete_on_success` 并发删 db/WAL 时抛 `FileNotFoundError`，使后续所有 run 目录漏回收。
  改为逐文件 + 逐目录吞 `OSError` 并继续（对齐 `_remove_db`）。

## [0.13.1] — 2026-06-15

修复版本：一次系统性核心能力 bug 审计（5 个并行 agent）发现的全部 16 个问题，
逐一修复并补回归测试。无 Public API 破坏性变更，纯修复 + 一个新示例。

### Fixed

- **（BLOCKER）压缩越界孤儿 `tool` 消息**：`_compactable_span` 会把折叠终点回退越过
  尾部 `tool` 消息，在“工具回合后用户继续说话”时留下没有配对 assistant 的孤儿
  `tool`，导致下一次请求 HTTP 400。移除该回退逻辑（边界已由 `_expand_back_to_atomic`
  保证），旧测试方向写反（误判通过）也一并改正。
- **（CRITICAL）`SessionStore` 非原子写**：sqlite 以 `isolation_level=None`（autocommit）
  打开，`with self._conn:` 从不真正开启事务 → 多语句写入（如 `append_message` 推进
  `next_seq`）非原子。改为延迟事务（`isolation_level=""`）。
- **轮次上限收尾绕过统一调用路径**：达到 `max_rounds` 的总结调用直接 `llm.complete`，
  绕过取消 / 重试 / 超时 / 每-loop 模型 / 流事件。改为走 `call_llm`，并加预调用取消检查点
  与同主循环一致的 degrade 处理。
- **流事件不配对**：LLM 调用失败 / 重试耗尽 / 取消时只发了 `STREAM_STARTED` 没有终结
  事件，订阅者悬挂。`STREAM_COMPLETED` 改到 `finally` 发出，必然配对。
- **重试退避不可取消、且会溢出**：退避 `asyncio.sleep` 不响应取消（与文档承诺相悖）→
  改为分片轮询 token 的 `_cancellable_sleep`；`2**(attempt-1)` 对超大 `max_attempts`
  会在 `min()` 前就 `OverflowError` → 先把指数 cap 在 32。
- **定时器 stale 恢复重复触发**：一次比 `stale_firing_s`（默认 120s）更久的“live”投递
  会被周期性恢复扫描重新 arm 并二次触发。新增 `SessionStore.heartbeat_firing_timer`，
  投递期间后台心跳持续刷新 `firing` 行（节流为 stale 窗口的 1/4，可经新构造参数
  `heartbeat_interval_s` 覆盖）。
- **`grep` 在 rg 与 Python 回退路径下结果分叉**：rg 仅排除少数目录、且 root-anchored、
  又排在 include glob 之前（会被覆盖）；改为对每个 `_COMMON_SKIP_DIRS` 生成
  `!**/<dir>/**`（任意深度）并置于 include glob 之后（rg 后者优先），与回退一致。
  顺带修正截断计数（按实际展示的非空行计数）。
- **结构化输出**：`_extract_first_json_object` 遇到首个 `{` 之前的游离 `}` 会让深度变负
  从而拒绝合法 JSON；忽略深度 0 处的 `}`。文档漂移修正：声称的“单引号修复”代码从未
  实现（正则也无法安全实现），文档对齐为保守的尾逗号修复。
- **工作流**：`from_json` 接受重复 agent id（resume 时会重放错节点）→ 拒绝重复；
  `eager_wake` 会重复唤醒父 agent（绕过 `TIMER_FIRE` 使 wake-guard 无法去重）→ eager
  路径先 claim journal `woke`。
- **早退工具循环留下悬挂 `tool_calls`**：`TOOL_AFTER` BREAK、以及 `request_user_input`
  与其它工具同批时，未执行的工具调用没有配对 `tool` 结果 → 非法序列 / 幻影 pending；
  新增 `_resolve_skipped_tool_calls` 补合成 `[skipped]` 结果。

### Added

- **示例 `28_docker_shell_backend.py`**：通过 `ShellBackend` 缝把内置 bash 换成
  `docker exec`，模型写的 shell 在隔离容器内执行（真实 LLM 验证）。

## [0.13.0] — 2026-06-14

### Added

- **动态工作流（`power_loop.workflow`，可选子模块）**：声明式 `WorkflowSpec` JSON
  DSL（`agent`/`sequence`/`parallel`/`foreach`/`branch`，创建即严格校验、问题一次性聚合），
  确定性 in-process 引擎，叶子是普通子代理。
  - **detached 执行 + 完成回调唤醒主 agent**（`run_detached` + `register_wake_guard`，
    经 durable timer → `follow_up`），`SharedBudget` 跨子代理 token 池，
    LLM-facing `create_workflow` / `workflow_status` 工具。
  - **跨进程重启的编排级 resume**（`resume_run` / `resume_detached`）：journal 持久化
    spec + 每步 text/payload，重放已完成步、只重跑未完成尾；`foreach` 以 aggregate 原子
    重放；幂等 key（`run_id:node_id`）注入叶子 metadata。
- **进程外执行器（subprocess executor）**：`run_spec_isolated` + `WorkerBootstrap`
  （每个子代理独立 SQLite 库，依赖只从配置重建）、`SubprocessExecutor`（每叶一进程，
  插现有 `Executor` 缝；取消=SIGTERM→SIGKILL、超时/崩溃→failed→resume 重跑、
  子库保留/GC `cleanup_run`/`reap_runs`）、**`WorkerLauncher` 缝**（按叶子注入
  runsc/docker 等进程级沙箱，fail-closed）。
- **作用域共享黑板（`runtime.blackboard`）**：`Blackboard` 异步 Protocol + 默认
  `SqliteBlackboard`（新 `shared_state` 表，append + 按条目更新/删除）、`RuntimeEnv`
  新增 `blackboard`/`blackboard_id` 注入缝、通用 `board_*` 工具
  （`register_blackboard_tools`，kinds/statuses 由宿主定策略）。
- **config 可选离线 echo provider**（`provider="echo"`）：确定性、无网络，便于子进程/
  集成测试。

### Changed

- `AgentSpec` 新增 `output_schema`（→ provider `response_format` + `parse_structured`）；
  `AgentLoopConfig` 新增 `model` / `response_format`（**每子代理/工作流步可覆盖全局模型**）。
- `run_agent_spec` 现在**转发 `stop_event`**（协作式取消子代理）、**surface `result.usage`**，
  并发布此前一直未接线的 `SUBAGENT_*` 生命周期事件（带 `AgentEvent.source="subagent"`）。
- `MAX_SPAWN_DEPTH` 由硬常量改为**每 store 可配**（`SessionStore.open(max_spawn_depth=)` /
  `StatefulAgentLoop(max_spawn_depth=)`，默认仍 3）。
- `SessionStore` 新增 `shared_state` 表与 `get/set/delete_shared_state`（owner-keyed JSON，
  不绑定 session）。

## [0.12.0] — 2026-06-12

### Added

- **周期性定时任务（一等语义，创建时声明）**：`timers` 表新增 `interval_s` /
  `fire_count` / `last_fired_at`（旧库自动微迁移补列）。
  - `interval_s IS NULL` = 一次性（`firing → fired`）；设置 = 每次投递后
    `firing → armed`，`due_at = 触发时刻 + interval`（**fixed-delay**：
    停机期间漏掉的周期坍缩成一次，不会补发风暴）；`cancel` 是周期任务唯一出口。
  - 工具 `schedule_wakeup` 新增可选 `every_seconds`；`loop.schedule_timer`
    新增 `interval_s`；`list_wakeups` 显示周期与已触发次数。
  - hook 语义随之落位：SKIP = 跳过本次、周期照常排下次；BREAK = 终止整个周期。
- **真实 LLM 测试** `tests/real/test_real_timers.py`：模型自排唤醒并在被叫醒后
  执行 note（暗号复述）；周期 timer 连续两次真实投递后 cancel 终止。

## [0.11.0] — 2026-06-12

### Added

- **持久定时唤醒（durable timers）**：store 新表 `timers`（timer = 数据非任务，
  跨重启存活，随 session 级联删除）。
  - Agent 侧默认工具：`schedule_wakeup(delay_seconds, note)` /
    `list_wakeups` / `cancel_wakeup(timer_id)` / `current_time`（不在任何
    preset 里，按需 `get_tool_definitions(include=[...])` 注册）。
  - 宿主侧 API：`loop.schedule_timer(sid, delay_s=|due_at_ms=, note=)` /
    `loop.cancel_timer` / `loop.list_timers`（与工具写同一批行）。
  - **`TimerRunner(loop)`**：进程内扫描器——`start()` 时回收 stale `firing`
    行（at-least-once，可能二次投递），到期 CAS 认领后经 **`follow_up`**
    投递（空闲 = send，运行中 = 轮边界注入；进会话只有一条路）。
    不启动 runner（或外部调度器轮询 `store.due_timers()`）则永不触发。
  - **`HookPoint.TIMER_FIRE`** + `TimerFireCtx`：投递前编排否决点——
    CONTINUE 投递 / SKIP 跳过 / BREAK 取消 / `postpone_s` 改期，可改写
    投递文本；无 hook 默认投递。
  - 事件 `timer_fired`（`TimerFiredPayload`，outcome: delivered / queued /
    skipped / cancelled / postponed / error）。
  - `loop.hooks` / `loop.event_bus` 公开只读属性。
- 新示例 `examples/26_timers.py`；新增单测 `tests/unit/test_timers.py`（9 个）。

## [0.10.0] — 2026-06-12

### Added

- **`AgentLoopConfig.max_tokens_per_run`**：per-run 真实 token 预算护栏。轮边界
  检查（越界的那一轮完整结束，不留未决 tool_calls），命中后 status =
  `budget_exceeded`（新 LoopStatus 值），发 `status_changed`
  （`BudgetExceededStatusPayload`，kind="budget_exceeded"）。默认关闭。
- **Session 统计**：store 新表 `session_stats`（每次 send 结束累加一次：sends /
  rounds / llm_calls / tool_calls / prompt / completion / total tokens /
  first_send_at / last_send_at），随
  `close_session` 级联删除；新 API `loop.get_session_stats(sid)` /
  `loop.list_session_stats()`（`SessionStatsRow`）。注意 `usage_rounds` 表按
  (session, round_index) 覆盖、跨 send 不可累计——记账请用 session_stats。
- **`power_loop.contrib.logging_sink.attach_logging_sink(bus)`**：标准结构化
  日志 sink，每个事件一行 JSON（stdlib-only，长字段截断），消灭每个接入方
  重写"事件→日志"胶水的重复劳动。

- `AgentLoopResult.tool_calls` / `StatefulResult.tool_calls`：本次 run 执行的
  工具调用次数（`ContextManager.tool_calls` 计数）。

### Changed

- **同步工具 handler 现在跑在工作线程**（`asyncio.to_thread`，contextvars 正常
  传播）：慢的同步工具不再阻塞事件循环和同进程的其它 session。需要留在事件循环
  线程的 handler 请改 `async def`。
- `@phase` 装饰器发布的 start/end 事件改为携带 `PhaseEventPayload`（typed
  `data`）——至此**所有**内部事件发射路径都保证 `event.data` 非 None（新增契约
  测试）。
- README：补「一个 store 文件 = 一个进程」的多进程边界声明（跨进程并发安全
  暂不实现，先文档约束）。

## [0.9.0] — 2026-06-12

### Added

- **`StatefulResult.usage` / `AgentLoopResult.usage`**：每次 `send`/run 返回累计
  token 用量（对该 run 的全部 LLM 调用求和：`prompt_tokens` / `completion_tokens` /
  `cache_read_tokens` / `reasoning_tokens` / `total_tokens` / `calls`）。此前只能
  订阅 `usage_updated` 事件自行累加（事件是单次调用量、覆盖式），编排方做成本
  记账需要自建 tracker——现在直接读返回值。
- **`ContextManager.usage_totals`**：`update_usage` 在保持 `token_usage`（末次
  调用）语义不变的同时累计总量。
- **`send(..., heal_pending=True)`**（含 `send_sync`）：session 因上一个 run
  被杀死在 tool-call 中途而带有未决 `tool_calls` 时，自动 `abort_pending` 后
  继续本次 send，不再抛 `SessionPendingError`。默认仍为 raise（自愈会丢弃
  未完成的工具结果，应由调用方显式选择）。
- `SessionPendingError` 报错信息补充三条恢复路径指引（`resume` /
  `abort_pending` / `heal_pending=True`）。
- 文档：README 增加 token 记账与 heal_pending 小节、明确「无内置定时器」的
  范围边界；events 文档明确 `usage_updated` 为单次调用量、整 run 总量应读
  `result.usage`、handler 中 `event.payload` 按 dict 取值。
- 新示例 `examples/25_token_usage.py`；新增单测 `tests/unit/test_usage_and_heal.py`。

### Notes

- 本版本由 DeepTalk 多 agent 编排落地反推：token 成本面板需要 per-run 用量、
  人类中断 run 后 session 被未决 tool_calls 卡死，是两处真实暴露的不足。

## [0.8.1] — 2026-06-11

### Fixed

- `SQLiteNoteMemory.remember` 的签名对齐 `MemoryProvider` 协议（keyword-only
  `snapshot=` / `session_id=`）；0.8.0 的位置参数签名会在 session 结束时触发
  `MEMORY_FAILED`（软失败，不影响回复，但有噪音日志）。

## [0.8.0] — 2026-06-11

### Added

- **Agent-authored notes（自主记忆）**：模型用工具自己维护的持久笔记，存在 session store 新增的
  `notes` 表里（按 session 隔离，`close_session` 级联删除）。
  - 新默认工具 `note_add` / `note_update` / `note_delete`（进入 `full` preset；`core`/`explore` 不含）。
  - `SQLiteNoteMemory`：实现 `MemoryProvider` 协议，`recall()` 把该 session 的笔记渲染成一条
    `memory_notes` system 消息每次 send 注入；`remember()` no-op（写入由工具实时完成）。
  - `NotesPolicy(max_notes=50, max_note_chars=1000, inject_max_chars=8000, eviction="reject")`：
    默认**拒绝式**容量——满了报错并提示模型先删/合并（静默遗忘是 agent 记忆最糟的失败模式）；
    `eviction="fifo"` 切换为队列式淘汰（pinned 永不被自动淘汰）。注入超预算时优先隐藏最老的
    未 pin 条目并在文本中声明隐藏数量。
  - `AgentLoopConfig.notes_policy` 字段；顶层导出 `NotesPolicy` / `NotesFullError` /
    `SQLiteNoteMemory` / `DEFAULT_NOTES_POLICY` / `render_notes`。
  - `SessionStore` 新方法：`add_note` / `update_note` / `delete_note` / `list_notes` / `count_notes`，
    `NoteRow` dataclass。旧数据库自动建表，完全向后兼容。
  - 新示例 `examples/24_agent_notes.py`；单测 `tests/unit/test_notes.py`（18 例）。

## [0.7.2] — 2026-06-11

### Fixed

- Export `runtime_env_context` from the top-level package so the documented `bind=False` flow works without an internal import.
- Forward `tools=` and `system_prompt=` through `send_sync()` and idle `follow_up_sync()` as well as the async APIs.

### Docs and tests

- Document per-call overrides, unbound registries, and `ShellBackend` in the English and Chinese guides/API reference.
- Extend example 23 with a real unbound-registry invocation across two runtime workspaces.
- Add regression coverage for sync overrides and runtime resolution of unbound handlers.

## [0.7.1] — 2026-06-11

### Docs

- New **example 23** (`examples/23_per_send_overrides.py`) demonstrating per-call `tools=` allowlisting and `system_prompt=` override, plus a README "Per-call overrides" section. No code changes vs 0.7.0.

## [0.7.0] — 2026-06-11

### Added — Per-call overrides & cleaner public surface

- **`StatefulAgentLoop.send(..., tools=, system_prompt=)`** and **`follow_up(..., tools=, system_prompt=)`** — per-call overrides that do not mutate loop/session state. `tools` accepts a sequence of tool names (allowlisted from the loop registry) or a `ToolRegistry`; the model only *sees* the permitted subset. `system_prompt` overrides for that run only (precedence: per-call > session > config). Enables multi-tenant reuse of one cached loop without runtime hook gating.
- **`ToolRegistry.subset(names)`** and **`ToolRegistry.names()`** — derive a restricted registry.
- **`create_default_tool_registry(..., bind=False)`** — build an **unbound** registry whose handlers read the current `RuntimeEnv` at call time (caller supplies it per call via `runtime_env_context`); no eager workspace requirement. `DEFAULT_TOOL_HANDLERS` is now part of the public API.
- **`ShellBackend.session_key(workspace_dir)`** — the persistent `BashSession` is now cached by the backend's execution-target key, so swapping backends (e.g. local ↔ sandbox, or distinct sandbox containers) no longer needs ad-hoc rebuilds.

### Fixed

- Follow-up dropped on a terminal round (0.6.0) — see below; plus `__init__` export/lint hygiene (`FollowUpQueued`, `DEFAULT_TOOL_HANDLERS` now exported).

### Docs

- README: explicit **"orchestration, not isolation"** scope note — built-in `bash`/file tools run in-process and are not a security boundary; sandbox via the `ShellBackend` seam.

## [0.6.0] — 2026-06-11

### Fixed — Follow-up on a terminal round

- A follow-up enqueued during an otherwise-terminal round (model returned a final answer with no tool calls) was never drained — the queue only drained at the *next* round start, which never came. The loop now drains pending follow-ups before completing and runs another round to address them, so absorbed steering input is always processed.

## [0.5.0] — 2026-06-11

### Added — Pluggable shell backend

- **`runtime.exec_backend`** (`ShellBackend` protocol, `LocalShellBackend`, `DEFAULT_SHELL_BACKEND`) and **`RuntimeEnv.shell_backend`** — host code can route the persistent shell into an isolated sandbox (e.g. `docker exec`) instead of an in-process `/bin/bash`, without changing tool implementations. Default behavior unchanged.

## [0.4.1] — 2026-06-08

### Added — In-flight steering (`follow_up`)

- **`StatefulAgentLoop.follow_up()` / `follow_up_sync()`** — enqueue steering input while a session run is in flight; idle sessions degrade to `send()`.
- **`FollowUpQueued`** — immediate return shape when input is queued for the next pipeline round.
- **Round-boundary drain** — merged follow-ups append as a wrapped `<follow_up>` user message before `prepare_round`.
- **Example 22**, bilingual docs, and unit/real tests for the steering path.

### Added — M2.8 Anthropic Messages API 传输（2026-06-06）

- **`AnthropicMessagesLLMService`**（`llm_client.anthropic_factory`）—— 新增原生 Anthropic Messages API transport，复用统一 `LLMRequest` / `LLMResponse`。
- **`LLMProviderConfig.provider` 成为路由键**：`provider="anthropic"` / `"claude"` / `"dashscope-anthropic"` 使用 Anthropic transport；其它 provider 仍使用 OpenAI-compatible transport。
- **消息转换**：OpenAI-style `tool_calls` → Anthropic `tool_use` blocks；`role="tool"` → `tool_result` blocks；返回的 `tool_use` 统一转回 `LLMResponse.tool_calls`，pipeline 无需分支。
- **测试配置**：real LLM helper 改为 `create_llm_service_from_env()`，支持 `POWER_LOOP_*` 与 legacy `OPENAI_COMPAT_*` 两组环境变量。
- **版本**：`power_loop.__version__ = "0.4.0"`。

### Public API（M2.8 新增）

`AnthropicChatConfig` / `AnthropicMessagesLLMService` 可从子模块导入；顶层 `LLMProviderConfig` 的 `provider` 字段现在影响 transport 路由。

### Changed — M2.7 显式 Session 创建（2026-06-06）

- **`StatefulAgentLoop.new_session(metadata=None, system_prompt=None) -> str`** —— 新增显式会话创建入口。调用方先拿到 `session_id`，再传给每次 `send()` / `send_sync()`。
- **Breaking**：`StatefulAgentLoop.send(user_input, session_id, *, stop_event=None)` 与 `send_sync(...)` 现在必须传入 `session_id`；不再在首次 `send()` 时隐式创建 session。
- **Breaking**：`metadata` 从 `send(metadata=...)` 移到 `new_session(metadata=...)`。这样会话级信息在会话创建时固定，避免首条消息和会话生命周期耦合。
- **文档 / 示例 / 测试**：README、双语 docs、examples、unit/real 测试全部改为 `sid = loop.new_session(); await loop.send(..., session_id=sid)`。
- **版本**：`power_loop.__version__ = "0.3.0"`。

### Public API（M2.7 变更）

`StatefulAgentLoop.new_session` 顶层入口新增；`StatefulAgentLoop.send / send_sync` 签名破坏性变更，`session_id` 必填。

### Added — M1.1 LLM 重试 / 超时 / 取消（2026-06-05）

- **`LLMRetryPolicy`**（`power_loop.runtime.retry`）—— 配置 `max_attempts` / `backoff_initial` / `backoff_max` / `total_timeout` / `retry_on`。指数退避（capped），跨所有 attempt 共享总超时；退避 sleep 是 cancel-aware 的（cancel 触发时不会傻等到底）。
- **`with_retry(call, *, policy, token, on_retry=None)`** —— 库内通用 helper，pipeline 用它包 `await self.llm.complete(...)`。``CancellationRequested`` / ``asyncio.CancelledError`` 直接透传，不会被吞。
- **`CancellationToken`**（`power_loop.runtime.cancellation`）—— 统一 cancel 形状：`from_any(...)` 接受 `asyncio.Event` / `threading.Event` / `Callable[[], bool]` / 已存在的 token / `None`。自带 owned 模式（`token.cancel(reason)`），供 hook `HookDirective.CANCEL`（M1.5）和外部 controller 使用。`is_cancelled()` 对用户 callable 抛出做容错（**绝不让 cancel 检查本身污染主循环控制流**）。
- **`AgentLoopConfig.retry_policy: LLMRetryPolicy | None = None`** —— 默认 None（保持现有 fail-fast 行为）；显式赋值即开启。
- **新事件**：
  - `AgentEventType.LLM_RETRY_ATTEMPTED` + `LlmRetryAttemptedPayload(attempt, max_attempts, error_type, error_message, next_sleep_seconds)`
  - `AgentEventType.LLM_DEGRADED` + `LlmDegradedPayload(reason, attempts, error_type, error_message)` —— `reason ∈ {"retry_exhausted", "timeout"}`
  - `AgentEventType.LOOP_CANCELLED` + `LoopCancelledPayload(reason, round_index)`
- **新错误**（`power_loop.contracts.errors`，全部 `PowerLoopError` 子类）：
  - `LLMTimeout(elapsed, attempts, total_timeout)`
  - `LLMRetryExhausted(attempts, last_error)`（`__cause__` 保留 last error）
  - `CancellationRequested(reason)`
  - `CompactionFailed`（M2.5 占位）
- **`LoopStatus`** 新增 `"degraded"`。Pipeline 在 `call_llm` 抛 `LLMRetryExhausted` / `LLMTimeout` 时：append 一条合成的 `assistant` 消息（`[degraded: …]`），emit `LLM_DEGRADED`，`status="degraded"` 返回。`CancellationRequested` 翻译为 `status="cancelled"` + `LOOP_CANCELLED`。
- **Pipeline 内部统一**：`stop_event` 仍接受任意 cancel-like 对象（API 向后兼容），但内部统一存为 `CancellationToken`；`StatefulAgentLoop.send / send_sync / resume` 的 `stop_event` 类型放宽为 `CancellationLike`。
- **测试**：`tests/unit/test_retry_cancel.py`（12 个，覆盖 `with_retry` 直测 + token 各形态 + pipeline 端到端三条路径）；`tests/real/test_real_retry.py`（2 个真实 LLM 集成 —— 注入 transient 失败后真实 complete 通；全失败走 degraded 不打真实网络）。

### Public API（M1.1 新增）

`LLMRetryPolicy` / `with_retry` / `CancellationToken` / `CancellationLike` / `LLMTimeout` / `LLMRetryExhausted` / `CancellationRequested` / `CompactionFailed` / `LlmRetryAttemptedPayload` / `LlmDegradedPayload` / `LoopCancelledPayload` 全部从 `power_loop` 顶层导出；`AgentEventType.LLM_RETRY_ATTEMPTED` / `LLM_DEGRADED` / `LOOP_CANCELLED` 已加入枚举。

### Added — M1.9 MemoryProvider 协议（2026-06-05）

> 库内**零实现**：定义协议 + 接线 + 注入位置不变量。具体后端（SQLite / HTTP API / 向量库）一律留在调用方或 `examples/`。

- **`MemoryProvider` Protocol**（`power_loop.runtime.memory`）—— 两个方法：
  - `async recall(*, messages, session_id, budget_tokens=1500) -> list[dict]`
  - `async remember(*, snapshot: MemorySnapshot, session_id) -> None`
- **`MemorySnapshot`** dataclass —— `session_id / messages / final_text / rounds / status / metadata`，在 SESSION_END 时传给 `remember`。
- **`tag_as_memory(messages)`** —— 工具函数，把任意 dict 列表规范化成 `role=system, name=memory_*`。Pipeline 在注入前自动调用，业务方不必关心。
- **`AgentLoopConfig.memory: MemoryProvider | None = None`** + **`memory_budget_tokens: int = 1500`**。默认 None（保持原有行为）。
- **注入位置不变量**：召回结果插在 ``self.history`` 的「最长 leading `role=system` 段」之后、对话历史之前。这与 `compact_note` 同区，受压缩器系统区保留保护。
- **失败模型**（库强制不破坏主流程）：
  - `recall` 抛 → 视为返回 `[]`，emit `MEMORY_FAILED(phase="recall")`，loop 照常跑。
  - `remember` 抛 → emit `MEMORY_FAILED(phase="remember")`，`StatefulResult` 原样返回。
- **新 hook**：`HookPoint.MEMORY_RECALLED` + `MemoryRecalledCtx(recalled, session_id, budget_tokens)`。业务可在注入前 redact / 去敏 / `directive=SKIP` 跳过整批注入（典型场景：双方授权 gate）。
- **新事件**：`AgentEventType.MEMORY_RECALLED` + `MemoryRecalledPayload(returned, injected, budget_tokens)`；`AgentEventType.MEMORY_FAILED` + `MemoryFailedPayload(phase, error_type, error_message)`。
- **Pipeline 内部**：`_finalize` 多了一个 `rounds` 形参，使 `MemorySnapshot.rounds` 正确反映已完成回合数；老调用点（cancelled 早出）保留默认行为。
- **测试**：`tests/unit/test_memory.py`（6 个，覆盖注入位置 + tag 规范化 + recall 软失败 + remember 软失败 + 快照内容 + MEMORY_RECALLED SKIP）。
- **example**：`examples/13_memory_sqlite.py` —— SQLite 事实 KV，跨 session 把「我叫阿岚 / 喜欢 37」记忆带回。

### Public API（M1.9 新增）

`MemoryProvider` / `MemorySnapshot` / `tag_as_memory` / `MemoryRecalledCtx` / `MemoryRecalledPayload` / `MemoryFailedPayload` 顶层导出；`HookPoint.MEMORY_RECALLED`、`AgentEventType.MEMORY_RECALLED` / `MEMORY_FAILED` 入枚举。

### Added — M1.3 结构化输出（2026-06-05）

- **`LLMRequest.response_format: dict[str, Any] | None = None`** —— OpenAI 兼容 `response_format` 字段；`llm_factory._request_kwargs` 与 `_build_resume_request` 透传。
- **`StructuredOutputSpec(name, schema, strict=True, description=None, examples=...)`**（`power_loop.runtime.structured`）—— 声明式包装；`.to_openai_response_format()` 渲染成 `{"type":"json_schema","json_schema":{name, schema, strict, description}}`。
- **`parse_structured(output, *, schema=None) -> dict`** —— 四级修复链：
  1. 直接 `json.loads`
  2. markdown ```json``` 围栏剥离
  3. 抓出第一个**括号平衡**的 `{...}` 子串（跳过字符串里的引号）
  4. 修补**尾逗号** `,]` / `,}`
- **`StructuredOutputError(reason, raw_text, detail)`** —— 失败原因机器可读：`no_json` / `invalid_json` / `not_object` / `missing_required:<field>`，`raw_text` 截断到 1000 字符方便调试。
- **本地 schema 校验有限**：仅强制 `type=="object"` 与顶层 `required` 字段存在。更深的 type / enum / pattern 留给 provider 在 strict mode 服务端校验，**避免本地实现与 provider 静默分歧**。
- **测试**：`tests/unit/test_structured.py`（14 个）+ `tests/real/test_real_structured.py`（1 个真实 LLM —— card 抽取来回跑通）。
- **example**：`examples/14_structured_card.py` —— 真实 LLM 抽取 → 修复带噪 JSON → schema 缺字段失败三段。

### Public API（M1.3 新增）

`StructuredOutputSpec` / `parse_structured` / `StructuredOutputError` 顶层导出。`LLMRequest.response_format` 已在 `llm_client` 层落地。

### Added — M1.6 ToolRegistry async-handler 工效学（2026-06-05）

- **`async def` 自动识别**：`ToolRegistry.register()` 用 `inspect.iscoroutinefunction` 在登记时缓存 `RegisteredTool.is_async`，覆盖普通 `async def` 与 `async __call__` callable 对象两种形态。
- **`invoke()`（sync）对 async 处理器抛 `AsyncToolInSyncContext`**：取代之前「silently 返回未 await 的 coroutine」的隐式坑，错误信息明确指向 `invoke_async`。
- **`invoke_async()` 是通用入口**：async handler 直接 `await tool.handler(...)`；sync handler 跑完后若返回 awaitable 仍会被自动 await（保留向后兼容）。
- pipeline 早已用 `invoke_async`，业务侧无需改动；只有显式调 `invoke()` 把 async 当 sync 用的旧代码会立刻看到清晰报错。
- ROADMAP 里提到的 `tests/real/test_real_streaming_subagent.py` 的 `get_event_loop().is_running()` hack 已在 stateful 重构时随旧测试一并删除，本次 polish 把上游 API 工效学也补齐。
- **测试**：`tests/unit/test_tool_registry_async.py`（7 个）—— async 检测、callable 对象、sync-on-async 报错、双形态 invoke_async、sync-returning-awaitable 兼容。

### Public API（M1.6 新增）

`AsyncToolInSyncContext` 顶层导出。`ToolRegistry.invoke / invoke_async` 行为变更（前者更严格，对 async handler 抛清晰错；后者更优雅，省一次 `inspect.isawaitable`）—— 既有调 `invoke_async` 的代码完全不受影响。

### Added — M1.4 LLMProviderConfig 统一（2026-06-05）

- **`LLMProviderConfig`**（`power_loop.runtime.provider`）—— provider-agnostic 配置：`base_url` / `api_key` / `model` 必填，`provider` 标签（informational，今天只走 openai-compatible 一条 transport，预留 M3 多 transport 路由 key），加 `timeout_s` / `max_tokens` / `temperature` / `max_retries` 等默认值。
- **`LLMProviderConfig.from_env(prefix="POWER_LOOP", fallback_prefix="OPENAI_COMPAT", env=None)`** —— 读 `POWER_LOOP_*` 环境变量，缺则回退 `OPENAI_COMPAT_*`，**老 `.env` 无须改字段**。`env` 形参用于测试（注入 dict）。
- **`create_llm_service_from_config(cfg)` / `create_llm_service_from_env(*, prefix=…)`** —— 一行造服务；内部通过 `to_openai_compatible()` 适配现有 `OpenAICompatibleChatLLMService`。
- **失败模式**：必填字段缺失 → 构造时 `ValueError`（不是首个 `complete()` 时），让配置错误在 pytest 阶段就暴露。
- **docs/providers.md** —— 环境变量表 + 4 个 provider snippet（OpenAI / DashScope / DeepSeek / 本地 OpenAI-compatible）+ 老调用方式迁移指引。
- **测试**：`tests/unit/test_provider.py`（11 个）—— 必填守卫 / 主前缀 / 回退前缀 / 主前缀优先 / 三家 provider 参数化建造 / `from_env` 一行入口 / `to_openai_compatible` 适配回环。

### Public API（M1.4 新增）

`LLMProviderConfig` / `create_llm_service_from_config` / `create_llm_service_from_env` 顶层导出。`OPENAI_COMPAT_*` 环境变量名继续可用，仅为回退；新代码请用 `POWER_LOOP_*`（或自定义 prefix）。

### Added — M1.2 trim_history（2026-06-05）

- **`trim_history(messages, max_tokens, *, keep_system=True, keep_last_n=2)`**（`power_loop.runtime.budget`）—— 纯裁剪 helper：保留 leading system + 最后 N 个 user-bounded 交换，从中间删消息直到落在 token 预算内。不调 LLM（不摘要），仅是业务侧调用前裁剪。
- **不变量**：
  1. 预算已够 → 返回原 list（不复制）。
  2. `keep_system=True` → 所有 leading `role=system` 消息保留；`keep_last_n` 个 user-bounded 交换在尾部保留。
  3. `assistant(tool_calls) ↔ tool(tool_call_id=...)` 对永不拆分 — 裁剪边界通过 tool_call_id 配对检测自动调整。
  4. 当 system + tail 都放不下时，降级为 tail-only（丢 system）再按需从尾部裁剪。
  5. 不修改输入（返回新 list）。
- **测试**：`tests/unit/test_budget.py`（9 个）—— 已合预算 / 空 / 零预算 / 系统保留 / 去系统 / 工具对原子性 / 工具对在边界 / 仅 tail / 非突变。
- `estimate_tokens` / `estimate_text_tokens` / `trim_history` 从 `power_loop` 顶层导出。

### Public API（M1.2 新增）

`trim_history` / `estimate_tokens` / `estimate_text_tokens` 顶层导出。

### Added — M2.5 错误体系收口（2026-06-05）

- **`ToolNotFound(tool_name)`** —— `ToolRegistry.invoke / invoke_async` 找不到工具时 raise。
- **`ToolValidationError(tool_name, message)`** —— 参数校验失败时 raise。
- **`SpecValidationError(message, *, field=None)`** —— 新的规范验证错误；`AgentSpecError` 现继承于它（而它继承 `PowerLoopError`），旧代码 `except AgentSpecError` 继续有效，新代码 `except SpecValidationError` 或 `except PowerLoopError` 一把抓。
- `ToolRegistry.invoke / invoke_async` 对 unknown tool 和 invalid args 现在 **raise 异常而非 return 字符串**；`invoke` 也 raise 而非 return 字符串作为错误。Pipeline 的 `execute_tool` 内部 catch 这两个异常并返回 `(str(exc), True)` 给 LLM 看到（保持向后兼容）。
- **`ToolRegistry.validate` 保留为 internal legacy**（仍返回 `str | None`），管线仍用它做第一层检测，但新代码应直接 invoke + catch。
- 所有 `__init__.py` 和 `__all__` 同步更新。

### Added — M2.1 Public API 稳定性约定（2026-06-05）

- **README §5** 新增 "Public API 稳定性约定" 节：**STABLE**（24 符号，跨 minor 保证兼容 + CHANGELOG 独立条目）、**PROVISIONAL**（顶层导入但 0.x 可调）、**INTERNAL**（无版本承诺）。与 `power_loop/STABLE_API` 元组同步。
- Examples 表补齐 11–14（persistence / retry-cancel / memory-sqlite / structured-card）。
- §7 环境变量节更新为 `POWER_LOOP_*` 优先 + `create_llm_service_from_env()` 一行法。

### Added — M2.2 Hook/Event 全表文档（2026-06-05）

- **`docs/hooks.md`** §3.9 新增 `memory.recalled` hook 点文档（Ctx 字段 / SKIP directive / 双方授权示例）。
- **`docs/events.md`** 新增 §2.7 LLM retry/cancel lifecycle（`llm_retry_attempted` / `llm_degraded` / `loop_cancelled`）和 §2.8 Memory（`memory_recalled` / `memory_failed`），含完整 payload 字段表、触发时机、典型订阅者。

## [0.2.0] — 2026-06-05

Stateful refactor. The library now revolves around `StatefulAgentLoop` and a SQLite-backed `SessionStore`; the stateless `AgentLoop` is removed. **Hard break — no compatibility shim.**

### Added

- **`StatefulAgentLoop`** — the only public entry point. `new_session()` / `send(user_input, session_id)` / `send_sync` / `resume(sid)` / `abort_pending(sid)` / `close_session(sid, cascade=True)` / `close()` / `get_messages(sid)` / `get_pending(sid)`. Per-session `asyncio.Lock` so one instance can drive any number of sessions concurrently.
- **`SessionStore`** (`power_loop.runtime.session_store`) — SQLite-backed, the **only** thing that writes to disk. Five tables: `sessions` / `messages` / `compactions` / `usage_rounds` / `session_state`. Single connection + `threading.RLock`; WAL + busy_timeout. Public API surface for sessions, messages, compactions, usage, lifecycle.
- **`MessageSink`** Protocol + `NullSink` + `SQLiteSink` — pipeline persistence hook. SQLiteSink owns the in-memory `_history_seqs` list that mirrors `pipeline.history` so the compactor can translate fold indices back to store rows.
- **Pending state machine** — `assistant(tool_calls)` falling-into-store immediately marks `session_state.pending`; each matching `tool` message clears it. Mid-tool crash leaves a recoverable state. Next `send` raises `SessionPendingError`; caller picks `resume()` (replay remaining tools) or `abort_pending(sid, reason=…)` (synthesize `<aborted>` tool messages).
- **Subagent on top of `SessionStore`** — `spawn_agent` rewritten as a thin shell over the shared store. Children get their own row with `parent_session_id` / `spawn_tool_call_id` / `spawn_depth` (`MAX_SPAWN_DEPTH=3` enforced at insert time).
- **`AgentSpec`** (`power_loop.runtime.spec`) — strict-schema declarative subagent: `name / system_prompt / tools / max_rounds / max_tokens / temperature / model / lifecycle / metadata`. Unknown fields → `AgentSpecError`. `from_dict` / `from_json` factories.
- **`run_agent` meta-tool** — declarative companion to `spawn_agent`. The parent LLM submits a full `AgentSpec` JSON; the library validates and dispatches via `run_agent_spec`. Both meta-tools registered by a single `register_spawn_agent(registry)`.
- **`SubagentLifecycle`** enum — `EPHEMERAL` (default, deleted on success, preserved on failure for debug) / `LINKED` (cascade-deleted with parent) / `DETACHED` (independent of parent's lifecycle).
- **`Compactor`** Protocol + **`DefaultCompactor`** (`power_loop.runtime.compact`) — pluggable LLM-summary compaction. Trigger at `max_tokens × trigger_ratio` (default 0.75) or absolute `CONTEXT_COMPACT_THRESHOLD`. Preserves all `role=system`, last `keep_last_n` user-bounded exchanges, and the `assistant(tool_calls) ↔ tool` atomic pair. Soft-fails to `None` on summary errors so the loop degrades gracefully.
- **`runtime/budget.py`** — `estimate_tokens(messages)` heuristic (≈4 chars/token, stdlib-only) used by the compactor's trigger logic.
- **Error hierarchy** — `PowerLoopError` base + `SessionNotFoundError` + `SessionPendingError(session_id, assistant_seq, pending_tool_calls)`. Caller catches the base class to handle every library-raised exception.
- **`_current_loop` contextvar** (`power_loop.core.agent_context`) — threads the active `StatefulAgentLoop` through tool invocations so meta-tools like `spawn_agent` find their parent without ambient state.
- **Examples 00–05** — progressive tutorial: minimal send → multi-turn → tool calling → subagent → compaction → pending recovery. Each file introduces exactly one new concept. `examples/_helpers.py` shares `.env` loading + LLM construction.
- **Real-LLM test suite** — `tests/real/test_real_stateful_loop.py` / `test_real_tool_use.py` / `test_real_subagent.py` / `test_real_pending_resume.py` / `test_real_compaction.py` / `test_examples.py` (6 examples). `tests/real/judge.py` provides an **LLM-as-judge** helper: tests assert `await assert_passes(question, answer, rubric)` and a separate power-loop evaluator returns `{passed, reason}` JSON, solving the LLM-non-determinism assertion problem.

### Changed — Breaking

- **Removed `AgentLoop` and `agent_loop_async`**. Replace `AgentLoop(llm, config).run(messages=…)` with `StatefulAgentLoop(llm=…, db_path=…, config=…).send(user_input, session_id=…)`. The stateless model is gone — callers no longer ship the full messages list per turn; power-loop loads history from the store.
- **`AgentLoopConfig.compactor: Compactor | None = DefaultCompactor()`** — default-on. Pass `None` to disable.
- **`CompactBeforeCtx`** loses `input_tokens` and `compact_threshold` — neither carried useful data after the runtime/compact.py rewrite. **`AutoCompactStatusPayload`** trades them for `before_tokens` / `after_tokens` (sourced from the `CompactionPlan`).
- **`ContextManager`** loses `compact_async` / `should_compact` / `compact_threshold` / `last_input_tokens` / `_compact_count` / `reset_usage`. It now owns only `update_usage` (telemetry parsing), `microcompact` (large tool-output spill-to-disk), `recent_files`, and `TodoManager`. LLM-summary compaction has moved to `runtime/compact.py`.
- **`power_loop.contracts.errors`** is now a real module (was unused).

### Removed

- `power_loop/agent/loop.py` — `AgentLoop` shell.
- `power_loop/core/agent.py` — `agent_loop_async` entry point.
- Six stale integration / real tests that depended on the removed API; replaced by the new `tests/real/test_real_*.py` suite + the 6 example-driven tests.

### Migration

| Before (0.1.x) | After (0.2.0) |
|---|---|
| `AgentLoop(llm, config).run(messages=[…])` | `sid = loop.new_session(); await loop.send(user_input, session_id=sid)` |
| Caller manages `messages` list | Library loads from `SessionStore` by `session_id` |
| No persistence | `db_path` (default `./power_loop_sessions.db`); `":memory:"` for tests |
| No pending detection | Crash mid-tool → next `send` raises `SessionPendingError`; pick `resume()` or `abort_pending()` |
| Compaction via `ContextManager.compact_async` | `AgentLoopConfig.compactor = DefaultCompactor()` (default-on); pluggable via `Compactor` protocol |
| `spawn_agent` with private `AgentLoop` | `register_spawn_agent(registry)` + `run_agent` meta-tool; shared `SessionStore` with parent linking |
| No declarative subagent | `AgentSpec` + `run_agent_spec(spec, input, parent_loop=…)` |

### Documentation

- README rewritten around `StatefulAgentLoop`. Sections: 1. what it is/isn't · 2. install · 3. quickstart (mirrors examples 00→05) · 4. core concepts (Session / SessionStore / Sink / Compactor / Pending / Subagent / Hooks vs Events) · 5. flat API reference · 6. examples table · 7. env-var config · 8. pipeline ASCII trace + persistence/seq notes + pending state machine · 9. tests (including the LLM-as-judge pattern) · 10. roadmap pointer.
- `docs/hooks.md` — every `HookPoint` with its typed Ctx fields + accepted directives + typical use cases.
- `docs/events.md` — every `AgentEventType` with its payload fields + when fired + typical subscriber.

### Added — M0 工程化基线（2026-06-05）
- `power_loop.__version__ = "0.1.0"` 单一来源 + `STABLE_API` 元组声明稳定面（`AgentLoop / AgentLoopConfig / AgentLoopResult / AgentHooks / AgentEventBus / HookPoint / HookDirective / ToolRegistry / ToolDefinition`）。
- `pyproject.toml` 补 license / classifiers / urls / dynamic version；新增 dev extras (`ruff` / `mypy`)；统一 `ruff` (line=120, E/F/I/UP/B) 与 `pytest` marker (`unit` / `integration` / `real_llm`)。
- `CHANGELOG.md` 与 `ROADMAP.md` 落地（M0–M3 + 关键短板拆解）。
- `tests/` 重排为 `unit/ integration/ real/`；`tests/conftest.py` 实现 `real_llm` 默认 ON、`--no-real` 反向跳、缺 env 自动 skip；`tests/real/conftest.py` 自动给 `tests/real/*` 打 marker。
- GitHub Actions：`ci.yml`（ruff + mypy 非阻塞 + pytest --no-real，3.10/3.12 矩阵）+ `real-llm-nightly.yml`（凭 repo secrets 跑真实 LLM）。
- `examples/00_minimal.py` —— DeepTalk Agent MVP 同款单回合用法；`tests/real/test_examples.py` 把 example 作为活文档锁定。
- README 新增 "实现亮点 / 卖点"（含 M1.7a 压缩策略、M1.9 记忆分层）与 STABLE / PROVISIONAL / INTERNAL 三层稳定性约定。

### Changed
- `ruff --fix` 自动现代化（PEP 585/604、import 排序、unused import 清理）共 458 处；剩余 28 处为真实代码债，按规则码登记到 `pyproject.toml` ignore，统一 M1 接触模块时清理。

### Fixed — M0.8 / M0.9（同日）
- ruff 残留 28 → 0：清掉 19 处 `__init__.py` E402（`STABLE_API` 重排到 imports 之后）；修真实 bug `llm_client/llm_factory.py:855` 未定义的 `e`；修闭包变量延迟绑定（B023）；`Union[…]` → `X | Y`；`zip(...)` 加 `strict=`；其余移除未用变量与 E402。
- mypy 残留 32 → 0：`LLMRequest.messages/tools` 类型放宽为 `list[dict[str, Any]]`（list 不变性问题）；`default_tools.py` 4 处隐式 Optional 改 `int | None` / `str | None`；Popen None 守卫；3 处 var-annotated；`pipeline.execute_tool` 加 ToolRegistry None 守卫；`phase.py` event_meta 显式 `dict[str, Any]`；`system_prompt.py` lambda 改 def 以保留类型推导。
- CI 中 mypy 改为阻塞（不再 `|| true`）。

### Known limitations
- M1 起的所有短板（重试 / 取消 / 压缩 / 持久化 / 记忆 / spec sub-agent / 动态工具）仍未实现。

## [0.1.0] — 2026-06-05

首个基线版本（M0 起点的现状快照）。

### Added
- `AgentLoop` 主外观类 + `AgentLoopConfig` / `AgentLoopResult`。
- `AgentPipeline`：按 phase 拆分的主循环，含 LLM 调用 / 工具调用 / 压缩 hook 点。
- `AgentHooks`：有序 sync/async hook 管理，支持 typed Ctx 与 legacy dict 两种回调。
- `AgentEventBus`：事件订阅 / 发布，订阅者错误隔离。
- `HookPoint` 15 个：`session.* / round.* / llm.* / tools.* / tool.* / compact.* / message.append`。
- `HookDirective`：`CONTINUE / SKIP / BREAK / SHORT_CIRCUIT` 控制流。
- `ToolRegistry`：动态工具注册 + JSON Schema 必填校验 + OpenAI tool schema 导出。
- 默认工具集：`run_bash / run_read / run_write / run_edit / run_grep / run_glob / apply_patch`，含路径白名单。
- `spawn_agent` 工具：命令式子代理，含深度守卫与父总线事件冒泡。
- `llm_client/`：OpenAI 兼容（含 DashScope）+ Anthropic 双家底；多模态、web search、tool-call 解析。
- `SystemPromptBuilder` + `SystemPromptContext`：可拼装的系统提示。
- `runtime/skills.py`：SKILL.md frontmatter 加载（粗糙版）。
- `runtime/env.py`：WORKSPACE / AGENT_DIR 路径白名单。
- 测试 3.7k 行：契约单测、fake LLM 集成、真实 DashScope 端到端 showcase。

### Documentation
- `README.md`：定位、目录结构、安装、env 配置、最小用法、Hook & Event 模型、工具系统、与 DeepTalk 分工表、Roadmap。
- `ROADMAP.md`：M0–M3 四阶段规划，含上下文压缩、AgentSpec、动态工具注册等关键短板。

### Known limitations
- Public API 边界尚未文档化（M0 中解决）。
- 缺乏 LLM 重试 / 超时 / 取消的统一策略（M1.1）。
- 历史窗口 / token 预算修剪缺工具（M1.2）。
- `LLMRequest` 无 `response_format`（结构化输出，M1.3）。
- Provider 实例化散落在 `llm_factory.py`，无 unified `LLMProviderConfig`（M1.4）。
- `compact.*` hook 点存在但**无默认 compactor 实现**，长会话会 hard-fail 在 context overflow（**M1.7a 必须**）。
- 无声明式 sub-agent（`AgentSpec`，M1.8）。
- 无运行时工具注册 meta-tool（M2.6）。
