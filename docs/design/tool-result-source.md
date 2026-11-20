# 工具结果的原调用来源

工具结果的 `meta["_assistant_seq"]` 是同一会话中实际产生该结果的 assistant 原始行 seq。
工具结果自身的 `tool_call_id` 再选择该行中的一个调用。ID 可以跨轮或跨 send 复用；不按最近
assistant、round 或全历史同 ID 推断来源。

正常 SQLiteSink 在结果的同一次 INSERT 中保存保留键，覆盖外部结果 meta 的同名值。
只有属于当前 unresolved 调用且持有真实 assistant seq 时才写来源。消息和 meta 使用局部副本，
调用者对象不变，`send_rows` 与持久行的来源一致。abort/prime 复用 pending 中的来源；
直接恢复未知结果使用已读 pending.assistant_seq，不增加反查、写入或工具执行。

直接调用 store 的代码须显式传递自己持有的真实位置，例如：

```python
assistant_seq = await store.append_message(
    sid, role="assistant", send_index=1,
    tool_calls=[{"id": "c1", "type": "function", "function": {
        "name": "echo", "arguments": '{"text":"hello"}',
    }}],
)
await store.append_message(
    sid, role="tool", send_index=1, tool_call_id="c1", content="hello",
    meta={"_assistant_seq": assistant_seq},
)
```

读取先按 `(session_id,seq)` 取结果，再用来源 seq、同 send 和 assistant role 定点读取
`tool_calls_json`，不取 assistant 正文或 meta。已持有结果的 `recall_original` 复用该行；
group item 使用相同 helper，没有 group.call_seq 回退。compacted_out 不妨碍原文读取。

缺来源键、原调用被清理、来源不属于同 send 的 assistant 或没有匹配 call ID 时，结果正文仍可读。
JSON 的 `arguments` 为 null；文本显示 `Original call arguments unavailable.`，不生成 assistant 调用块。
不自动回填开发旧行，不扫描历史或猜测来源。重复 ID 时只返回实际来源的单一调用，纠正此前首个匹配或多块错配。

正常工具精确召回为结果一条查询、原调用一条查询；非工具只取结果，缺来源键或 call ID 时不查调用。
带位置但原调用不可用时，仍执行这一条窄读，再明确返回参数不可用。
错误 seq 才读 send 首末形成原错误 span。group 的既有 runtime-state 读取及 background payload
的独立结果读取另外计量；选中结果和单个 assistant 调用数组很大时，字节仍随选中内容增长。

本合同来自已批准的 [S09-R5 施工卡](../../../outputs/design216-impl/S09/rework-05.md)，
不改变 schema、协议、正常 send 的查询数/事务数、pending 格式或预算。

## pending 的批次定位

pending 只保存 `{"assistant_seq": n}`。它定位一个尚未关闭的 assistant 工具批；每个工具结果的
meta._assistant_seq 定位这条结果的实际调用来源，两者都指向原始消息 seq。调用参数不再保存在 pending。

正常中间结果只写原消息/序号，保持 marker；最后结果和清 marker 同事务。未完成集合只在成功 append 后
移除当前 ID，不逐个结果复制剩余集合。非 Runtime sink 的入口也只写同一 marker。

repair 和显式 abort 集中读取 source 和当前批结果，按已有 next_seq 的严格上界求剩余；
原 assistant 可以是 compacted_out。repair 使用已有 owner 事务，abort 保留原来的逐次条件写入，
包括全部已完成但 marker 仍在时的清理。来源不可用明确报 pending_source_unavailable，不清后假报成功。
旧 `_execute_pending` 自动重放和它唯一使用的 `_prime_sink_from_pending` 已删除；R5 的部分完成/abort 来源行为
由真实 sink/abort/repair 入口继续保证。get_pending 原样返回 marker，不暗中加载完整调用。

没有旧数组/ID-only 格式识别或自动迁移，没有增加 schema、索引、缓存、热路径检查或恢复步骤。
详见 [S09-R6](../../../outputs/design216-impl/S09/rework-06.md)；参数字节、更新后行图像与物理 I/O 必须分别计量。
