# Session 输入取消

send 返回包含 InputRef 的 Receipt。保存该身份；受理结果不确定时，用原 message_id 查询，不分配新的输入身份。取消按原输入及其当前 send 生效。

```python
receipt = await runtime.client.send(target, "执行工作", message_id="request-1", actor=actor)
await runtime.client.inputs.cancel(receipt.input_ref, actor=actor)
result = await runtime.client.inputs.get_result(receipt.input_ref, actor=actor)
```

取消本地等待不会取消已受理输入。worker 在模型和工具边界响应取消；结果未知的工具不自动重做，恢复补齐未知结果后让会话继续。linked 子会话跟随父级取消，independent 子会话保留自己的生命周期。

RequestOptions 设置单次输入的模型重试上限；模型请求重试不重复已经完成的工具。参见 [Runtime 12](../runtime-12.md) 和 `examples/12_retry_and_cancel.py`。
