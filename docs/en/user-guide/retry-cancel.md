# Session input cancellation

A send returns a Receipt with InputRef. Persist that identity and reuse the original message_id when querying an uncertain acceptance. Cancellation is scoped to the accepted input and its current send.

```python
receipt = await runtime.client.send(target, "work", message_id="request-1", actor=actor)
await runtime.client.inputs.cancel(receipt.input_ref, actor=actor)
result = await runtime.client.inputs.get_result(receipt.input_ref, actor=actor)
```

Canceling a local wait does not cancel the accepted input. A worker stops cooperatively at model/tool boundaries. A tool whose outcome is unknown is not automatically repeated; recovery closes its missing result so the session can continue. A linked child follows the parent cancellation contract; independent children retain their own lifecycle.

RequestOptions controls per-input model retry limits. Retrying a model request does not re-execute completed tools. See [Runtime 12](../../runtime-12.md) and `examples/12_retry_and_cancel.py`.
