# 工作流

workflow 提交时固定图定义，创建独立的调度器 session/job/start 待办；每次叶子执行也有
自己的 session 和 job。同一套 Runtime client/worker 协议支持单进程与宿主部署的多个
进程。原父 Agent 的 Python 对象结束，不会结束健康叶子。

## 提交和读取结果

先创建 Runtime、在 RegistryRuntimeFactory 登记带版本的环境构造器、启动 worker，
再用 RuntimeDescriptor 创建根 session。完整示例见
[27_dynamic_workflow.py](../../../examples/27_dynamic_workflow.py)。

```python
from power_loop.workflow import create_workflow

spec = {
    "name": "review", "budget": {"max_tokens": 50000}, "timeout_s": 300,
    "root": {"type": "sequence", "steps": [
        {"type": "agent", "id": "draft", "input": "写一份简短提案。",
         "spec": {"name": "draft", "system_prompt": "写清楚方案。", "tools": []}},
        {"type": "agent", "id": "review", "inputs_from": ["draft"],
         "spec": {"name": "review", "system_prompt": "检查提案。", "tools": []}},
    ]},
}
handle = await runtime.client.workflow(target, spec, message_id="review-1", actor=actor)
result = await handle.wait()
print(result.status, result.usage)

# 便利入口调用同一条受理路径；相同 ID、相同内容返回原受理。
result = await create_workflow(spec, client=runtime.client, target=target,
                              actor=actor, message_id="review-1").run()
```

handle 只有持久的 run_id、job_ref、input_ref，不持有本地执行任务。客户端重建后可用
`client.wait(input_ref, actor=actor, deadline_ms=...)` 读取原结果；返回的统一 Result
在 structured_output 中携带工作流结果。等待超时只结束等待，取消执行需要明确调用 cancel。
相同请求 ID 携带不同内容会报冲突。

## 图的行为

- sequence 顺序执行。inputs_from 追加上游结果；`{{input}}`、`{{变量}}`、
  `{{node.key}}` 可用于叶子输入和系统提示词。
- parallel 按 max_concurrency 并行；foreach 遍历 items/items_from，嵌套迭代各有
  独立身份。在循环体外引用 foreach 的汇总 ID，不引用某次循环体叶子。
- branch 读取先前结果选分支；when 可跳过叶子。结构化字段的生产节点必须声明 output_schema。
- loop 按 until/max_iterations 迭代；continue_nodes 在原节点会话中启动下一次执行，
  保留上下文，不恢复旧 Python 调用栈。
- retry 为已知失败创建新 attempt；fallback、continue_on_error 按声明执行。
  容器 on_error="halt" 会停止同一容器尚未结束的分支；"continue" 收集失败后继续其它
  分支。需要同时查看 status、ok 和 errors。
- continuation 可为轮数用尽或上下文检查点结果追加执行，复用原节点 job/session。
  默认门检查未完成 todo，gate="always" 使用声明的继续次数。unknown/interrupted
  外部动作不会自动重试。

节点 ID 在定义内全局唯一，依赖只能指向同一路径上已完成的节点，不能引用并行兄弟。
JSON 与 YAML 编译为同一个 WorkflowSpec；无效定义在受理前集中返回问题。

## 调度器重建、停止和显式继续

原 start input 和已经选定的正代次 JobRef 始终读取原代次结果，包括旧调度器的中断结果。
接手后需要先检查 workflow job 的当前代次，再等待或取消该新代次。

run/path/iteration/attempt 形成稳定节点键。节点 job、session、start 待办和 journal
引用同事务保存；节点结果与父 inbox 通知同事务保存。叶子不覆盖整份父 journal，调度器
消费结果时更新自己的节点记录。

新 worker 取得调度器自身 session 执行权后，按原 job/version 条件推进 generation。
它读取固定图，复用已完成结果、等待健康原叶子。节点结果和费用保持原归属。信息不足时
允许报告 interrupted/unknown，不自动重做未知外部动作来补结果。

```python
await handle.cancel("用户停止工作流")  # 先关闭整树新派发
await runtime.client.cancel(node_ref, actor=actor, scope="self")  # 指定节点和代次
await runtime.client.send(node_ref, "补充要求", message_id="note-1", actor=actor)

# 工作流已有终态结果时，明确给出当前 generation。
continued = await handle.resume(generation=current_generation, node_id="draft",
    message="修改提案", message_id="revision-1")
result = await continued.wait()
```

显式继续在所选节点原 job 中开启新代次，其依赖尾部用新 attempt 重算；未变化分支复用
原结果。节点在 foreach/loop 中存在多个迭代时，图级 resume 拒绝歧义；先检查具体 job，
再提交范围明确的新任务。运行中节点用 send 接收消息；图级 resume 要求工作流
已经终结。已取消代次不会重新启动。

自动与显式继续都保留累计 token 用量和原工作流截止时间。额度用尽仍可读取已有结果。
调用次数与费用上限使用 Runtime 的 BudgetConfig 和原逻辑归属。图 token 上限是派发前
软限制，已运行的并行调用可以在到达上限后完成。未结费用继续可见，并阻止监督范围关闭。

## worker、模型工具和资源

跨进程部署须配置共享 store、SessionOwnershipCoordinator、MessageTransport，并设置
shared_execution=True；默认内存适配器仅供单进程使用。所有 worker 必须能解析冻结的
版本化环境定义。宿主负责 worker 路由和进程/容器隔离，SDK 不再逐叶子拉起进程或创建
独立数据库。

register_workflow_tool 提供 create/validate/status/list/leaves/logs、节点 send（可选 urgent）
和 cancel，logs 列出节点最近的消息；register_workflow_tools 将同一受理路径注册为 create/validate/status 三个
工具名。模型创建立即返回收据，结果进入原父 inbox。WorkflowToolPorts 保留定义加载、
策略转换、工具授权收窄和有时限的展示观察函数；观察函数不决定任务完成。

宿主资源定义可提供 resources.workflow_file_io，实现 output_path(node_id, iteration=...)
和 render_ref(path, slice_expr)，支持 output_file 与 `@@FILEREF:path@@`。各 worker
必须能访问指定资源，不把 Python 资源对象序列化进请求。文件适配器失败会明确报错。
必要子任务和等待用户回答归属叶子自身 session/job。

本次未发布设计 203 分支使用 schema 15/protocol 42 和新建数据库。旧 parent_loop
workflow 入口、本地恢复句柄、完成回调、eager_wake、detached/wait/sync 参数和 subprocess
executor 已删除，不提供旧数据迁移或兼容运行路径。
