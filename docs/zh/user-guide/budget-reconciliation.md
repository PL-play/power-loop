# 计费事件与重报

SDK 12 不预扣、不保存第二套费用账本。模型与工具继续发送正常用量事件；DeepTalk 宿主按设计 214 生成 BillingChargeEvent，由独立消费者幂等落账。重报使用宿主已记录的事件，不重放模型或工具，也不根据历史缺失信息猜费用。

执行坐标为 store/session/send_index，工具使用 MessageRef/tool_call_id。session 的完成、取消或恢复不改变已经发生的费用事实。当前运行接口见 [Runtime 12](../../runtime-12.md)。
