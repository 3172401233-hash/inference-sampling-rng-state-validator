# inference-sampling-rng-state-validator

验证推理采样 RNG 状态、批次和分支：对同一批请求种子、采样参数、批次拆分、
token 步数、beam 分支、合并与重试事件做**确定性采样回放**，逐步计算应使用的
RNG 状态与 token，并输出 token 序列、状态漂移和不可复现原因。

> **注意：本仓库中所有分数（logits、概率、token）均为模拟值。**
> `src/sampler.js` 中的 `mockLogits` 用确定性哈希生成"模型"分数，仅用于复现
> RNG 状态管理问题，不代表任何真实模型的输出质量或分布。

## 环境

- Node.js 24.16.0（`engines: >=24.16.0`）

## 运行

```bash
node --test        # 运行全部测试
node bin/replay.js # 运行演示场景，打印逐步 RNG 状态、token、漂移与原因
```

## 设计

RNG 状态为 `(base, cursor)`：`base` 是 64 位流标识，`cursor` 是已提交的
抽取次数，第 `i` 次抽取使用 `mix64(base, i)` 经 splitmix64 得到均匀随机数。

确定性策略（`src/replay.js`，`policy: 'deterministic'`）保证：

- **请求种子不受 batch 重排影响**：`base = mix64('request', seed, requestId)`，
  只依赖请求自身，与批次中的位置无关。
- **beam 分支从确定性派生状态开始**：`deriveState(parent, 'branch', k)`，
  分支拥有独立流，且父流不受分支影响。
- **合并不回退游标**：合并保留各分支中最大的已提交 cursor，继续生成时
  不会复用已提交的随机数。
- **重试不复用已提交的随机序列**：第 `n` 次重试使用
  `deriveState(attemptStart, 'retry', n)` 派生的新流。

对照的朴素策略（`policy: 'naive'`）复现四类缺陷：batch 共享 RNG（重排即
漂移）、分支复制父状态（分支流坍缩相同）、合并回退游标、重试重放已提交
序列。`validate(events)` 同时运行两种策略，输出每个请求的 token 漂移
（首次分叉位置、不一致数）与不可复现原因列表。

## 事件模型

```js
{ type: 'batch',  requests: [{ id, seed, params, ... }] } // params: temperature/topK/topP
{ type: 'step',   requestId }                              // 解码一个 token
{ type: 'branch', requestId, count }                       // beam 分支
{ type: 'merge',  requestId, keep }                        // 合并，保留第 keep 个分支
{ type: 'retry',  requestId }                              // 重试当前尝试
```

## 测试

`test/replay.test.js` 覆盖：batch 重排不变性（确定性策略）与重排漂移（朴素
策略）、beam 分支派生状态、合并游标不回退、重试不复用序列、完整回放可复现性。
