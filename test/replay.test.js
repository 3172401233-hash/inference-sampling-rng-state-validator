import { test } from 'node:test';
import assert from 'node:assert/strict';
import { replay, validate, compareRuns, REASONS } from '../src/replay.js';
import { seedRequest, deriveState, draw, stateKey } from '../src/rng.js';

const REQUESTS = [
  { id: 'req-a', seed: 11, params: { temperature: 0.9, topP: 0.95 } },
  { id: 'req-b', seed: 22, params: { temperature: 1.0, topK: 8 } },
  { id: 'req-c', seed: 33, params: { temperature: 0.7 } },
];

function batchEvents(order) {
  return [
    { type: 'batch', requests: order },
    { type: 'step', requestId: 'req-a' },
    { type: 'step', requestId: 'req-b' },
    { type: 'step', requestId: 'req-c' },
    { type: 'step', requestId: 'req-a' },
    { type: 'step', requestId: 'req-b' },
  ];
}

test('request seeds are invariant under batch reorder (deterministic policy)', () => {
  const forward = replay(batchEvents(REQUESTS), { policy: 'deterministic' });
  const reversed = replay(batchEvents([...REQUESTS].reverse()), { policy: 'deterministic' });
  assert.deepEqual(reversed.sequences, forward.sequences);
});

test('naive shared batch RNG drifts under batch reorder', () => {
  const forward = replay(batchEvents(REQUESTS), { policy: 'naive' });
  const reversed = replay(batchEvents([...REQUESTS].reverse()), { policy: 'naive' });
  const drift = compareRuns(forward, reversed);
  const drifted = Object.values(drift).filter((entry) => entry.mismatches > 0);
  assert.ok(drifted.length > 0, 'expected reorder to change naive results');
});

test('same seed and event log replays identically', () => {
  const events = batchEvents(REQUESTS);
  const first = replay(events, { policy: 'deterministic' });
  const second = replay(events, { policy: 'deterministic' });
  assert.deepEqual(first, second);
});

test('beam branches start from deterministically derived distinct states', () => {
  const events = [
    { type: 'batch', requests: [REQUESTS[0]] },
    { type: 'step', requestId: 'req-a' },
    { type: 'branch', requestId: 'req-a', count: 3 },
  ];
  const run = replay(events, { policy: 'deterministic' });
  const parent = run.sequences['req-a'][0];
  const branchStates = run.sequences['req-a'].slice(1).map((seq) => seq.finalState);
  assert.equal(branchStates.length, 2);
  assert.ok(new Set(branchStates).size === 2, 'branch states must be distinct');
  assert.ok(!branchStates.includes(parent.finalState), 'branch state must differ from parent');

  // Branch states are exactly the deterministic derivation from the parent.
  const parentState = draw(seedRequest(REQUESTS[0].seed, 'req-a')).next;
  assert.equal(branchStates[0], stateKey(deriveState(parentState, 'branch', 1)));
  assert.equal(branchStates[1], stateKey(deriveState(parentState, 'branch', 2)));

  // Branching must not disturb the parent's own stream.
  const noBranch = replay([events[0], events[1]], { policy: 'deterministic' });
  assert.equal(parent.finalState, noBranch.sequences['req-a'][0].finalState);
});

test('naive branches collapse to identical streams', () => {
  const events = [
    { type: 'batch', requests: [REQUESTS[0]] },
    { type: 'branch', requestId: 'req-a', count: 2 },
  ];
  const run = replay(events, { policy: 'naive' });
  const [parent, branch] = run.sequences['req-a'];
  assert.equal(parent.finalState, branch.finalState);
});

test('merge never rolls back the cursor', () => {
  const events = [
    { type: 'batch', requests: [REQUESTS[0]] },
    { type: 'step', requestId: 'req-a' },
    { type: 'branch', requestId: 'req-a', count: 2 },
    { type: 'step', requestId: 'req-a' },
    { type: 'merge', requestId: 'req-a', keep: 1 },
    { type: 'step', requestId: 'req-a' },
  ];
  const run = replay(events, { policy: 'deterministic' });
  const postMerge = run.records.at(-1);
  const cursor = Number(postMerge.state.split(':')[1]);
  assert.ok(cursor >= 1, `cursor must not roll back, got ${cursor}`);

  const report = validate(events);
  assert.ok(report.drift['req-a'].mismatches > 0);
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.MERGE_CURSOR_ROLLBACK));
});

test('retry does not reuse the committed random sequence', () => {
  const events = [
    { type: 'batch', requests: [REQUESTS[1]] },
    { type: 'step', requestId: 'req-b' },
    { type: 'step', requestId: 'req-b' },
    { type: 'retry', requestId: 'req-b' },
    { type: 'step', requestId: 'req-b' },
    { type: 'step', requestId: 'req-b' },
  ];
  const run = replay(events, { policy: 'deterministic' });
  const beforeRetry = run.records.slice(0, 2).map((rec) => rec.state);
  const afterRetry = run.records.slice(2).map((rec) => rec.state);
  for (const state of afterRetry) {
    assert.ok(!beforeRetry.includes(state), 'retry must not reuse committed states');
  }

  // Naive retry rewinds to the attempt start and replays committed RNG states.
  const naive = replay(events, { policy: 'naive' });
  assert.equal(naive.records[2].state, naive.records[0].state);
  assert.equal(naive.records[3].state, naive.records[1].state);

  const report = validate(events);
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.RETRY_REUSE));
});

test('validate reports drift and all applicable reasons', () => {
  const events = [
    { type: 'batch', requests: REQUESTS },
    { type: 'step', requestId: 'req-a' },
    { type: 'branch', requestId: 'req-a', count: 2 },
    { type: 'step', requestId: 'req-a' },
    { type: 'merge', requestId: 'req-a', keep: 1 },
    { type: 'retry', requestId: 'req-b' },
    { type: 'step', requestId: 'req-b' },
  ];
  const report = validate(events);
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.BATCH_SHARED_RNG));
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.BRANCH_SHARED_STATE));
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.MERGE_CURSOR_ROLLBACK));
  assert.ok(report.nonReproducibilityReasons.includes(REASONS.RETRY_REUSE));
  assert.equal(Object.keys(report.drift).length, REQUESTS.length);
});

test('rng primitives: derived states are deterministic and distinct', () => {
  const base = seedRequest(42, 'req-x');
  const again = seedRequest(42, 'req-x');
  assert.equal(stateKey(base), stateKey(again));
  const child = deriveState(base, 'branch', 1);
  assert.notEqual(stateKey(child), stateKey(base));
  const first = draw(base);
  const second = draw(base);
  assert.equal(first.value, second.value);
  assert.equal(first.next.cursor, 1);
});
