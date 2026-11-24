import { validate, replay } from '../src/replay.js';

const requests = [
  { id: 'req-a', seed: 101, params: { temperature: 0.8, topP: 0.95 } },
  { id: 'req-b', seed: 202, params: { temperature: 1.0, topK: 8 } },
  { id: 'req-c', seed: 303, params: { temperature: 0.7 } },
];

const events = [
  { type: 'batch', requests },
  { type: 'step', requestId: 'req-a' },
  { type: 'step', requestId: 'req-b' },
  { type: 'step', requestId: 'req-a' },
  { type: 'branch', requestId: 'req-a', count: 3 },
  { type: 'step', requestId: 'req-a' },
  { type: 'step', requestId: 'req-c' },
  { type: 'merge', requestId: 'req-a', keep: 2 },
  { type: 'step', requestId: 'req-a' },
  { type: 'retry', requestId: 'req-b' },
  { type: 'step', requestId: 'req-b' },
  { type: 'step', requestId: 'req-b' },
];

const report = validate(events);

console.log('=== deterministic replay (per-step RNG state -> token) ===');
for (const record of report.deterministic.records) {
  console.log(
    `step ${String(record.step).padStart(2)}  ${record.requestId}  [${record.sequenceId}]` +
      `  state=${record.state}  token=${record.token}`,
  );
}

console.log('\n=== tokens per request (deterministic) ===');
for (const [id, sequences] of Object.entries(report.deterministic.sequences)) {
  for (const sequence of sequences) {
    console.log(`${id} ${sequence.sequenceId}: [${sequence.tokens.join(', ')}]  final=${sequence.finalState}`);
  }
}

console.log('\n=== state drift vs naive implementation ===');
for (const [id, drift] of Object.entries(report.drift)) {
  console.log(
    `${id}: firstDivergence=${drift.firstDivergence} mismatches=${drift.mismatches}` +
      `  det=[${drift.referenceTokens.join(', ')}]  naive=[${drift.targetTokens.join(', ')}]`,
  );
}

console.log('\n=== non-reproducibility reasons (naive policy) ===');
for (const reason of report.nonReproducibilityReasons) {
  console.log(`- ${reason}`);
}

const reordered = [
  { type: 'batch', requests: [...requests].reverse() },
  ...events.slice(1),
];
const rerun = replay(reordered, { policy: 'deterministic' });
const stable = Object.entries(report.deterministic.sequences).every(
  ([id, sequences]) => JSON.stringify(rerun.sequences[id]) === JSON.stringify(sequences),
);
console.log(`\nbatch-reorder invariant (deterministic policy): ${stable ? 'PASS' : 'FAIL'}`);
