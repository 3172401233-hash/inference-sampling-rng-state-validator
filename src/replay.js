import { seedRequest, deriveState, draw, mix64, stateKey } from './rng.js';
import { mockLogits, sampleToken, DEFAULT_VOCAB_SIZE } from './sampler.js';

export const REASONS = {
  BATCH_SHARED_RNG: 'batch-shared-rng: requests draw from one shared stream, so batch reorder changes per-request results',
  BRANCH_SHARED_STATE: 'branch-shared-state: beam branches copy the parent state instead of deriving their own, so branches collapse to identical streams',
  MERGE_CURSOR_ROLLBACK: 'merge-cursor-rollback: merge restores the pre-branch cursor, reusing committed random numbers',
  RETRY_REUSE: 'retry-reuse: retry restarts from the attempt-start state, replaying the committed random sequence',
};

class Sequence {
  constructor(id, state) {
    this.id = id;
    this.state = state;
    this.tokens = [];
    this.steps = [];
  }
}

export function replay(events, { policy = 'deterministic', vocabSize = DEFAULT_VOCAB_SIZE } = {}) {
  const requests = new Map();
  const records = [];
  let sharedState = null;
  let stepCounter = 0;

  const ensureRequest = (id) => {
    const request = requests.get(id);
    if (!request) throw new Error(`unknown request: ${id}`);
    return request;
  };

  const activeSequence = (request) => request.sequences[request.sequences.length - 1];

  const drawUniform = (request, sequence) => {
    const stateBefore = { ...sequence.state };
    const { value, next } = draw(sequence.state);
    sequence.state = next;
    return { uniform: value, stateBefore };
  };

  for (const event of events) {
    switch (event.type) {
      case 'batch': {
        const order = event.requests.map((r) => r.id);
        if (policy === 'naive') {
          sharedState = {
            base: mix64('batch', ...event.requests.map((r) => String(r.seed))),
            cursor: 0,
          };
        }
        event.requests.forEach((config, index) => {
          let state;
          if (policy === 'naive') {
            // Bug: request streams are dealt from one shared batch stream in
            // arrival order, so reordering the batch reshuffles every seed.
            const dealt = draw(sharedState);
            sharedState = dealt.next;
            state = { base: mix64('dealt', String(config.seed), String(dealt.value)), cursor: 0 };
          } else {
            // Request state depends only on (seed, request id), never on
            // batch position, so reordering cannot change the stream.
            state = seedRequest(config.seed, config.id);
          }
          const sequence = new Sequence(`${config.id}#0`, state);
          requests.set(config.id, {
            config,
            batchOrder: order,
            sequences: [sequence],
            retryAttempt: 0,
            attemptStartState: state ? { ...state } : null,
            branchCheckpoint: null,
          });
        });
        break;
      }
      case 'step': {
        const request = ensureRequest(event.requestId);
        const sequence = activeSequence(request);
        const position = sequence.tokens.length;
        const { uniform, stateBefore } = drawUniform(request, sequence);
        const logits = mockLogits(
          sequence.state.base,
          position,
          vocabSize,
        );
        const token = sampleToken(logits, request.config.params ?? {}, uniform);
        sequence.tokens.push(token);
        const record = {
          step: stepCounter,
          requestId: event.requestId,
          sequenceId: sequence.id,
          state: stateKey(stateBefore),
          token,
        };
        stepCounter += 1;
        sequence.steps.push(record);
        records.push(record);
        break;
      }
      case 'branch': {
        const request = ensureRequest(event.requestId);
        const parent = activeSequence(request);
        const count = event.count ?? 2;
        if (policy === 'naive') {
          request.branchCheckpoint = { ...parent.state };
          // Bug: branches copy the parent state; every branch shares one stream.
          for (let index = 1; index < count; index += 1) {
            const child = new Sequence(`${parent.id}.b${index}`, { ...parent.state });
            child.tokens = [...parent.tokens];
            request.sequences.push(child);
          }
        } else {
          for (let index = 1; index < count; index += 1) {
            const childState = deriveState(parent.state, 'branch', index);
            const child = new Sequence(`${parent.id}.b${index}`, childState);
            child.tokens = [...parent.tokens];
            request.sequences.push(child);
          }
        }
        break;
      }
      case 'merge': {
        const request = ensureRequest(event.requestId);
        if (request.sequences.length < 2) break;
        const keep = event.keep ?? request.sequences.length - 1;
        const kept = request.sequences[keep];
        if (policy === 'naive') {
          // Bug: roll the cursor back to the pre-branch checkpoint.
          kept.state = { ...request.branchCheckpoint };
        } else {
          // Never roll back: continue from the furthest committed cursor.
          const maxCursor = Math.max(...request.sequences.map((s) => s.state.cursor));
          kept.state = { base: kept.state.base, cursor: Math.max(kept.state.cursor, maxCursor) };
        }
        request.sequences = [kept];
        break;
      }
      case 'retry': {
        const request = ensureRequest(event.requestId);
        const sequence = activeSequence(request);
        if (policy === 'naive') {
          // Bug: restart from the attempt-start state, reusing committed draws.
          if (request.attemptStartState) sequence.state = { ...request.attemptStartState };
        } else {
          request.retryAttempt += 1;
          sequence.state = deriveState(
            request.attemptStartState,
            'retry',
            request.retryAttempt,
          );
          request.attemptStartState = { ...sequence.state };
        }
        break;
      }
      default:
        throw new Error(`unknown event type: ${event.type}`);
    }
  }

  const sequences = {};
  for (const [id, request] of requests) {
    sequences[id] = request.sequences.map((sequence) => ({
      sequenceId: sequence.id,
      tokens: sequence.tokens,
      finalState: stateKey(sequence.state),
    }));
  }
  return { policy, records, sequences };
}

export function compareRuns(reference, target) {
  const drift = {};
  for (const [id, refSequences] of Object.entries(reference.sequences)) {
    const targetSequences = target.sequences[id];
    if (!targetSequences) {
      drift[id] = { firstDivergence: 0, mismatches: -1, reason: 'missing request in target run' };
      continue;
    }
    const refTokens = refSequences.flatMap((s) => s.tokens);
    const targetTokens = targetSequences.flatMap((s) => s.tokens);
    let firstDivergence = -1;
    let mismatches = 0;
    const length = Math.max(refTokens.length, targetTokens.length);
    for (let index = 0; index < length; index += 1) {
      if (refTokens[index] !== targetTokens[index]) {
        mismatches += 1;
        if (firstDivergence === -1) firstDivergence = index;
      }
    }
    drift[id] = {
      firstDivergence,
      mismatches,
      referenceTokens: refTokens,
      targetTokens,
    };
  }
  return drift;
}

export function diagnose(events) {
  const reasons = [];
  const hasBranch = events.some((e) => e.type === 'branch');
  const hasMerge = events.some((e) => e.type === 'merge');
  const hasRetry = events.some((e) => e.type === 'retry');
  const hasBatch = events.some((e) => e.type === 'batch' && e.requests.length > 1);
  if (hasBatch) reasons.push(REASONS.BATCH_SHARED_RNG);
  if (hasBranch) reasons.push(REASONS.BRANCH_SHARED_STATE);
  if (hasMerge) reasons.push(REASONS.MERGE_CURSOR_ROLLBACK);
  if (hasRetry) reasons.push(REASONS.RETRY_REUSE);
  return reasons;
}

export function validate(events, options = {}) {
  const deterministic = replay(events, { ...options, policy: 'deterministic' });
  const naive = replay(events, { ...options, policy: 'naive' });
  return {
    deterministic,
    naive,
    drift: compareRuns(deterministic, naive),
    nonReproducibilityReasons: diagnose(events),
  };
}
