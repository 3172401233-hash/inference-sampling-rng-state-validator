import { mix64, splitmix64, toUnit } from './rng.js';

export const DEFAULT_VOCAB_SIZE = 64;

// Mock logits: deterministic in (request base, position, token id) so the
// "model" itself never introduces nondeterminism. All scores are simulated.
export function mockLogits(requestBase, position, vocabSize = DEFAULT_VOCAB_SIZE) {
  const logits = new Array(vocabSize);
  for (let token = 0; token < vocabSize; token += 1) {
    const { value } = splitmix64(mix64('logit', requestBase, BigInt(position), BigInt(token)));
    logits[token] = toUnit(value) * 20 - 10;
  }
  return logits;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - max));
  const sum = exps.reduce((acc, value) => acc + value, 0);
  return exps.map((value) => value / sum);
}

export function sampleToken(logits, params, uniform) {
  const temperature = params.temperature ?? 1;
  const scaled = logits.map((logit) => logit / temperature);
  let probs = softmax(scaled).map((prob, token) => ({ token, prob }));

  if (params.topK != null && params.topK > 0 && params.topK < probs.length) {
    probs = probs.sort((a, b) => b.prob - a.prob).slice(0, params.topK);
  }
  if (params.topP != null && params.topP > 0 && params.topP < 1) {
    const sorted = [...probs].sort((a, b) => b.prob - a.prob);
    const kept = [];
    let cumulative = 0;
    for (const entry of sorted) {
      kept.push(entry);
      cumulative += entry.prob;
      if (cumulative >= params.topP) break;
    }
    probs = kept;
  }

  const total = probs.reduce((acc, entry) => acc + entry.prob, 0);
  let threshold = uniform * total;
  for (const entry of probs) {
    threshold -= entry.prob;
    if (threshold <= 0) return entry.token;
  }
  return probs[probs.length - 1].token;
}
