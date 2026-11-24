const MASK64 = (1n << 64n) - 1n;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;

export function fnv1a64(input) {
  const text = String(input);
  let hash = FNV_OFFSET;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= BigInt(text.charCodeAt(index));
    hash = (hash * FNV_PRIME) & MASK64;
  }
  return hash;
}

export function mix64(...parts) {
  let hash = FNV_OFFSET;
  for (const part of parts) {
    const value = typeof part === 'bigint' ? part : fnv1a64(part);
    hash ^= value;
    hash = (hash * FNV_PRIME) & MASK64;
    hash ^= hash >> 29n;
  }
  return hash;
}

export function splitmix64(state) {
  const nextState = (state + 0x9e3779b97f4a7c15n) & MASK64;
  let z = nextState;
  z = ((z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n) & MASK64;
  z = ((z ^ (z >> 27n)) * 0x94d049bb133111ebn) & MASK64;
  z = z ^ (z >> 31n);
  return { state: nextState, value: z };
}

export function toUnit(value) {
  return Number(value >> 11n) / 2 ** 53;
}

// A request RNG state is (base, cursor). The base is derived only from the
// request seed and request id, never from batch position, so reordering a
// batch cannot change a request's stream. The cursor counts committed draws.
export function stateForDraw(state) {
  return mix64(state.base, BigInt(state.cursor));
}

export function draw(state) {
  const { value } = splitmix64(stateForDraw(state));
  return { value: toUnit(value), next: { base: state.base, cursor: state.cursor + 1 } };
}

export function seedRequest(seed, requestId) {
  return { base: mix64('request', String(seed), String(requestId)), cursor: 0 };
}

// Derive a child state (beam branch, retry attempt) deterministically from a
// parent state and a label. The child starts at cursor 0 of its own stream.
export function deriveState(parent, ...labels) {
  return {
    base: mix64(parent.base, BigInt(parent.cursor), ...labels.map(String)),
    cursor: 0,
  };
}

export function stateKey(state) {
  return `${state.base.toString(16)}:${state.cursor}`;
}
