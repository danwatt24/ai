import { BPETable, Token } from "./types";

export function encode(text: string, table: BPETable): number[] {
  const tokens = splitIntoChars(text); // step 1: break into chars
  const merged = applyMerges(tokens, table); // step 2: repeatedly merge
  return merged.map((tok) => table.tokenToId.get(tok)!);
}

function splitIntoChars(text: string): Token[] {
  return [...text]; // for now this is fine
}

function applyMerges(tokens: Token[], table: BPETable): Token[] {
  let seq = tokens.slice();

  // Extract merge operations sorted by rank
  const merges = [...table.merges.values()].sort((a, b) => a.rank - b.rank);

  for (const { pair } of merges) {
    let changed = true;
    while (changed) {
      const next = mergeOnce(seq, pair);
      changed = next.length !== seq.length || next.some((t, i) => t !== seq[i]);
      seq = next;
    }
  }
  return tokens;
}

function mergeOnce(tokens: Token[], pair: [Token, Token]): Token[] {
  const [A, B] = pair;
  const out: Token[] = [];
  let i = 0;

  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === A && tokens[i + 1] === B) {
      out.push(A + B);
      i += 2;
    } else {
      out.push(tokens[i]);
      i += 1;
    }
  }

  return out;
}
