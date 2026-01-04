import { BPETable, Merge, Token } from "./types";

export interface BPETrainingConfig {
  vocabSize: number; // final desired vocab size
}

export function trainBPE(text: string, config: BPETrainingConfig): BPETable {
  let corpus = text.split("").map((c) => c as Token);

  const merges: Merge[] = [];
  const targetMerges = config.vocabSize - countUnique(corpus);

  for (let i = 0; i < targetMerges; i++) {
    const pairFreq = countPairFrequencies(corpus);
    const best = getMostFrequentPair(pairFreq);
    if (!best) break;

    merges.push({ pair: best, rank: i });
    corpus = mergePairInCorpus(corpus, best);
  }

  return buildTable(corpus, merges);
}

function countUnique(tokens: Token[]): number {
  return new Set(tokens).size;
}

function countPairFrequencies(tokens: Token[]): Map<string, number> {
  const freq = new Map<string, number>();
  for (let i = 0; i < tokens.length - 1; i++) {
    const pair = `${tokens[i]},${tokens[i + 1]}`;
    freq.set(pair, (freq.get(pair) || 0) + 1);
  }
  return freq;
}

function getMostFrequentPair(freq: Map<string, number>): [Token, Token] | null {
  let best: string | null = null;
  let maxCount = 0;

  for (const [pair, count] of freq.entries()) {
    if (count > maxCount) {
      best = pair;
      maxCount = count;
    }
  }

  if (!best) return null;

  const [a, b] = best.split(",");

  return [a as Token, b as Token];
}

function mergePairInCorpus(tokens: Token[], pair: [Token, Token]): Token[] {
  const [A, B] = pair;
  const merged = [];
  let i = 0;

  while (i < tokens.length) {
    if (i < tokens.length - 1 && tokens[i] === A && tokens[i + 1] === B) {
      merged.push(A + B); // merge into single token
      i += 2;
    } else {
      merged.push(tokens[i]);
      i += 1;
    }
  }
  return tokens;
}

function buildTable(finalCorpus: Token[], merges: Merge[]): BPETable {
  const table: BPETable = {
    merges: new Map(),
    tokenToId: new Map(),
    idToToken: new Map(),
  };

  for (const m of merges) {
    table.merges.set(`${m.pair[0]},${m.pair[1]}`, m);
  }

  const vocab = Array.from(new Set(finalCorpus));
  vocab.forEach((tok, i) => {
    table.tokenToId.set(tok, i);
    table.idToToken.set(i, tok);
  });

  return table;
}
