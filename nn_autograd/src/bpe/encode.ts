import { MergeTable, Token } from "./types";

function splitIntoSymbols(segment: string): string[] {
  // JS iterates by codepoint, not UTF-16 unit
  return Array.from(segment);
}

export function bpeEncodeSegment(segment: string, merges: MergeTable): Token[] {
  let symbols = splitIntoSymbols(segment);

  if (symbols.length < 2) return symbols;

  while (true) {
    let bestRank = Infinity;
    let bestIndex = -1;

    // Find best adjacent merge
    for (let i = 0; i < symbols.length - 1; i++) {
      const pair = symbols[i] + symbols[i + 1];
      const rank = merges.ranks.get(pair);
      if (rank !== undefined && rank < bestRank) {
        bestRank = rank;
        bestIndex = i;
      }
    }

    // No valid merges remain
    if (bestIndex === -1) break;

    // Merge best pair
    const merged = symbols[bestIndex] + symbols[bestIndex + 1];

    symbols.splice(bestIndex, 2, merged);
  }

  return symbols;
}
