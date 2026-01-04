export type Token = string;
export type TokenId = number;

export interface MergeTable {
  // key: concatenated token pair, e.g. "t" + "h" -> "th"
  ranks: Map<string, number>;
}

export interface Vocab {
  tokenToId: Map<Token, TokenId>;
  idToToken: Map<TokenId, Token>;
}

export interface Merge {
  pair: [Token, Token];
  rank: number;
}

export interface BPETable {
  merges: Map<string, Merge>;
  tokenToId: Map<Token, TokenId>;
  idToToken: Map<TokenId, Token>;
}
