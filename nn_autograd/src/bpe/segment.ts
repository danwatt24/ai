const SEGMENT_REGEX = new RegExp(
  [
    // 1. new lines (keep runs)
    `\\n+`,

    // 2. whitespace (spaces, tabs, etc - but not newlines)
    `[ \\t\\r\\f\\v]+`,

    // 3. emoji clusters (approx, js-safe)
    // converts most emoji + modifiers + ZWJ sequences
    `(?:` +
      `\\p{Extended_Pictographic}` +
      `(?:\\uFE0F|\\uFE0E)?` +
      `(?:\\u200D\\p{Extended_Pictographic}` +
      `(?:\\uFE0F|\\uFE0E)?)*` +
      `)`,

    // 4. letters (unicod words)
    `\\p{L}+`,

    // 5. numbers
    `\\p{N}+`,

    // 6. punctuation / symbols (runs)
    `[\\p{P}\\p{S}]+`,
  ].join("|"),
  "gu"
);

export function segment(text: string): string[] {
  const segments: string[] = [];

  let lastIndex = 0;
  for (const match of text.matchAll(SEGMENT_REGEX)) {
    const start = match.index;
    const end = start + match[0].length;

    // If there's a gap (should be rare), emit chars one by one
    if (start > lastIndex) {
      for (const ch of text.slice(lastIndex, start)) {
        segments.push(ch);
      }
    }

    segments.push(match[0]);
    lastIndex = end;
  }

  // trailing gap
  if (lastIndex < text.length) {
    for (const ch of text.slice(lastIndex)) {
      segments.push(ch);
    }
  }

  return segments;
}
