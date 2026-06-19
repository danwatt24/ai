async function* sendPrompt(prompt: string) {
  const res = await fetch("http://localhost:3000/chat", {
    body: JSON.stringify({ prompt }),
    headers: {
      "Content-Type": "application/json",
    },
    method: "post",
  });

  const reader = res.body!.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) return;

    yield decoder.decode(value);
  }
}

export default {
  sendPrompt,
};
