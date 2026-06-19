const baseUrl = "http://localhost:3000";

async function* sendPrompt(prompt: string) {
  const res = await fetch(`${baseUrl}/chat`, {
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

async function getContext<T>() {
  const res = await fetch(`${baseUrl}/context`, {
    method: "get",
  });

  return (await res.json()) as T;
}

export default {
  sendPrompt,
  getContext,
};
