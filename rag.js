import { pipeline } from "@xenova/transformers";
import chalk from "chalk";

// Cosine similarity helper
function cosineSimilarity(a, b) {
  let dot = 0,
    normA = 0,
    normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] ** 2;
    normB += b[i] ** 2;
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

// Get embedding vector from embedder pipeline
async function getEmbedding(embedder, text) {
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data); // Correct extraction
}

export class RAG {
  constructor(data) {
    this.data = data;
    this.embedder = null;
    this.generator = null;
  }

  async init() {
    this.embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2"
    );
    this.generator = await pipeline("text-generation", "Xenova/distilgpt2");

    // this.generator = await pipeline(
    //   "text2text-generation",
    //   "Xenova/flan-t5-large"
    // );
    // this.generator = await pipeline(
    //   "text2text-generation",
    //   "Xenova/flan-t5-small"
    // );

    // this.generator = await pipeline(
    //   "text-generation",
    //   "Xenova/gpt2" // or "Xenova/distilgpt2"
    // );

    console.log("Generating embeddings for data...");
    for (const item of this.data) {
      const combinedText = `${item.text} ${item.description || ""}`;
      item.embedding = await getEmbedding(this.embedder, combinedText);
    }
    console.log("Embeddings ready.");
  }

  async answerQuestion(query, topK = 2) {
    if (!this.embedder || !this.generator) {
      throw new Error("RAG not initialized. Call init() first.");
    }

    const queryEmbedding = await getEmbedding(this.embedder, query);

    const similarities = this.data.map((item) => ({
      id: item.id,
      text: item.text,
      description: item.description || "",
      score: cosineSimilarity(queryEmbedding, item.embedding),
    }));

    console.debug("Similarities:", similarities);

    similarities.sort((a, b) => b.score - a.score);
    const topChunks = similarities.slice(0, topK);

    const contextText = topChunks
      .map((c) => `${c.text} ${c.description}`)
      .join("\n");
    // const prompt = `Answer the question briefly.\nQuestion: ${query} \n Context:\n${contextText}\n\nAnswer:`;
    const prompt = `
You are a helpful assistant. Use ONLY the context below to answer the question. If the answer is not in the context, say "Information not available."

Context:
${contextText}


Question: ${query}
Answer:
`;
    console.log("Top chunks:", topChunks);
    console.log("context text:", contextText);
    console.log("query text:", query);
    console.log(chalk.bgCyan("Prompt:", prompt));

    // const output = await this.generator(prompt, {
    //   max_new_tokens: 300,
    //   temperature: 0.7,
    // });
    const output = await this.generator(prompt, {
      max_new_tokens: 50,
      temperature: 1.0,
      repetition_penalty: 1.2,
      // ⭐️ Creative responses in this LOWER MODEL
      // ⭐️ All these added because these model is not trained for RAG or lower quality model
      // This model is not trained for creativity
      // do_sample: true, // enable sampling (if supported)
      // top_k: 50, // limit sampling to top-k tokens (optional)
      // top_p: 0.9,
    });

    let answer = output[0].generated_text.replace(prompt, "").trim();
    // answer = answer.split("\n")[0].trim(); // Keep full sentence

    return answer;
  }
}
