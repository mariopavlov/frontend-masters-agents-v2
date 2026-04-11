import { generateText, type ModelMessage } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { tools } from "./tools/index.ts";
import { getTracer, Laminar } from "@lmnr-ai/lmnr";
import { executeTools } from "./executeTools.ts";
import { SYSTEM_PROMPT } from "./system/prompt.ts";
import type { AgentCallbacks } from "../types.ts";

const ollama = createOpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "ollama",
});

const MODEL_NAME = "gemma4:26b";

Laminar.initialize({
  projectApiKey: process.env.LMNR_PROJECT_API_KEY,
});

export async function runAgent(
  userMessage: string,
  conversationHistory: ModelMessage[],
  callbacks: AgentCallbacks,
): Promise<any> {
  try {
    const { text, toolCalls } = await generateText({
      model: ollama(MODEL_NAME),
      system: SYSTEM_PROMPT,
      prompt: userMessage,
      tools: tools,
      experimental_telemetry: {
        isEnabled: true,
        tracer: getTracer(),
      },
    });

    console.log(text, toolCalls);

    toolCalls.forEach(async (tc) => {
      const result = await executeTools(tc.toolName, tc.input);
      console.log(result);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

await Laminar.flush();

runAgent(
  "What is the current time? Can you call 3 times the tool so that we are sure of the result",
);
