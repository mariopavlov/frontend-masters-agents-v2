import "dotenv/config";

import { generateText, type ModelMessage } from "ai";
import { ollama } from "ai-sdk-ollama";

import { SYSTEM_PROMPT } from "./system/prompt";
import type { AgentCallbacks } from "../types";

const MODEL_NAME = "kimi-k2.7-code:cloud";

export const runAgent = async (
  userMessage: string,
  conversationHistory: ModelMessage[],
  callbacks: AgentCallbacks,
) => {
  const { text } = await generateText({
    model: ollama(MODEL_NAME),
    prompt: userMessage,
    system: SYSTEM_PROMPT,
    temperature: 0.9,
  });

  console.log(text);
};

runAgent("Hello, can you hear me?");
