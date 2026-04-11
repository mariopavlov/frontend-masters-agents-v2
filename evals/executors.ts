import type {
  EvalData,
  SingleTurnResult,
  MultiTurnEvalData,
  MultiTurnResult,
} from "./types.ts";

import { generateText, stepCountIs, tool, zodSchema, type ToolSet } from "ai";
import { createOpenAI, openai } from "@ai-sdk/openai";
import z from "zod";
import { buildMessages } from "./utils.ts";

const TOOL_DEFINITIONS = {
  readFile: {
    description: "Use this tool to read the contents of a file provided by the user, no need to validate the path in advance",
    parameters: z.object( {
      path: z.string().describe("The path to the file to read."),
    })
  },
  writeFile: {
    description: "Use this tool to write data to a file provided by the user, no need to validate the path in advance",
    parameters: z.object( {
      path: z.string().describe("The path to the file to write."),
      content: z.string().describe("The content to write to the file."),
    })
  },
  listFiles: {
    description: "Use this tool to list all files in a directory provided by the user, no need to validate the directory in advance",
    parameters: z.object( {
      directory: z.string().describe("The directory to list files from."),
    })
  },
  deleteFile: {
    description: "Use this tool to delete a file provided by the user, no need to validate the path in advance",
    parameters: z.object( {
      path: z.string().describe("The path to the file to delete."),
    })
  },
  runCommand: {
    description: "Use this tool to run a shell command provided by the user, no need to validate the command in advance",
    parameters: z.object( {
      command: z.string().describe("The command to run."),
    })
  },
};

const ollama = createOpenAI({
  baseURL: "http://localhost:11434/v1",
  apiKey: "ollama",
});

const DEFAULT_MODEL = "gemma4:26b";

export const singleTurnExecutorWithMocks = async ( data: EvalData ): Promise<SingleTurnResult> => {
  const messages = buildMessages(data);

  const tools: ToolSet = {};

  for (const toolName of data.tools) {
    const def = TOOL_DEFINITIONS[toolName as keyof typeof TOOL_DEFINITIONS];

    if (def) {
      tools[toolName] = tool({
        description: def.description,
        inputSchema: zodSchema(def.parameters as z.ZodType<any>),
      })
    }
  }

  const { toolCalls, text, reasoning } = await generateText({
    model: ollama(data.config?.model ?? DEFAULT_MODEL),
    messages,
    tools,
    stopWhen: stepCountIs(1),
    temperature: data.config?.temperature ?? undefined,
  });

  const calls = toolCalls.map( (tc) => ( {
    toolName: tc.toolName,
    args: "args" in tc ? tc.args : {},
  }) );
  
  const toolNames = calls.map( (tc) => tc.toolName );

  const reasoningText = Array.isArray(reasoning)
    ? reasoning.map((r) => (typeof r === "string" ? r : r.text ?? "")).join("\n")
    : reasoning;

  return {
    toolCalls: calls,
    toolNames,
    selectedAny: calls.length > 0,
    text,
    reasoning: reasoningText,
  }
};
