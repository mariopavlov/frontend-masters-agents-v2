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
    description: "Read the contents of a file.",
    parameters: z.object( {
      path: z.string().describe("The path to the file to read."),
    })
  },
  writeFile: {
    description: "Write data to a file.",
    parameters: z.object( {
      path: z.string().describe("The path to the file to write."),
      content: z.string().describe("The content to write to the file."),
    })
  },
  listFiles: {
    description: "List all files in a directory.",
    parameters: z.object( {
      directory: z.string().describe("The directory to list files from."),
    })
  },
  deleteFile: {
    description: "Delete a file.",
    parameters: z.object( {
      path: z.string().describe("The path to the file to delete."),
    })
  },
  runCommand: {
    description: "Run a shell command.",
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

export const singleTurnExecutor = async ( data: EvalData ): Promise<SingleTurnResult> => {
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

  const { toolCalls } = await generateText({
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

  return {
    toolCalls: calls,
    toolNames,
    selectedAny: calls.length > 0,
  }
};
