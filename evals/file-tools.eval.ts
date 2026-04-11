import { evaluate } from "@lmnr-ai/lmnr";
import { toolSelectionScore } from "./evaluators";

import type { EvalData, EvalTarget } from "./types";

import dataset from "./data/file-tools.json" with { type: "json" };
import { singleTurnExecutorWithMocks } from "./executors";

const executor = async (data: EvalData) => {
  return await singleTurnExecutorWithMocks(data);
};

const MODELS = [
    "gemma4:26b",
    //"gemma4:31b",
    // "nemotron-cascade-2:30b",
    "glm-5.1:cloud",
    "minimax-m2.7:cloud",
    "qwen3.5:397b-cloud",
]

for (const model of MODELS) {
    const data = (dataset as any).map((entry: any) => ({
        ...entry,
        data: { ...entry.data, config: { ...entry.data.config, model } },
    }));

    evaluate({
        data,
        executor,
        evaluators: {
            selectionScore: (output, target: any) => {
                if (target.category === "secondary") {
                    return 1;
                }

                return toolSelectionScore(output, target);
            },
        },
        groupName: `file-tools-selection-${model}`,
    });
}
