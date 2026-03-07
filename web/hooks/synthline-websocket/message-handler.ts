import { AtomicPrompt, Results } from "@/app/types";
import { Dispatch, SetStateAction } from "react";

import { OperationType, UiError } from "./types";

interface MessageHandlerDeps {
    setProgress: Dispatch<SetStateAction<number>>;
    setAtomicPrompts: Dispatch<SetStateAction<AtomicPrompt[]>>;
    setCurrentPrompt: (value: string) => void;
    setIsOptimizingPrompt: (value: boolean) => void;
    setIsPromptOptimized: (value: boolean) => void;
    setOptimizationSuccess: (value: string | null) => void;
    setOptimizedAtomicPrompts: (value: AtomicPrompt[]) => void;
    setCurrentPromptIndex: (value: number) => void;
    setIsGenerating: (value: boolean) => void;
    setResults: (value: Results | null) => void;
    setStatus: (value: string) => void;
    setUiError: (value: UiError | null) => void;
    toUserFriendlyError: (rawMessage: string) => string;
    getActiveOperationId: (operation: OperationType) => string | null;
    clearActiveOperationId: (operation: OperationType, operationId?: string) => void;
}

export function createMessageHandler(deps: MessageHandlerDeps) {
    const setProgressMonotonic = (nextProgress: unknown): void => {
        if (typeof nextProgress !== "number" || Number.isNaN(nextProgress)) {
            return;
        }
        const clamped = Math.max(0, Math.min(100, nextProgress));
        deps.setProgress((previous) => Math.max(previous, clamped));
    };

    const resolveOperationType = (data: any): OperationType | null => {
        if (data?.operation === "generation" || data?.operation === "optimization") {
            return data.operation;
        }
        if (data?.type === "verification_progress" || data?.type === "generation_complete" || data?.type === "complete") {
            return "generation";
        }
        if (data?.type === "optimize_complete_batch" || data?.type === "prompt_update") {
            return "optimization";
        }
        return null;
    };

    const shouldHandleOperationEvent = (data: any): boolean => {
        const operation = resolveOperationType(data);
        if (!operation) {
            return true;
        }
        if (typeof data?.operation_id !== "string" || !data.operation_id) {
            return false;
        }
        const activeOperationId = deps.getActiveOperationId(operation);
        return !!activeOperationId && activeOperationId === data.operation_id;
    };

    return (data: any): void => {
        if (!shouldHandleOperationEvent(data)) {
            return;
        }

        switch (data.type) {
            case "progress":
                setProgressMonotonic(data.progress);
                if (typeof data.message === "string" && data.message.trim()) {
                    deps.setStatus(data.message);
                }
                break;

            case "prompt_update":
                if (typeof data.atomic_config_index === "number") {
                    const configIndex = data.atomic_config_index;
                    deps.setAtomicPrompts((prevPrompts) => {
                        if (configIndex < prevPrompts.length) {
                            const updatedPrompts = [...prevPrompts];
                            updatedPrompts[configIndex] = {
                                ...updatedPrompts[configIndex],
                                prompt: data.prompt,
                                score: data.score
                            };
                            return updatedPrompts;
                        }
                        return prevPrompts;
                    });
                } else {
                    deps.setCurrentPrompt(data.prompt);
                }
                break;

            case "verification_progress":
                setProgressMonotonic(data.progress);
                deps.setStatus(typeof data.message === "string" ? data.message : "Verifying alignment");
                break;

            case "optimize_complete_batch":
                deps.setIsOptimizingPrompt(false);
                deps.setProgress(100);
                deps.clearActiveOperationId("optimization", data.operation_id);
                deps.setOptimizedAtomicPrompts(
                    data.optimized_results.map((result: {
                        prompt: string;
                        score: number;
                        atomic_config: Record<string, unknown>;
                    }) => ({
                        config: result.atomic_config as AtomicPrompt["config"],
                        prompt: result.prompt,
                        score: result.score
                    }))
                );
                deps.setIsPromptOptimized(true);
                deps.setCurrentPromptIndex(0);
                deps.setOptimizationSuccess("Optimization complete!");
                deps.setStatus("Optimization complete");
                setTimeout(() => deps.setOptimizationSuccess(null), 10000);
                break;

            case "generation_complete":
                deps.setIsGenerating(false);
                deps.setProgress(100);
                deps.clearActiveOperationId("generation", data.operation_id);
                deps.setResults({
                    samples: data.samples,
                    output_content: data.output_content,
                    report: data.report,
                });
                deps.setStatus("Generation complete.");
                break;

            case "error": {
                const rawMessage = typeof data.message === "string" ? data.message : "Unexpected backend error";
                const operation: OperationType = typeof data.operation === "string"
                    ? data.operation
                    : (rawMessage.toLowerCase().includes("optimization error") ? "optimization" : "generation");
                deps.setUiError({
                    operation,
                    message: deps.toUserFriendlyError(rawMessage),
                });
                if (operation === "optimization") {
                    deps.setIsOptimizingPrompt(false);
                    deps.clearActiveOperationId("optimization", data.operation_id);
                } else {
                    deps.setIsGenerating(false);
                    deps.clearActiveOperationId("generation", data.operation_id);
                }
                break;
            }

            case "complete":
                if (typeof data.progress === "number") {
                    setProgressMonotonic(data.progress);
                } else {
                    setProgressMonotonic(100);
                }
                if (typeof data.message === "string" && data.message.trim()) {
                    deps.setStatus(data.message);
                }
                deps.clearActiveOperationId("generation", data.operation_id);
                break;
        }
    };
}
