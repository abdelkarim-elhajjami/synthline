import { FormData, GenerationOutput } from "@/app/types";
import { getSessionId } from "@/lib/session";

import { responseErrorMessage } from "./http-error";
import { UiError } from "./types";

interface GenerateActionParams {
    formData: FormData;
    validateForm: () => string;
    apiKeys?: Record<string, string>;
    connectionId: string;
    wsReady: boolean;
    operationId: string;
    isPromptOptimized: boolean;
    optimizedAtomicPrompts: Array<{ config: Record<string, unknown>; prompt: string; score: number }>;
    setUiError: (value: UiError | null) => void;
    setOutput: (value: GenerationOutput | null) => void;
    setIsGenerating: (value: boolean) => void;
    setStatus: (value: string) => void;
    setProgress: (value: number) => void;
    toUserFriendlyError: (raw: string) => string;
}

export async function runGenerateAction(params: GenerateActionParams): Promise<boolean> {
    const errorMessage = params.validateForm();
    if (errorMessage) {
        params.setUiError({ operation: "generation", message: errorMessage });
        return false;
    }

    if (!params.wsReady) {
        params.setUiError({ operation: "generation", message: "Connecting to server, please wait..." });
        return false;
    }

    params.setOutput(null);
    params.setIsGenerating(true);
    params.setStatus("Generating samples");
    params.setUiError(null);
    params.setProgress(0);

    try {
        const requestData: {
            features: Record<string, unknown>;
            connection_id: string;
            operation_id: string;
            api_keys?: Record<string, string>;
        } = {
            features: { ...params.formData },
            connection_id: params.connectionId,
            operation_id: params.operationId,
        };

        if (params.formData.prompt_approach === "PACE" && params.isPromptOptimized && params.optimizedAtomicPrompts.length > 0) {
            requestData.features.optimized_atomic_prompts = params.optimizedAtomicPrompts.map((promptData) => ({
                config: promptData.config,
                optimized_prompt: promptData.prompt,
                pace_score: promptData.score,
            }));
        }

        if (params.apiKeys) {
            requestData.api_keys = params.apiKeys;
        }

        const response = await fetch("/api/generate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Session-ID": getSessionId()
            },
            body: JSON.stringify(requestData)
        });

        if (!response.ok) {
            throw new Error(await responseErrorMessage(response, "Generation failed"));
        }
        return true;
    } catch (err) {
        params.setUiError({
            operation: "generation",
            message: params.toUserFriendlyError(
                err instanceof Error ? err.message : "An error occurred during generation"
            ),
        });
        params.setStatus("Generation failed");
        params.setIsGenerating(false);
        return false;
    }
}
