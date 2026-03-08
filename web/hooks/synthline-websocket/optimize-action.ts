import { FormData, GenerationOutput } from "@/app/types";
import { getSessionId } from "@/lib/session";

import { UiError } from "./types";

interface OptimizeActionParams {
    formData: FormData;
    validateForm: () => string;
    apiKeys?: Record<string, string>;
    connectionId: string;
    wsReady: boolean;
    operationId: string;
    setUiError: (value: UiError | null) => void;
    setIsOptimizingPrompt: (value: boolean) => void;
    setOutput: (value: GenerationOutput | null) => void;
    setProgress: (value: number) => void;
    setStatus: (value: string) => void;
    toUserFriendlyError: (raw: string) => string;
}

export async function runOptimizeAction(params: OptimizeActionParams): Promise<boolean> {
    const errorMessage = params.validateForm();
    if (errorMessage) {
        params.setUiError({ operation: "optimization", message: errorMessage });
        return false;
    }

    if (!params.wsReady) {
        params.setUiError({ operation: "optimization", message: "Connecting to server, please wait..." });
        return false;
    }

    params.setIsOptimizingPrompt(true);
    params.setOutput(null);
    params.setUiError(null);
    params.setProgress(0);
    params.setStatus("Optimizing prompts");

    try {
        const response = await fetch("/api/optimize-prompt", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-Session-ID": getSessionId()
            },
            body: JSON.stringify({
                features: params.formData,
                connection_id: params.connectionId,
                operation_id: params.operationId,
                api_keys: params.apiKeys
            })
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || "Prompt optimization failed");
        }
        return true;
    } catch (err) {
        params.setUiError({
            operation: "optimization",
            message: params.toUserFriendlyError(
                err instanceof Error ? err.message : "An error occurred during prompt optimization"
            ),
        });
        params.setIsOptimizingPrompt(false);
        return false;
    }
}
