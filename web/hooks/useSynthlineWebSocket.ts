import { useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";

import { AtomicPrompt, Results, FormData } from "@/app/types";
import { toUserFriendlyError } from "@/hooks/synthline-websocket/error-utils";
import { runGenerateAction } from "@/hooks/synthline-websocket/generate-action";
import { createMessageHandler } from "@/hooks/synthline-websocket/message-handler";
import { runOptimizeAction } from "@/hooks/synthline-websocket/optimize-action";
import { usePreviewSync } from "@/hooks/synthline-websocket/preview-sync";
import { useSocketLifecycle } from "@/hooks/synthline-websocket/socket-lifecycle";
import { OperationType, UiError } from "@/hooks/synthline-websocket/types";

interface UseSynthlineWebSocketProps {
    formData: FormData;
    validateForm: () => string;
    apiKeys?: Record<string, string>;
    fmVersion?: number;
}

export function useSynthlineWebSocket({ formData, validateForm, apiKeys, fmVersion = 0 }: UseSynthlineWebSocketProps) {
    const [connectionId] = useState(() => uuidv4());
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState("");
    const [uiError, setUiError] = useState<UiError | null>(null);

    // Prompt Optimization State
    const [isOptimizingPrompt, setIsOptimizingPrompt] = useState(false);
    const [isPromptOptimized, setIsPromptOptimized] = useState(false);
    const [optimizationSuccess, setOptimizationSuccess] = useState<string | null>(null);
    const [currentPrompt, setCurrentPrompt] = useState("");
    const [atomicPrompts, setAtomicPrompts] = useState<AtomicPrompt[]>([]);
    const [optimizedAtomicPrompts, setOptimizedAtomicPrompts] = useState<AtomicPrompt[]>([]);
    const [currentPromptIndex, setCurrentPromptIndex] = useState(0);

    // Generation State
    const [isGenerating, setIsGenerating] = useState(false);
    const [results, setResults] = useState<Results | null>(null);

    const activeOperationIdsRef = useRef<Record<OperationType, string | null>>({
        generation: null,
        optimization: null,
    });

    const setActiveOperationId = (operation: OperationType, operationId: string | null) => {
        activeOperationIdsRef.current[operation] = operationId;
    };

    const getActiveOperationId = (operation: OperationType): string | null => {
        return activeOperationIdsRef.current[operation];
    };

    const clearActiveOperationId = (operation: OperationType, operationId?: string) => {
        const current = activeOperationIdsRef.current[operation];
        if (!operationId || current === operationId) {
            activeOperationIdsRef.current[operation] = null;
        }
    };

    const resetRuntimeState = () => {
        setProgress(0);
        setStatus("");
        setUiError(null);
        setIsOptimizingPrompt(false);
        setIsPromptOptimized(false);
        setOptimizationSuccess(null);
        setCurrentPrompt("");
        setAtomicPrompts([]);
        setOptimizedAtomicPrompts([]);
        setCurrentPromptIndex(0);
        setIsGenerating(false);
        setResults(null);
        setActiveOperationId("generation", null);
        setActiveOperationId("optimization", null);
    };

    const socketMessageHandler = createMessageHandler({
        setProgress,
        setAtomicPrompts,
        setCurrentPrompt,
        setIsOptimizingPrompt,
        setIsPromptOptimized,
        setOptimizationSuccess,
        setOptimizedAtomicPrompts,
        setCurrentPromptIndex,
        setIsGenerating,
        setResults,
        setStatus,
        setUiError,
        toUserFriendlyError,
        getActiveOperationId,
        clearActiveOperationId,
    });

    const wsReady = useSocketLifecycle({
        connectionId,
        onMessage: socketMessageHandler,
    });

    usePreviewSync({
        formData,
        validateForm,
        fmVersion,
        setCurrentPrompt,
        setAtomicPrompts,
        setCurrentPromptIndex,
        setIsPromptOptimized,
        setOptimizedAtomicPrompts,
    });

    const handleOptimizePrompt = async () => {
        const operationId = uuidv4();
        setActiveOperationId("optimization", operationId);
        const started = await runOptimizeAction({
            formData,
            validateForm,
            apiKeys,
            connectionId,
            operationId,
            wsReady,
            setUiError,
            setIsOptimizingPrompt,
            setResults,
            setProgress,
            setStatus,
            toUserFriendlyError,
        });
        if (!started) {
            clearActiveOperationId("optimization", operationId);
        }
    };

    const handleGenerate = async () => {
        const operationId = uuidv4();
        setActiveOperationId("generation", operationId);
        const started = await runGenerateAction({
            formData,
            validateForm,
            apiKeys,
            connectionId,
            operationId,
            wsReady,
            isPromptOptimized,
            optimizedAtomicPrompts: optimizedAtomicPrompts.map((promptData) => ({
                config: promptData.config as Record<string, unknown>,
                prompt: promptData.prompt,
                score: promptData.score,
            })),
            setUiError,
            setResults,
            setIsGenerating,
            setStatus,
            setProgress,
            toUserFriendlyError,
        });
        if (!started) {
            clearActiveOperationId("generation", operationId);
        }
    };

    return {
        progress,
        status,
        uiError,
        currentPrompt,
        isGenerating,
        isOptimizingPrompt,
        results,
        optimizationSuccess,
        isPromptOptimized,
        atomicPrompts,
        optimizedAtomicPrompts,
        currentPromptIndex,
        setCurrentPromptIndex,
        wsReady,
        resetRuntimeState,
        handleOptimizePrompt,
        handleGenerate
    };
}
