import { useState, useEffect, useCallback } from "react";
import { v4 as uuidv4 } from 'uuid';
import { AtomicPrompt, Results, FormData } from "@/app/types";
import { REQUIRED_FIELDS } from "@/app/constants";

interface UseSynthlineWebSocketProps {
    formData: FormData;
    hasValidValue: (field: keyof FormData) => boolean;
    validateForm: () => string;
}

export function useSynthlineWebSocket({ formData, hasValidValue, validateForm }: UseSynthlineWebSocketProps) {
    const [connectionId] = useState(() => uuidv4());
    const [wsReady, setWsReady] = useState(false);
    const [progress, setProgress] = useState(0);
    const [status, setStatus] = useState("");
    const [error, setError] = useState("");

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

    // WebSocket connection
    useEffect(() => {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.host;
        const wsUrl = `${protocol}//${wsHost}/ws/${connectionId}`;
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => setWsReady(true);
        ws.onclose = () => setWsReady(false);

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);

                switch (data.type) {
                    case 'progress':
                        setProgress(data.progress);
                        break;

                    case 'prompt_update':
                        if (data.config_index !== undefined) {
                            setAtomicPrompts(prevPrompts => {
                                if (data.config_index < prevPrompts.length) {
                                    const updatedPrompts = [...prevPrompts];
                                    updatedPrompts[data.config_index] = {
                                        ...updatedPrompts[data.config_index],
                                        prompt: data.prompt,
                                        score: data.score
                                    };
                                    return updatedPrompts;
                                }
                                return prevPrompts;
                            });
                        } else {
                            setCurrentPrompt(data.prompt);
                        }
                        break;

                    case 'optimize_complete':
                        setIsOptimizingPrompt(false);
                        setProgress(100);
                        setCurrentPrompt(data.optimized_prompt);
                        setIsPromptOptimized(true);
                        setOptimizationSuccess(`Prompt optimization complete!`);
                        setTimeout(() => setOptimizationSuccess(null), 10000);
                        break;

                    case 'optimize_complete_batch':
                        setIsOptimizingPrompt(false);
                        setProgress(100);
                        const optimizedPrompts = data.optimized_results.map((result: {
                            prompt: string;
                            score: number;
                            atomic_config: Record<string, unknown>;
                        }) => ({
                            config: result.atomic_config as AtomicPrompt['config'],
                            prompt: result.prompt,
                            score: result.score
                        }));
                        setOptimizedAtomicPrompts(optimizedPrompts);
                        setIsPromptOptimized(true);
                        setCurrentPromptIndex(0);
                        setOptimizationSuccess(`Optimization complete!`);
                        setTimeout(() => setOptimizationSuccess(null), 10000);
                        break;

                    case 'generation_complete':
                        setIsGenerating(false);
                        setProgress(100);
                        setResults({
                            samples: data.samples,
                            output_content: data.output_content,
                            output_format: data.output_format,
                            fewer_samples_received: data.fewer_samples_received
                        });
                        setStatus(`Generation complete! ${data.samples.length} samples generated`);
                        if (data.fewer_samples_received) {
                            setStatus(prev => prev + " (Note: Fewer samples were received than requested due to token limits)");
                        }
                        break;

                    case 'error':
                        setError(data.message);
                        setIsOptimizingPrompt(false);
                        setIsGenerating(false);
                        break;

                    case 'complete':
                        setProgress(100);
                        break;
                }
            } catch (error) {
                console.error("WebSocket message parsing error:", error);
            }
        };

        return () => {
            setWsReady(false);
            ws.close();
        };
    }, [connectionId]);

    // Fetch prompt preview effect
    useEffect(() => {
        setIsPromptOptimized(false);
        setOptimizedAtomicPrompts([]);

        const missingRequiredFields = REQUIRED_FIELDS.filter(field => !hasValidValue(field));

        if (missingRequiredFields.length === 0) {
            const fetchPromptPreview = async () => {
                try {
                    const response = await fetch('/api/preview-prompt', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ features: formData })
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.atomic_prompts?.length > 0) {
                            setAtomicPrompts(data.atomic_prompts);
                            setCurrentPromptIndex(0);
                        } else {
                            setAtomicPrompts([]);
                            setCurrentPrompt(data.prompt);
                        }
                    }
                } catch (err) {
                    console.error('Preview generation failed:', err);
                }
            };

            fetchPromptPreview();
        } else {
            setCurrentPrompt("");
            setAtomicPrompts([]);
        }
    }, [
        formData.label,
        formData.label_definition,
        formData.domain,
        formData.language,
        formData.specification_format,
        formData.specification_level,
        formData.stakeholder,
        formData.samples_per_prompt,
        formData.prompt_approach,
        formData.llm,
        hasValidValue
    ]);

    const handleOptimizePrompt = async () => {
        const errorMessage = validateForm();
        if (errorMessage) {
            setError(errorMessage);
            return;
        }

        if (!wsReady) {
            setError("Connecting to server, please wait...");
            return;
        }

        setIsOptimizingPrompt(true);
        setResults(null);
        setError("");
        setProgress(0);
        setStatus("Optimizing...");

        try {
            const response = await fetch('/api/optimize-prompt', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    features: formData,
                    connection_id: connectionId
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Prompt optimization failed');
            }
            await response.json();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred during prompt optimization');
            setIsOptimizingPrompt(false);
        }
    };

    const handleGenerate = async () => {
        const errorMessage = validateForm();
        if (errorMessage) {
            setError(errorMessage);
            return;
        }

        if (!wsReady) {
            setError("Connecting to server, please wait...");
            return;
        }

        setResults(null);
        setIsGenerating(true);
        setStatus("Generating...");
        setError("");
        setProgress(0);

        try {
            const requestData = {
                features: { ...formData },
                connection_id: connectionId
            };

            if (formData.prompt_approach === "PACE" && isPromptOptimized) {
                if (optimizedAtomicPrompts.length > 0) {
                    (requestData.features as Record<string, unknown>).optimized_atomic_prompts = optimizedAtomicPrompts.map(p => ({
                        config: p.config,
                        optimized_prompt: p.prompt
                    }));
                } else if (currentPrompt) {
                    (requestData.features as Record<string, unknown>).optimized_prompt = currentPrompt;
                }
            }

            const response = await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Generation failed');
            }
            await response.json();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'An error occurred during generation');
            setStatus("Generation failed");
            setIsGenerating(false);
        }
    };

    return {
        progress,
        status,
        error,
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
        handleOptimizePrompt,
        handleGenerate
    };
}
