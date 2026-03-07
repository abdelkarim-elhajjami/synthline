import { useState, useEffect, useCallback } from 'react';
import { ApiKeys } from '@/context/SynthlineContext';

export interface ModelOption {
    value: string;
    label: string;
}

export interface GroupedModels {
    label: string;
    items: ModelOption[];
}

const DEFAULT_LOCAL_MODELS = [
    { value: "ollama/ministral-3:14b", label: "Ministral-3:14B" }
];

// Deployment : 'local' or 'hf' (defaults to 'hf')
const DEPLOYMENT = process.env.NEXT_PUBLIC_DEPLOYMENT || 'hf';
const IS_LOCAL = DEPLOYMENT === 'local';
const IS_HF = DEPLOYMENT === 'hf';

export function useModelFetcher(apiKeys: ApiKeys) {
    const [models, setModels] = useState<GroupedModels[]>([]);
    const [loading, setLoading] = useState(false);

    const fetchModels = useCallback(async () => {
        setLoading(true);
        const newModels: GroupedModels[] = [];

        // 1. Local (only if local deployment)
        if (IS_LOCAL) {
            newModels.push({
                label: "Local (Ollama)",
                items: DEFAULT_LOCAL_MODELS
            });
        }

        // 2. HuggingFace (only if HF deployment)
        if (IS_HF) {
            try {
                const res = await fetch('/api/models/fetch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ provider: 'huggingface' })
                });
                if (res.ok) {
                    const data: ModelOption[] = await res.json();
                    if (data.length > 0) {
                        newModels.push({ label: "Models via Hugging Face (Free) 🔍", items: data });
                    }
                }
            } catch (e) {
                console.error("Failed to fetch HuggingFace models", e);
            }
        }

        // 3. OpenAI (always show group, but only fetch if key is present)
        if (apiKeys.openai) {
            try {
                const res = await fetch('/api/models/fetch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        provider: 'openai',
                        api_key: apiKeys.openai
                    })
                });
                if (res.ok) {
                    const data = await res.json();
                    if (data.length > 0) {
                        newModels.push({ label: "Models via OpenAI 🔍", items: data });
                    }
                }
            } catch (e) {
                console.error("Failed to fetch OpenAI models", e);
            }
        } else {
            // Show empty OpenAI group so users know it exists
            newModels.push({ label: "Models via OpenAI 🔍", items: [] });
        }

        // 4. OpenRouter
        try {
            const res = await fetch('/api/models/fetch', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: 'openrouter',
                    api_key: apiKeys.openrouter || ""
                })
            });
            if (res.ok) {
                const data: ModelOption[] = await res.json();
                if (data.length > 0) {
                    newModels.push({ label: "Models via OpenRouter 🔍", items: data });
                }
            }
        } catch (e) {
            console.error("Failed to fetch OpenRouter models", e);
        }

        setModels(newModels);
        setLoading(false);
    }, [apiKeys.openai, apiKeys.openrouter]);

    useEffect(() => {
        fetchModels();
    }, [fetchModels]);

    return { availableModels: models, loadingModels: loading, refreshModels: fetchModels };
}
