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

export function useModelFetcher(apiKeys: ApiKeys) {
    const [models, setModels] = useState<GroupedModels[]>([]);
    const [loading, setLoading] = useState(false);

    const fetchModels = useCallback(async () => {
        setLoading(true);
        const newModels: GroupedModels[] = [];

        // 1. Local 
        newModels.push({
            label: "Local (Ollama)",
            items: DEFAULT_LOCAL_MODELS
        });

        // 2. OpenAI
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
                        newModels.push({ label: "OpenAI (Searchable)", items: data });
                    }
                }
            } catch (e) {
                console.error("Failed to fetch OpenAI models", e);
            }
        }

        // 3. OpenRouter
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
                    newModels.push({ label: "OpenRouter (Searchable)", items: data });
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
