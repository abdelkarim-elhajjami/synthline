"use client"

import React, { createContext, useContext, ReactNode, useState } from 'react';
import { useSynthlineForm } from '@/hooks/useSynthlineForm';
import { useSynthlineWebSocket } from '@/hooks/useSynthlineWebSocket';
import { useModelFetcher, GroupedModels } from '@/hooks/useModelFetcher';
import { FormData, Results, AtomicPrompt } from '@/app/types';

// Define ApiKeys locally if not in types
export interface ApiKeys {
    openai?: string;
    openrouter?: string;
    [key: string]: string | undefined;
}

interface SynthlineContextType {
    // Form Data
    formData: FormData;
    handleInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    hasValidValue: (field: keyof FormData) => boolean;
    validateForm: () => string;

    // Generation & WebSocket State
    wsReady: boolean;
    progress: number;
    status: string;
    error: string;
    isGenerating: boolean;
    results: Results | null;

    // Optimization State
    isOptimizingPrompt: boolean;
    isPromptOptimized: boolean;
    optimizationSuccess: string | null;
    currentPrompt: string;
    atomicPrompts: AtomicPrompt[];
    optimizedAtomicPrompts: AtomicPrompt[];
    currentPromptIndex: number;
    setCurrentPromptIndex: (index: number) => void;

    // API Keys State
    apiKeys: ApiKeys;
    setApiKeys: (keys: ApiKeys) => void;

    // Dynamic Models
    availableModels: GroupedModels[];
    loadingModels: boolean;
    refreshModels: () => Promise<void>;

    // Actions
    handleOptimizePrompt: () => Promise<void>;
    handleGenerate: () => Promise<void>;
}

const SynthlineContext = createContext<SynthlineContextType | undefined>(undefined);

export function SynthlineProvider({ children }: { children: ReactNode }) {
    const { formData, handleInputChange, hasValidValue, validateForm } = useSynthlineForm();
    const [apiKeys, setApiKeys] = useState<ApiKeys>({});

    // Model Fetching Hook
    const { availableModels, loadingModels, refreshModels } = useModelFetcher(apiKeys);

    // Cast apiKeys to compatible type for the hook if needed, or update hook type.
    // The hook expects Record<string, string>. Our ApiKeys has string | undefined.
    // We can filter out undefined values before passing or trust that it works at runtime.
    // To satisfy TS:
    const cleanApiKeys = Object.entries(apiKeys).reduce((acc, [key, value]) => {
        if (value) acc[key] = value;
        return acc;
    }, {} as Record<string, string>);

    const wsState = useSynthlineWebSocket({
        formData,
        hasValidValue,
        validateForm,
        apiKeys: cleanApiKeys
    });

    const value = {
        formData,
        handleInputChange,
        hasValidValue,
        validateForm,
        apiKeys,
        setApiKeys,
        availableModels,
        loadingModels,
        refreshModels,
        ...wsState
    };

    return (
        <SynthlineContext.Provider value={value}>
            {children}
        </SynthlineContext.Provider>
    );
}

export function useSynthline() {
    const context = useContext(SynthlineContext);
    if (context === undefined) {
        throw new Error('useSynthline must be used within a SynthlineProvider');
    }
    return context;
}
