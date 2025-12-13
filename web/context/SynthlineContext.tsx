"use client"

import React, { createContext, useContext, ReactNode } from 'react';
import { useSynthlineForm } from '@/hooks/useSynthlineForm';
import { useSynthlineWebSocket } from '@/hooks/useSynthlineWebSocket';
import { FormData, Results, AtomicPrompt } from '@/app/types';

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

    // Actions
    handleOptimizePrompt: () => Promise<void>;
    handleGenerate: () => Promise<void>;
}

const SynthlineContext = createContext<SynthlineContextType | undefined>(undefined);

export function SynthlineProvider({ children }: { children: ReactNode }) {
    const { formData, handleInputChange, hasValidValue, validateForm } = useSynthlineForm();

    const wsState = useSynthlineWebSocket({
        formData,
        hasValidValue,
        validateForm
    });

    const value = {
        formData,
        handleInputChange,
        hasValidValue,
        validateForm,
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
