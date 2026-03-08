"use client"

import { createContext, useContext, ReactNode, useState, useCallback, useMemo } from 'react';
import { useSynthlineForm } from '@/hooks/useSynthlineForm';
import { useSynthlineWebSocket } from '@/hooks/useSynthlineWebSocket';
import { useModelFetcher, GroupedModels } from '@/hooks/useModelFetcher';
import { FormData, GenerationOutput, AtomicPrompt, FMDocument, FMNode } from '@/app/types';
import { UiError } from '@/hooks/synthline-websocket/types';
import { useFM } from '@/hooks/useFM';

export interface ApiKeys {
    openai?: string;
    openrouter?: string;
    [key: string]: string | undefined;
}

interface SynthlineContextType {
    formData: FormData;
    handleInputChange: <K extends keyof FormData>(field: K, value: FormData[K]) => void;
    hasValidValue: (field: keyof FormData) => boolean;
    validateForm: () => string;

    fm: FMDocument | null;
    root: FMNode | null;
    index: FMDocument['index'];
    loadingFM: boolean;
    getNode: (nodeId: string) => FMNode | null;
    uploadFeatureModel: (file: File) => Promise<void>;
    uploadGlossary: (file: File) => Promise<{ replaced: boolean; entries: number }>;

    wsReady: boolean;
    progress: number;
    status: string;
    uiError: UiError | null;
    isGenerating: boolean;
    output: GenerationOutput | null;

    isOptimizingPrompt: boolean;
    isPromptOptimized: boolean;
    optimizationSuccess: string | null;
    currentPrompt: string;
    atomicPrompts: AtomicPrompt[];
    optimizedAtomicPrompts: AtomicPrompt[];
    currentPromptIndex: number;
    setCurrentPromptIndex: (index: number) => void;

    apiKeys: ApiKeys;
    setApiKeys: (keys: ApiKeys) => void;

    availableModels: GroupedModels[];
    loadingModels: boolean;
    refreshModels: () => Promise<void>;

    handleOptimizePrompt: () => Promise<void>;
    handleGenerate: () => Promise<void>;
}

const SynthlineContext = createContext<SynthlineContextType | undefined>(undefined);

export function SynthlineProvider({ children }: { children: ReactNode }) {
    const { formData, handleInputChange } = useSynthlineForm();
    const { fm, root, index, loadingFM, getNode, uploadFM, uploadGlossary: uploadGlossaryFile } = useFM();
    const [apiKeys, setApiKeys] = useState<ApiKeys>({});
    const [fmVersion, setFmVersion] = useState(0);

    const hasNodeSelection = useCallback((node: FMNode): boolean => {
        const cfg = formData.fm_configuration;
        const options = cfg.selected_options[node.id] || [];
        const values = cfg.string_values[node.id] || [];
        const selectedFeature = cfg.selected_features.includes(node.id);

        if (options.length > 0 || values.length > 0 || selectedFeature) {
            return true;
        }

        const prefix = `${node.id}.`;
        const hasSelectedDescendant = Object.entries(cfg.selected_options).some(([key, vals]) => key.startsWith(prefix) && vals.length > 0);
        const hasStringDescendant = Object.entries(cfg.string_values).some(([key, vals]) => key.startsWith(prefix) && vals.length > 0);
        const hasFeatureDescendant = cfg.selected_features.some(key => key.startsWith(prefix));

        return hasSelectedDescendant || hasStringDescendant || hasFeatureDescendant;
    }, [formData.fm_configuration]);

    const validateFMConfiguration = useCallback((): string => {
        if (!root) return '';

        const cfg = formData.fm_configuration;

        const walk = (node: FMNode): string | null => {
            if (node.node_type === 'alt' || node.node_type === 'or') {
                const selected = cfg.selected_options[node.id] || [];
                if (node.mandatory && selected.length === 0) {
                    return `${node.name} is required.`;
                }

                for (const childId of selected) {
                    const child = getNode(childId);
                    if (!child) continue;
                    const childError = walk(child);
                    if (childError) return childError;
                }
                return null;
            }

            const isStringNode = node.attributes.some(attr => attr.type.toLowerCase() === 'string');
            if (isStringNode) {
                const values = cfg.string_values[node.id] || [];
                if (node.mandatory && values.length === 0) {
                    return `${node.name} is required.`;
                }
            }

            for (const child of node.children) {
                if (!child.mandatory && !hasNodeSelection(child)) {
                    continue;
                }
                const childError = walk(child);
                if (childError) return childError;
            }

            return null;
        };

        return walk(root) || '';
    }, [formData.fm_configuration, getNode, hasNodeSelection, root]);

    const hasValidValue = useCallback((field: keyof FormData): boolean => {
        const value = formData[field];
        if (field === 'fm_configuration') {
            return validateFMConfiguration() === '';
        }
        if (Array.isArray(value)) return value.length > 0;
        if (typeof value === 'string') return value.trim() !== '';
        return value !== undefined && value !== null;
    }, [formData, validateFMConfiguration]);

    const validateForm = validateFMConfiguration;

    const { availableModels, loadingModels, refreshModels } = useModelFetcher(apiKeys);

    const cleanApiKeys = useMemo(() => Object.entries(apiKeys).reduce((acc, [key, value]) => {
        if (value) acc[key] = value;
        return acc;
    }, {} as Record<string, string>), [apiKeys]);

    const wsState = useSynthlineWebSocket({
        formData,
        validateForm,
        apiKeys: cleanApiKeys,
        fmVersion,
    });

    const uploadFeatureModel = useCallback(async (file: File) => {
        await uploadFM(file);
        // Reset FM configuration to avoid stale node IDs from the previous model.
        handleInputChange('fm_configuration', {
            selected_options: {},
            string_values: {},
            selected_features: [],
            or_group_mode: {},
        });
        wsState.resetRuntimeState();
        setFmVersion(v => v + 1);
    }, [handleInputChange, uploadFM, wsState]);

    const uploadGlossary = useCallback(async (file: File) => {
        const result = await uploadGlossaryFile(file);
        setFmVersion(v => v + 1);
        return result;
    }, [uploadGlossaryFile]);

    const value = {
        formData,
        handleInputChange,
        hasValidValue,
        validateForm,
        fm,
        root,
        index,
        loadingFM,
        getNode,
        uploadFeatureModel,
        uploadGlossary,
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
