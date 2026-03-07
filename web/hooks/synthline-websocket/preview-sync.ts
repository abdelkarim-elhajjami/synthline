import { FormData, AtomicPrompt } from "@/app/types";
import { getSessionId } from "@/lib/session";
import { useEffect } from "react";

interface PreviewSyncParams {
    formData: FormData;
    validateForm: () => string;
    fmVersion: number;
    setCurrentPrompt: (value: string) => void;
    setAtomicPrompts: (value: AtomicPrompt[]) => void;
    setCurrentPromptIndex: (value: number) => void;
    setIsPromptOptimized: (value: boolean) => void;
    setOptimizedAtomicPrompts: (value: AtomicPrompt[]) => void;
}

export function usePreviewSync(params: PreviewSyncParams): void {
    const {
        formData,
        validateForm,
        fmVersion,
        setCurrentPrompt,
        setAtomicPrompts,
        setCurrentPromptIndex,
        setIsPromptOptimized,
        setOptimizedAtomicPrompts,
    } = params;

    const fetchPreview = async () => {
        const cfg = formData.fm_configuration;
        const hasSelections =
            Object.values(cfg.selected_options || {}).some((value) => value && value.length > 0) ||
            Object.values(cfg.string_values || {}).some((value) => value && value.length > 0) ||
            (cfg.selected_features || []).length > 0;

        const validationError = validateForm();

        if (!validationError && hasSelections) {
            try {
                const response = await fetch("/api/preview-prompt", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "X-Session-ID": getSessionId()
                    },
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
            } catch (error) {
                console.error("Preview generation failed:", error);
            }
        } else {
            setCurrentPrompt("");
            setAtomicPrompts([]);
        }
    };

    useEffect(() => {
        let cancelled = false;
        setIsPromptOptimized(false);
        setOptimizedAtomicPrompts([]);

        const run = async () => {
            if (!cancelled) {
                await fetchPreview();
            }
        };
        void run();

        return () => {
            cancelled = true;
        };
    }, [
        JSON.stringify(formData.fm_configuration),
        formData.classification_label,
        formData.classification_label_def,
        formData.samples_per_prompt,
        formData.prompt_approach,
        formData.llm,
        validateForm,
        fmVersion,
        setIsPromptOptimized,
        setOptimizedAtomicPrompts,
    ]);
}
