import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AtomicPrompt } from "@/app/types";

interface PromptPreviewProps {
    currentPrompt: string;
    atomicPrompts: AtomicPrompt[];
    optimizedAtomicPrompts: AtomicPrompt[];
    currentPromptIndex: number;
    setCurrentPromptIndex: (index: number) => void;
    isGenerating: boolean;
    isOptimizingPrompt: boolean;
    isPromptOptimized: boolean;
    promptApproach: string;
}

export function PromptPreview({
    currentPrompt,
    atomicPrompts,
    optimizedAtomicPrompts,
    currentPromptIndex,
    setCurrentPromptIndex,
    isGenerating,
    isOptimizingPrompt,
    isPromptOptimized,
    promptApproach
}: PromptPreviewProps) {

    const displayPrompt = promptApproach === "PACE" && isPromptOptimized && optimizedAtomicPrompts.length > 0
        ? optimizedAtomicPrompts[currentPromptIndex]?.prompt || ''
        : atomicPrompts.length > 0
            ? atomicPrompts[currentPromptIndex]?.prompt || ''
            : currentPrompt;

    const currentAtomicConfig = atomicPrompts[currentPromptIndex]?.config || {};
    const fmConstraints = Array.isArray((currentAtomicConfig as Record<string, unknown>).__fm_constraints)
        ? (currentAtomicConfig as Record<string, unknown>).__fm_constraints as Array<{ id: string; label: string; value: unknown }>
        : [];

    return (
        <div>
            <div className="flex justify-between items-center mb-3">
                <Label className="text-[var(--foreground)] text-sm font-medium">Prompt Preview</Label>

                {atomicPrompts.length > 1 && (
                    <div className="flex items-center space-x-2 text-sm">
                        <span className="text-[var(--foreground-muted)]">
                            {currentPromptIndex + 1} of {atomicPrompts.length}
                        </span>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPromptIndex(Math.max(0, currentPromptIndex - 1))}
                            disabled={currentPromptIndex === 0 || isGenerating || isOptimizingPrompt}
                            className="h-8 px-3 elegant-input text-[var(--foreground)]"
                        >
                            Prev
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPromptIndex(Math.min(atomicPrompts.length - 1, currentPromptIndex + 1))}
                            disabled={currentPromptIndex === atomicPrompts.length - 1 || isGenerating || isOptimizingPrompt}
                            className="h-8 px-3 elegant-input text-[var(--foreground)]"
                        >
                            Next
                        </Button>
                    </div>
                )}
            </div>

            <Textarea
                value={displayPrompt}
                className="elegant-input h-40 resize-none text-[var(--foreground)]"
                placeholder="Complete all required fields to see prompt preview"
                readOnly
                disabled={isGenerating || isOptimizingPrompt}
            />

            {atomicPrompts.length > 0 && (
                <div className="mt-3 text-xs text-[var(--foreground-muted)]">
                    <div className="font-medium mb-1">Current configuration:</div>
                    <div className="flex flex-wrap gap-2">
                        {fmConstraints.length > 0 ? (
                            fmConstraints.map((constraint) => (
                                <span key={constraint.id} className="bg-[var(--input-bg)] border border-[var(--input-border)] px-2 py-0.5 rounded text-xs">
                                    {constraint.label}: {Array.isArray(constraint.value) ? constraint.value.join(', ') : String(constraint.value)}
                                </span>
                            ))
                        ) : (
                            Object.entries(currentAtomicConfig)
                                .filter(([key]) => !key.startsWith('__'))
                                .filter(([, value]) => value && (Array.isArray(value) ? value.length > 0 : true))
                                .map(([key, value]) => (
                                    <span key={key} className="bg-[var(--input-bg)] border border-[var(--input-border)] px-2 py-0.5 rounded text-xs">
                                        {key.split('.').pop() || key}: {Array.isArray(value) ? value.join(', ') : value?.toString()}
                                    </span>
                                ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
