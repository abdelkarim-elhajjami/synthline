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

    return (
        <div>
            <div className="flex justify-between items-center mb-2">
                <Label className="text-white text-sm font-medium">Prompt Preview</Label>

                {atomicPrompts.length > 1 && (
                    <div className="flex items-center space-x-2 text-sm">
                        <span className="text-zinc-400">
                            {currentPromptIndex + 1} of {atomicPrompts.length} atomic prompts
                        </span>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPromptIndex(Math.max(0, currentPromptIndex - 1))}
                            disabled={currentPromptIndex === 0 || isGenerating || isOptimizingPrompt}
                            className="h-8 px-2"
                        >
                            Previous
                        </Button>
                        <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setCurrentPromptIndex(Math.min(atomicPrompts.length - 1, currentPromptIndex + 1))}
                            disabled={currentPromptIndex === atomicPrompts.length - 1 || isGenerating || isOptimizingPrompt}
                            className="h-8 px-2"
                        >
                            Next
                        </Button>
                    </div>
                )}
            </div>

            <Textarea
                value={displayPrompt}
                className="bg-[#1A1A1A] border-[#2A2A2A] text-white h-40 resize-none"
                placeholder="Complete all required fields to see prompt preview"
                readOnly
                disabled={isGenerating || isOptimizingPrompt}
            />

            {atomicPrompts.length > 0 && (
                <div className="mt-2 text-xs text-zinc-500">
                    <div className="font-medium text-zinc-400 mb-1">Current atomic configuration:</div>
                    {Object.entries(currentAtomicConfig)
                        .filter(([key]) => ['specification_format', 'specification_level', 'stakeholder', 'domain', 'language'].includes(key))
                        .map(([key, value]) => (
                            <div key={key} className="text-zinc-500">
                                • <span className="text-zinc-400">{key}:</span> {value?.toString()}
                            </div>
                        ))
                    }
                </div>
            )}
        </div>
    );
}
