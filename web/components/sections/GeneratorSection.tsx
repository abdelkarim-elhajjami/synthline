"use client"

import * as React from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Card } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { Progress } from "@/components/ui/progress"
import {
    Command,
    CommandEmpty,
    CommandGroup,
    CommandInput,
    CommandItem,
    CommandList,
} from "@/components/ui/command"
import {
    Popover,
    PopoverContent,
    PopoverTrigger,
} from "@/components/ui/popover"

import { PromptPreview } from "@/components/PromptPreview"
import { ApiKeySettings } from "@/components/ApiKeySettings"
import { useSynthline } from "@/context/SynthlineContext"

export function GeneratorSection() {
    const {
        formData,
        handleInputChange,
        isGenerating,
        isOptimizingPrompt,
        progress,
        optimizationSuccess,
        handleOptimizePrompt,
        currentPrompt,
        atomicPrompts,
        optimizedAtomicPrompts,
        currentPromptIndex,
        isPromptOptimized,
        setCurrentPromptIndex,
        availableModels,
        loadingModels
    } = useSynthline()

    const [open, setOpen] = React.useState(false)
    const [searchValue, setSearchValue] = React.useState("")

    // Helper to find label
    const selectedModelLabel = React.useMemo(() => {
        for (const group of availableModels) {
            const found = group.items.find(item => item.value === formData.llm);
            if (found) return found.label;
        }
        return formData.llm || "Select Model";
    }, [availableModels, formData.llm]);

    return (
        <section className="space-y-4">
            <div className="flex justify-between items-center">
                <h2 className="text-2xl font-medium text-[#8A2BE2]">Generator</h2>
                <ApiKeySettings />
            </div>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
                <Label className="text-white mb-4 block text-base font-medium">LLM Settings</Label>

                <div className="mb-6">
                    <Popover open={open} onOpenChange={setOpen}>
                        <PopoverTrigger asChild>
                            <Button
                                variant="outline"
                                role="combobox"
                                aria-expanded={open}
                                className="w-full justify-between bg-[#1A1A1A] border-[#2A2A2A] text-white hover:bg-[#252525] hover:text-white"
                                disabled={isGenerating || isOptimizingPrompt || loadingModels}
                            >
                                {loadingModels ? "Loading models..." : selectedModelLabel}
                                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-[#1A1A1A] border-[#2A2A2A] text-white">
                            <Command className="bg-[#1A1A1A] text-white">
                                <CommandInput placeholder="Search model..." className="text-white" value={searchValue} onValueChange={setSearchValue} />
                                <CommandList>
                                    <CommandEmpty>No model found.</CommandEmpty>
                                    {availableModels.map((group, index) => {
                                        // Special handling for Searchable groups (OpenRouter, OpenAI) to implement "Search-Only" UI
                                        if (group.label === "OpenRouter (Searchable)" || group.label === "OpenAI (Searchable)") {
                                            return (
                                                <CommandGroup key={index} heading={group.label} className="text-zinc-400">
                                                    {!searchValue ? (
                                                        <CommandItem
                                                            disabled
                                                            className="hidden" // Hidden item forces group header to render
                                                            value="openrouter-placeholder"
                                                        >
                                                            placeholder
                                                        </CommandItem>
                                                    ) : (
                                                        group.items.map((option) => (
                                                            <CommandItem
                                                                key={option.value}
                                                                value={option.label}
                                                                onSelect={() => {
                                                                    handleInputChange('llm', option.value);
                                                                    setOpen(false);
                                                                }}
                                                                className="text-white aria-selected:bg-[#8A2BE2] aria-selected:text-white cursor-pointer"
                                                            >
                                                                <Check
                                                                    className={cn(
                                                                        "mr-2 h-4 w-4",
                                                                        formData.llm === option.value ? "opacity-100" : "opacity-0"
                                                                    )}
                                                                />
                                                                {option.label}
                                                            </CommandItem>
                                                        ))
                                                    )}
                                                </CommandGroup>
                                            );
                                        }

                                        // Default rendering for other groups (Local, OpenAI)
                                        return (
                                            <CommandGroup key={index} heading={group.label} className="text-zinc-400">
                                                {group.items.map((option) => (
                                                    <CommandItem
                                                        key={option.value}
                                                        value={option.label}
                                                        onSelect={() => {
                                                            handleInputChange('llm', option.value);
                                                            setOpen(false);
                                                        }}
                                                        className="text-white aria-selected:bg-[#8A2BE2] aria-selected:text-white cursor-pointer"
                                                    >
                                                        <Check
                                                            className={cn(
                                                                "mr-2 h-4 w-4",
                                                                formData.llm === option.value ? "opacity-100" : "opacity-0"
                                                            )}
                                                        />
                                                        {option.label}
                                                    </CommandItem>
                                                ))}
                                            </CommandGroup>
                                        );
                                    })}
                                </CommandList>
                            </Command>
                        </PopoverContent>
                    </Popover>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <Label className="text-white">Temperature</Label>
                            <span className="text-zinc-400 text-sm">{formData.temperature}</span>
                        </div>
                        <Slider
                            value={[formData.temperature]}
                            onValueChange={(values) => handleInputChange('temperature', values[0])}
                            max={2}
                            step={0.1}
                            className="py-4"
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-zinc-500">Controls randomness (0=deterministic, 2=random)</p>
                    </div>

                    <div className="space-y-2">
                        <div className="flex justify-between">
                            <Label className="text-white">Top P</Label>
                            <span className="text-zinc-400 text-sm">{formData.top_p}</span>
                        </div>
                        <Slider
                            value={[formData.top_p]}
                            onValueChange={(values) => handleInputChange('top_p', values[0])}
                            max={1}
                            step={0.1}
                            className="py-4"
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-zinc-500">Controls diversity (0=focused, 1=diverse)</p>
                    </div>
                </div>
            </Card>

            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
                <Label className="text-white mb-4 block text-base font-medium">Prompt Settings</Label>
                <div className="space-y-6">
                    <div className="space-y-2">
                        <Label className="text-white text-sm font-medium">Samples Per Prompt</Label>
                        <Input
                            type="number"
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={formData.samples_per_prompt}
                            onChange={(e) => handleInputChange('samples_per_prompt', Math.max(1, Number(e.target.value)))}
                            min={1}
                            required
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-zinc-500">Specify the number of samples to generate per prompt</p>
                    </div>

                    <div className="h-px bg-[#2A2A2A] my-4"></div>

                    <div className="space-y-2">
                        <Label className="text-white text-sm font-medium">Prompt Approach</Label>
                        <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4">
                            <RadioGroup
                                value={formData.prompt_approach}
                                onValueChange={(value) => handleInputChange('prompt_approach', value)}
                                className="space-y-2"
                                disabled={isGenerating || isOptimizingPrompt}
                            >
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="Default" id="default" className="text-[#8A2BE2]" />
                                    <Label htmlFor="default" className="text-white">Default</Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="PACE" id="pace" className="text-[#8A2BE2]" />
                                    <Label htmlFor="pace" className="text-white">PACE Optimization</Label>
                                </div>
                            </RadioGroup>
                        </div>
                    </div>

                    {formData.prompt_approach === "PACE" && (
                        <div className="space-y-4">
                            <div className="space-y-2">
                                <div className="flex justify-between">
                                    <Label className="text-white">Iterations</Label>
                                    <span className="text-zinc-400 text-sm">{formData.pace_iterations}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_iterations]}
                                    onValueChange={(values) => handleInputChange('pace_iterations', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-4"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                                <p className="text-xs text-zinc-500">Number of prompt refinement cycles to run</p>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between">
                                    <Label className="text-white">Actor-Critic Pairs</Label>
                                    <span className="text-zinc-400 text-sm">{formData.pace_actors}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_actors]}
                                    onValueChange={(values) => handleInputChange('pace_actors', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-4"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                                <p className="text-xs text-zinc-500">Number of parallel LLM evaluations per prompt</p>
                            </div>

                            <div className="space-y-2">
                                <div className="flex justify-between">
                                    <Label className="text-white">Candidates Per Iteration</Label>
                                    <span className="text-zinc-400 text-sm">{formData.pace_candidates}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_candidates]}
                                    onValueChange={(values) => handleInputChange('pace_candidates', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-4"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                                <p className="text-xs text-zinc-500">Number of alternative prompt candidates</p>
                            </div>

                            <Button
                                onClick={handleOptimizePrompt}
                                disabled={isOptimizingPrompt || isGenerating}
                                className="w-full bg-[#8A2BE2] hover:bg-opacity-80 text-white"
                            >
                                {isOptimizingPrompt ? "Optimizing..." : "Optimize Prompt"}
                            </Button>

                            {/* Progress bar for optimization */}
                            {isOptimizingPrompt && (
                                <div className="space-y-2">
                                    <div className="flex justify-end">
                                        <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
                                    </div>
                                    <Progress value={progress} className="h-2 bg-[#2A2A2A]" />
                                </div>
                            )}

                            {/* Optimization success notification */}
                            {optimizationSuccess && (
                                <div className="bg-green-900/30 border border-green-500 text-green-200 p-3 rounded-md">
                                    {optimizationSuccess}
                                </div>
                            )}
                        </div>
                    )}

                    <PromptPreview
                        currentPrompt={currentPrompt}
                        atomicPrompts={atomicPrompts}
                        optimizedAtomicPrompts={optimizedAtomicPrompts}
                        currentPromptIndex={currentPromptIndex}
                        setCurrentPromptIndex={setCurrentPromptIndex}
                        isGenerating={isGenerating}
                        isOptimizingPrompt={isOptimizingPrompt}
                        isPromptOptimized={isPromptOptimized}
                        promptApproach={formData.prompt_approach}
                    />
                </div>
            </Card>
        </section>
    );
}
