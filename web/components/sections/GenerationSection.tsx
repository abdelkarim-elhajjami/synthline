"use client"

import * as React from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
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
import { RequiredLabel } from "@/components/RequiredLabel"

export function GenerationSection() {
    const {
        formData,
        handleInputChange,
        isGenerating,
        isOptimizingPrompt,
        progress,
        status,
        optimizationSuccess,
        handleOptimizePrompt,
        currentPrompt,
        atomicPrompts,
        optimizedAtomicPrompts,
        currentPromptIndex,
        isPromptOptimized,
        setCurrentPromptIndex,
        availableModels,
        loadingModels,
        apiKeys
    } = useSynthline()

    const [open, setOpen] = React.useState(false)
    const [searchValue, setSearchValue] = React.useState("")

    const selectedModelLabel = React.useMemo(() => {
        for (const group of availableModels) {
            const found = group.items.find(item => item.value === formData.llm);
            if (found) return found.label;
        }
        return formData.llm || "Select Model";
    }, [availableModels, formData.llm]);

    // Helper to get group heading with key status badge (only show when key is missing)
    const getGroupHeading = (label: string) => {
        if (label.includes('OpenAI') && !apiKeys.openai) {
            return (
                <span className="flex items-center gap-2">
                    {label}
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-600">
                        Key required
                    </span>
                </span>
            );
        }
        if (label.includes('OpenRouter') && !apiKeys.openrouter) {
            return (
                <span className="flex items-center gap-2">
                    {label}
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-100 text-red-600">
                        Key required
                    </span>
                </span>
            );
        }
        return label;
    };

    return (
        <section className="space-y-4">
            <div className="flex justify-between items-center">
                <h2 className="section-heading">Generation</h2>
                <ApiKeySettings />
            </div>

            <div className="elegant-card p-6">
                <Label className="text-[var(--foreground)] mb-4 block text-sm font-medium">
                    <RequiredLabel required indicatorClassName="text-orange-500">LLM Settings</RequiredLabel>
                </Label>

                <div className="mb-6">
                    <Popover open={open} onOpenChange={setOpen}>
                        <PopoverTrigger asChild>
                            <Button
                                variant="outline"
                                role="combobox"
                                aria-expanded={open}
                                className="w-full justify-between elegant-input text-[var(--foreground)] hover:border-[var(--purple)]"
                                disabled={isGenerating || isOptimizingPrompt || loadingModels}
                            >
                                {loadingModels ? "Loading models..." : selectedModelLabel}
                                <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                            </Button>
                        </PopoverTrigger>
                        <PopoverContent className="w-[--radix-popover-trigger-width] p-0 bg-white border-[var(--card-border)]">
                            <Command className="bg-white">
                                <CommandInput placeholder="Search model..." value={searchValue} onValueChange={setSearchValue} />
                                <CommandList>
                                    <CommandEmpty>No model found.</CommandEmpty>
                                    {availableModels.map((group, index) => {
                                        if (group.label.includes("Models via")) {
                                            return (
                                                <CommandGroup key={index} heading={getGroupHeading(group.label)} className="text-[var(--foreground-muted)]">
                                                    {!searchValue ? (
                                                        <CommandItem disabled className="hidden" value="placeholder">placeholder</CommandItem>
                                                    ) : (
                                                        group.items.map((option) => (
                                                            <CommandItem
                                                                key={option.value}
                                                                value={option.label}
                                                                onSelect={() => {
                                                                    handleInputChange('llm', option.value);
                                                                    setOpen(false);
                                                                }}
                                                                className="cursor-pointer aria-selected:bg-[var(--purple)] aria-selected:text-white"
                                                            >
                                                                <Check className={cn("mr-2 h-4 w-4", formData.llm === option.value ? "opacity-100" : "opacity-0")} />
                                                                {option.label}
                                                            </CommandItem>
                                                        ))
                                                    )}
                                                </CommandGroup>
                                            );
                                        }

                                        return (
                                            <CommandGroup key={index} heading={getGroupHeading(group.label)} className="text-[var(--foreground-muted)]">
                                                {group.items.map((option) => (
                                                    <CommandItem
                                                        key={option.value}
                                                        value={option.label}
                                                        onSelect={() => {
                                                            handleInputChange('llm', option.value);
                                                            setOpen(false);
                                                        }}
                                                        className="cursor-pointer aria-selected:bg-[var(--purple)] aria-selected:text-white"
                                                    >
                                                        <Check className={cn("mr-2 h-4 w-4", formData.llm === option.value ? "opacity-100" : "opacity-0")} />
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
                    <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-2">
                        Synthline requires strict JSON Schema structured outputs.
                    </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <div className="flex justify-between mb-2">
                            <Label className="text-[var(--foreground)] text-sm">Temperature</Label>
                            <span className="text-[var(--foreground-muted)] text-sm">{formData.temperature.toFixed(1)}</span>
                        </div>
                        <Slider
                            value={[formData.temperature]}
                            onValueChange={(values) => handleInputChange('temperature', values[0])}
                            max={2}
                            step={0.1}
                            className="py-2"
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-1.5">Controls randomness (0=deterministic, 2=random)</p>
                    </div>

                    <div>
                        <div className="flex justify-between mb-2">
                            <Label className="text-[var(--foreground)] text-sm">Top P</Label>
                            <span className="text-[var(--foreground-muted)] text-sm">{formData.top_p.toFixed(1)}</span>
                        </div>
                        <Slider
                            value={[formData.top_p]}
                            onValueChange={(values) => handleInputChange('top_p', values[0])}
                            max={1}
                            step={0.1}
                            className="py-2"
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-1.5">Controls diversity (0=focused, 1=diverse)</p>
                    </div>
                </div>
            </div>

            <div className="elegant-card p-6">
                <Label className="text-[var(--foreground)] mb-4 block text-sm font-medium">Prompt Settings</Label>
                <div className="space-y-6">
                    <div>
                        <Label className="text-[var(--foreground)] text-sm">Samples Per Prompt</Label>
                        <Input
                            type="number"
                            className="elegant-input mt-2"
                            value={formData.samples_per_prompt}
                            onChange={(e) => handleInputChange('samples_per_prompt', Math.max(1, Number(e.target.value)))}
                            min={1}
                            required
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-1.5">Number of samples the LLM generates per call</p>
                    </div>

                    <div className="elegant-divider"></div>

                    <div className="space-y-3">
                        <Label className="text-[var(--foreground)] text-sm">Prompt Approach</Label>
                        <div className="bg-[var(--input-bg)] border border-[var(--input-border)] rounded-lg p-4">
                            <RadioGroup
                                value={formData.prompt_approach}
                                onValueChange={(value) => handleInputChange('prompt_approach', value)}
                                className="space-y-2"
                                disabled={isGenerating || isOptimizingPrompt}
                            >
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="Default" id="default" className="text-[var(--purple)]" />
                                    <Label htmlFor="default" className="text-[var(--foreground)]">Default</Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="PACE" id="pace" className="text-[var(--purple)]" />
                                    <Label htmlFor="pace" className="text-[var(--foreground)]">PACE Optimization</Label>
                                </div>
                            </RadioGroup>
                        </div>
                    </div>

                    {formData.prompt_approach === "PACE" && (
                        <div className="space-y-4">
                            <div>
                                <div className="flex justify-between mb-2">
                                    <Label className="text-[var(--foreground)] text-sm">Iterations</Label>
                                    <span className="text-[var(--foreground-muted)] text-sm">{formData.pace_iterations}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_iterations]}
                                    onValueChange={(values) => handleInputChange('pace_iterations', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-2"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                            </div>

                            <div>
                                <div className="flex justify-between mb-2">
                                    <Label className="text-[var(--foreground)] text-sm">Actor-Critic Pairs</Label>
                                    <span className="text-[var(--foreground-muted)] text-sm">{formData.pace_actors}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_actors]}
                                    onValueChange={(values) => handleInputChange('pace_actors', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-2"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                            </div>

                            <div>
                                <div className="flex justify-between mb-2">
                                    <Label className="text-[var(--foreground)] text-sm">Candidates Per Iteration</Label>
                                    <span className="text-[var(--foreground-muted)] text-sm">{formData.pace_candidates}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_candidates]}
                                    onValueChange={(values) => handleInputChange('pace_candidates', values[0])}
                                    min={1}
                                    max={10}
                                    step={1}
                                    className="py-2"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                            </div>

                            <div>
                                <div className="flex justify-between mb-2">
                                    <Label className="text-[var(--foreground)] text-sm">Alignment Weight</Label>
                                    <span className="text-[var(--foreground-muted)] text-sm">{formData.pace_alpha.toFixed(1)}</span>
                                </div>
                                <Slider
                                    value={[formData.pace_alpha]}
                                    onValueChange={(values) => handleInputChange('pace_alpha', values[0])}
                                    min={0}
                                    max={1}
                                    step={0.1}
                                    className="py-2"
                                    disabled={isGenerating || isOptimizingPrompt}
                                />
                                <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-1.5">Balance between alignment and diversity scoring (0=diversity only, 1=alignment only)</p>
                            </div>

                            <Button
                                onClick={handleOptimizePrompt}
                                disabled={isOptimizingPrompt || isGenerating || !formData.llm}
                                className="w-full bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all"
                            >
                                {isOptimizingPrompt ? "Optimizing..." : "Optimize Prompt"}
                            </Button>

                            {isOptimizingPrompt && (
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between gap-4">
                                        <span className="text-sm text-[var(--foreground-muted)] truncate">{status || "Optimizing prompts"}</span>
                                        <span className="text-sm font-medium text-[var(--foreground-muted)] whitespace-nowrap">{Math.round(progress)}%</span>
                                    </div>
                                    <div className="h-2 bg-[var(--background-alt)] rounded-full overflow-hidden">
                                        <div className="h-full bg-[var(--purple)] rounded-full transition-all" style={{ width: `${progress}%` }} />
                                    </div>
                                </div>
                            )}

                            {optimizationSuccess && (
                                <div className="bg-green-50 border border-green-200 text-green-700 p-3 rounded-lg">
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
            </div>

            <div className="elegant-card p-6">
                <div className="space-y-4">
                    <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                            <Label htmlFor="align-verify" className="text-[var(--foreground)] text-sm font-medium">Alignment Verification</Label>
                            <Switch
                                id="align-verify"
                                checked={formData.align_verify}
                                onCheckedChange={(checked) => handleInputChange('align_verify', checked)}
                                disabled={isGenerating || isOptimizingPrompt}
                            />
                        </div>
                        <p className="text-xs text-[var(--foreground-muted)] opacity-70">
                            Verify generated samples against feature constraints using NLI-based scoring
                        </p>
                    </div>

                    {formData.align_verify && (
                        <div className="pt-1">
                            <div className="flex justify-between mb-2">
                                <Label className="text-[var(--foreground)] text-sm">Alignment Threshold</Label>
                                <span className="text-[var(--foreground-muted)] text-sm">{formData.align_threshold.toFixed(2)}</span>
                            </div>
                            <Slider
                                value={[formData.align_threshold]}
                                onValueChange={(values) => handleInputChange('align_threshold', values[0])}
                                min={0}
                                max={1}
                                step={0.05}
                                className="py-2"
                                disabled={isGenerating || isOptimizingPrompt}
                            />
                            <p className="text-xs text-[var(--foreground-muted)] opacity-70 mt-1.5">
                                Minimum alignment score to accept a sample (higher = stricter)
                            </p>
                        </div>
                    )}
                </div>
            </div>

            <div className="elegant-card p-6">
                <div className="space-y-2">
                    <Label className="text-[var(--foreground)] text-sm font-semibold">Total Samples</Label>
                    <Input
                        type="number"
                        value={formData.total_samples}
                        onChange={(e) => handleInputChange('total_samples', parseInt(e.target.value) || 0)}
                        className="elegant-input w-full text-left pl-3"
                        min={1}
                        disabled={isGenerating || isOptimizingPrompt}
                    />
                    <p className="text-xs text-[var(--foreground-muted)] opacity-70">
                        Specify the total number of samples to generate
                    </p>
                </div>
            </div>
        </section>
    );
}
