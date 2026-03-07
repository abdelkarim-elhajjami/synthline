"use client"

import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useSynthline } from '@/context/SynthlineContext';

export function OutputSection() {
    const { formData, handleInputChange, isGenerating, isOptimizingPrompt } = useSynthline();

    return (
        <section className="space-y-4">
            <h2 className="section-heading">Output</h2>
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
