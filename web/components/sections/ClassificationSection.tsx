"use client"

import React from 'react';
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { useSynthline } from '@/context/SynthlineContext';

export function ClassificationSection() {
    const { formData, handleInputChange, isGenerating, isOptimizingPrompt } = useSynthline();

    return (
        <section className="space-y-4">
            <h2 className="text-2xl font-medium text-[#8A2BE2]">Classification</h2>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Label</Label>
                        <Input
                            type="text"
                            placeholder="e.g., Functional"
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={formData.label}
                            onChange={(e) => handleInputChange('label', e.target.value)}
                            required
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>
                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Label Definition</Label>
                        <Input
                            type="text"
                            placeholder="Define what this label means"
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={formData.label_definition}
                            onChange={(e) => handleInputChange('label_definition', e.target.value)}
                            required
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>
                </div>
            </Card>
        </section>
    );
}
