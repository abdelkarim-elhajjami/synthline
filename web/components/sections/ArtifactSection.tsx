"use client"

import React from 'react';
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { MultiSelectTags } from "@/components/MultiSelectTags";
import { MultiInputField } from "@/components/MultiInputField";
import { SPECIFICATION_FORMATS, SPECIFICATION_LEVELS, STAKEHOLDERS } from "@/app/constants";
import { useSynthline } from '@/context/SynthlineContext';

export function ArtifactSection() {
    const { formData, handleInputChange, isGenerating, isOptimizingPrompt } = useSynthline();

    return (
        <section className="space-y-4">
            <h2 className="text-2xl font-medium text-[#8A2BE2]">Requirements Artifact</h2>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
                <div className="space-y-6">
                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Specification Format</Label>
                        <MultiSelectTags
                            options={SPECIFICATION_FORMATS}
                            selected={formData.specification_format}
                            fieldName="specification_format"
                            onInputChange={handleInputChange}
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>

                    <div className="h-px bg-[#2A2A2A] my-4"></div>

                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Specification Level</Label>
                        <MultiSelectTags
                            options={SPECIFICATION_LEVELS}
                            selected={formData.specification_level}
                            fieldName="specification_level"
                            onInputChange={handleInputChange}
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>

                    <div className="h-px bg-[#2A2A2A] my-4"></div>

                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Stakeholder</Label>
                        <MultiSelectTags
                            options={STAKEHOLDERS}
                            selected={formData.stakeholder}
                            fieldName="stakeholder"
                            onInputChange={handleInputChange}
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>

                    <div className="h-px bg-[#2A2A2A] my-4"></div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <MultiInputField
                            fieldName="domain"
                            placeholder="e.g., Banking"
                            label="Domain (add multiple)"
                            values={formData.domain}
                            onInputChange={handleInputChange}
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                        <MultiInputField
                            fieldName="language"
                            placeholder="e.g., English"
                            label="Language (add multiple)"
                            values={formData.language}
                            onInputChange={handleInputChange}
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>
                </div>
            </Card>
        </section>
    );
}
