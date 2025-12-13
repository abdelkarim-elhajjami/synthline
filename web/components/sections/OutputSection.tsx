"use client"

import React from 'react';
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { useSynthline } from '@/context/SynthlineContext';

export function OutputSection() {
    const { formData, handleInputChange, isGenerating, isOptimizingPrompt } = useSynthline();

    return (
        <section className="space-y-4">
            <h2 className="text-2xl font-medium text-[#8A2BE2]">Output</h2>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">Total Samples</Label>
                        <Input
                            type="number"
                            className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                            value={formData.total_samples}
                            onChange={(e) => handleInputChange('total_samples', Number(e.target.value))}
                            required
                            disabled={isGenerating || isOptimizingPrompt}
                        />
                    </div>

                    <div className="space-y-2">
                        <Label className="text-white text-base font-medium">File Format</Label>
                        <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4">
                            <RadioGroup
                                value={formData.output_format}
                                onValueChange={(value) => handleInputChange('output_format', value as "CSV" | "JSON")}
                                className="flex space-x-4"
                                disabled={isGenerating || isOptimizingPrompt}
                            >
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="JSON" id="json" className="text-[#8A2BE2]" />
                                    <Label htmlFor="json" className="text-white">JSON</Label>
                                </div>
                                <div className="flex items-center space-x-2">
                                    <RadioGroupItem value="CSV" id="csv" className="text-[#8A2BE2]" />
                                    <Label htmlFor="csv" className="text-white">CSV</Label>
                                </div>
                            </RadioGroup>
                        </div>
                    </div>
                </div>
            </Card>
        </section>
    );
}
