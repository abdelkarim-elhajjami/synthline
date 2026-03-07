"use client"

import { SingleInputField } from "@/components/SingleInputField";
import { useSynthline } from '@/context/SynthlineContext';

export function ClassificationSection() {
    const { formData, handleInputChange, isGenerating, isOptimizingPrompt } = useSynthline();
    const isDisabled = isGenerating || isOptimizingPrompt;

    return (
        <section className="space-y-4">
            <h2 className="section-heading">Classification</h2>
            <div className="elegant-card p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <SingleInputField
                        fieldName="classification_label"
                        placeholder="Enter the label"
                        label="Label"
                        value={formData.classification_label || ''}
                        onInputChange={handleInputChange}
                        disabled={isDisabled}
                    />
                    <SingleInputField
                        fieldName="classification_label_def"
                        placeholder="Enter the label definition"
                        label="Label Definition"
                        value={formData.classification_label_def || ''}
                        onInputChange={handleInputChange}
                        disabled={isDisabled}
                    />
                </div>
            </div>
        </section>
    );
}
