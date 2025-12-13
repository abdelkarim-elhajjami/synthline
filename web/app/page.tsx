"use client"

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { ClassificationSection } from "@/components/sections/ClassificationSection";
import { ArtifactSection } from "@/components/sections/ArtifactSection";
import { GeneratorSection } from "@/components/sections/GeneratorSection";
import { OutputSection } from "@/components/sections/OutputSection";
import { ResultsDisplay } from "@/components/ResultsDisplay";
import { SynthlineProvider, useSynthline } from "@/context/SynthlineContext";

function SynthlineContent() {
  const {
    error,
    results,
    status,
    progress,
    isGenerating,
    handleGenerate,
    isOptimizingPrompt
  } = useSynthline();

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      <div className="container mx-auto py-8 max-w-5xl">
        {/* Header */}
        <header className="mb-6 text-center">
          <h1 className="text-6xl font-bold mb-2 text-[#8A2BE2] animate-in fade-in zoom-in duration-500">Synthline</h1>
          <p className="text-zinc-400 text-xl">Generate High-Quality Synthetic Data for Requirements Engineering</p>
        </header>

        {/* Error message */}
        {error && (
          <div className="bg-red-900/30 border border-red-500 text-red-200 p-4 rounded-md mb-6 sticky top-2 z-10 animate-in slide-in-from-top-2">
            {error}
          </div>
        )}

        <div className="space-y-6">
          <ClassificationSection />

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          <ArtifactSection />

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          <GeneratorSection />

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          <OutputSection />

          {/* Action Buttons */}
          <div className="space-y-4 pt-8">
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || isOptimizingPrompt}
              className="w-full py-6 text-lg font-medium bg-[#8A2BE2] hover:bg-opacity-80 text-white transition-all transform hover:scale-[1.01] active:scale-[0.99]"
            >
              {isGenerating ? 'Generating...' : 'Generate Data'}
            </Button>

            {/* Progress bar with percentage */}
            {isGenerating && (
              <div className="space-y-2 animate-in fade-in">
                <div className="flex justify-end">
                  <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-[#2A2A2A]" />
              </div>
            )}
          </div>

          {/* Results */}
          {results && (
            <div className="animate-in slide-in-from-bottom-10 fade-in duration-500">
              <ResultsDisplay results={results} status={status} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function SynthlineApp() {
  return (
    <SynthlineProvider>
      <SynthlineContent />
    </SynthlineProvider>
  );
}