"use client"

import * as React from "react";

import { Button } from "@/components/ui/button";
import { ClassificationSection } from "@/components/sections/ClassificationSection";
import { FeatureModelSection } from "@/components/sections/FeatureModelSection";
import { ArtefactSection } from "@/components/sections/ArtefactSection";
import { GenerationSection } from "@/components/sections/GenerationSection";
import { OutputDisplay } from "@/components/OutputDisplay";
import { SynthlineProvider, useSynthline } from "@/context/SynthlineContext";

function SynthlineContent() {
  const {
    uiError,
    output,
    status,
    progress,
    isGenerating,
    handleGenerate,
    isOptimizingPrompt,
    formData,
    apiKeys,
    root,
  } = useSynthline();
  const errorBannerRef = React.useRef<HTMLDivElement | null>(null);

  React.useEffect(() => {
    if (!uiError || !errorBannerRef.current) {
      return;
    }

    errorBannerRef.current.scrollIntoView({
      behavior: "smooth",
      block: "center",
    });
  }, [uiError]);

  // Check if selected model requires a key that's missing
  const missingKeyMessage = React.useMemo(() => {
    const model = formData.llm || '';
    if (model.startsWith('openrouter/') && !apiKeys.openrouter) {
      return 'OpenRouter API key required';
    }
    if (model.startsWith('openai/') && !apiKeys.openai) {
      return 'OpenAI API key required';
    }
    return null;
  }, [formData.llm, apiKeys]);

  const isGenerateDisabled = isGenerating || isOptimizingPrompt || !formData.llm || !!missingKeyMessage;

  return (
    <div className="min-h-screen">

      <header className="bg-[var(--purple)] py-12 mb-12 text-center shadow-sm">
        <h1 className="font-serif text-4xl md:text-5xl font-black mb-4 text-[var(--highlight)] drop-shadow-sm">
          Synthline
        </h1>
        <p className="text-white text-lg font-bold opacity-90">
          Feature Model–Guided Synthetic Data Generator
        </p>
      </header>

      <div className="container mx-auto pb-12 px-6 max-w-5xl">

        {uiError && (
          <div
            ref={errorBannerRef}
            role="alert"
            aria-live="assertive"
            className="sticky top-4 z-20 bg-red-50 border border-red-200 text-red-700 p-4 rounded-xl mb-8 shadow-sm animate-in slide-in-from-top-2"
          >
            <p className="font-semibold mb-1">
              {uiError.operation === "optimization" ? "Optimization failed" : "Generation failed"}
            </p>
            <p className="whitespace-pre-line">{uiError.message}</p>
          </div>
        )}

        <div className="space-y-8">
          <FeatureModelSection />

          {root && (
            <>
              <div className="elegant-divider"></div>

              <ClassificationSection />

              <div className="elegant-divider"></div>

              <ArtefactSection />

              <div className="elegant-divider"></div>

              <GenerationSection />

              <div className="space-y-4 pt-6">
                <Button
                  onClick={handleGenerate}
                  disabled={isGenerateDisabled}
                  title={missingKeyMessage || undefined}
                  className="w-full py-6 text-lg font-medium bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-xl transition-all duration-300 transform hover:translate-y-[-2px] hover:shadow-lg active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  {isGenerating ? 'Generating...' : missingKeyMessage || 'Generate'}
                </Button>

                {isGenerating && (
                  <div className="space-y-2 animate-in fade-in">
                    <div className="flex items-center justify-between gap-4">
                      <span className="text-sm text-[var(--foreground-muted)] truncate">{status || "Generating samples"}</span>
                      <span className="text-sm font-medium text-[var(--foreground-muted)] whitespace-nowrap">{Math.round(progress)}%</span>
                    </div>
                    <div className="h-2 bg-[var(--background-alt)] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[var(--purple)] rounded-full transition-all duration-300"
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>

              {output && (
                <div className="animate-in slide-in-from-bottom-10 fade-in duration-500">
                  <OutputDisplay
                    output={output}
                    status={status}
                    downloadFilename={
                      `${(formData.classification_label || 'output').trim().replace(/[^a-z0-9]+/gi, '_').toLowerCase()}_${formData.total_samples || 0}_${new Date().toISOString().split('T')[0]}`
                    }
                  />
                </div>
              )}
            </>
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
