"use client"

import { useEffect, useState } from "react";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { useSynthline } from '@/context/SynthlineContext';
import { RequiredLabel } from "@/components/RequiredLabel";

export function FeatureModelSection() {
    const { uploadFeatureModel, uploadGlossary, root, loadingFM, isGenerating, isOptimizingPrompt } = useSynthline();
    const [fmFile, setFmFile] = useState<File | null>(null);
    const [glossaryFile, setGlossaryFile] = useState<File | null>(null);
    const [uploadingFM, setUploadingFM] = useState(false);
    const [uploadingGlossary, setUploadingGlossary] = useState(false);
    const [fmStatus, setFmStatus] = useState<string | null>(null);
    const [fmError, setFmError] = useState<string | null>(null);
    const [glossaryStatus, setGlossaryStatus] = useState<string | null>(null);
    const [glossaryError, setGlossaryError] = useState<string | null>(null);

    const isDisabled = loadingFM || uploadingFM || uploadingGlossary || isGenerating || isOptimizingPrompt;

    useEffect(() => {
        if (!fmStatus) return;
        const timeoutId = window.setTimeout(() => setFmStatus(null), 4000);
        return () => window.clearTimeout(timeoutId);
    }, [fmStatus]);

    useEffect(() => {
        if (!glossaryStatus) return;
        const timeoutId = window.setTimeout(() => setGlossaryStatus(null), 4000);
        return () => window.clearTimeout(timeoutId);
    }, [glossaryStatus]);

    const handleFMUpload = async () => {
        if (!fmFile || isDisabled) return;
        setUploadingFM(true);
        setFmError(null);
        setFmStatus(null);

        try {
            await uploadFeatureModel(fmFile);
            setFmStatus(root ? "Feature model replaced." : "Feature model uploaded.");
            setFmFile(null);
        } catch (e) {
            const message = e instanceof Error ? e.message : "Failed to upload feature model.";
            setFmError(message);
        } finally {
            setUploadingFM(false);
        }
    };

    const handleGlossaryUpload = async () => {
        if (!glossaryFile || isDisabled) return;
        setUploadingGlossary(true);
        setGlossaryError(null);
        setGlossaryStatus(null);

        try {
            const result = await uploadGlossary(glossaryFile);
            setGlossaryStatus(result.replaced ? "Glossary replaced." : "Glossary uploaded.");
            setGlossaryFile(null);
        } catch (e) {
            const message = e instanceof Error ? e.message : "Failed to upload glossary.";
            setGlossaryError(message);
        } finally {
            setUploadingGlossary(false);
        }
    };

    return (
        <section className="space-y-3">
            <h2 className="section-heading">Feature Model</h2>
            <div className="elegant-card p-4 md:p-5 space-y-4">
                <div className="space-y-2">
                    <Label className="text-[var(--foreground)] text-sm font-medium">
                        <RequiredLabel required>Feature Model (.xml)</RequiredLabel>
                    </Label>
                    <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-3 items-center">
                        <input
                            type="file"
                            accept=".xml"
                            disabled={isDisabled}
                            onChange={(e) => {
                                setFmStatus(null);
                                setFmError(null);
                                setFmFile(e.target.files?.[0] || null);
                            }}
                            className="block w-full text-sm text-[var(--foreground-muted)] file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border file:border-[var(--input-border)] file:bg-[var(--input-bg)] file:text-[var(--foreground)] hover:file:border-[var(--purple)]"
                        />
                        <Button
                            onClick={handleFMUpload}
                            disabled={isDisabled || !fmFile}
                            className="w-44 justify-center bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all"
                        >
                            {uploadingFM ? "Uploading..." : "Upload New FM"}
                        </Button>
                    </div>
                </div>

                {fmStatus && (
                    <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg p-3">
                        {fmStatus}
                    </div>
                )}

                {fmError && (
                    <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                        {fmError}
                    </div>
                )}

                <div className="h-px bg-gradient-to-r from-transparent via-[var(--card-border)] to-transparent opacity-70 my-2" />

                <div className="space-y-2 pt-1">
                    <Label className="text-[var(--foreground)] text-sm font-medium">Glossary (.yaml)</Label>
                    <div className="grid grid-cols-1 md:grid-cols-[minmax(0,1fr)_auto] gap-3 items-center">
                        <input
                            type="file"
                            accept=".yaml,.yml"
                            disabled={isDisabled}
                            onChange={(e) => {
                                setGlossaryStatus(null);
                                setGlossaryError(null);
                                setGlossaryFile(e.target.files?.[0] || null);
                            }}
                            className="block w-full text-sm text-[var(--foreground-muted)] file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border file:border-[var(--input-border)] file:bg-[var(--input-bg)] file:text-[var(--foreground)] hover:file:border-[var(--purple)]"
                        />
                        <Button
                            onClick={handleGlossaryUpload}
                            disabled={isDisabled || !glossaryFile}
                            className="w-44 justify-center bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all"
                        >
                            {uploadingGlossary ? "Uploading..." : "Add New Glossary"}
                        </Button>
                    </div>
                </div>

                {glossaryStatus && (
                    <div className="text-sm text-green-700 bg-green-50 border border-green-200 rounded-lg p-3">
                        {glossaryStatus}
                    </div>
                )}

                {glossaryError && (
                    <div className="text-sm text-red-700 bg-red-50 border border-red-200 rounded-lg p-3">
                        {glossaryError}
                    </div>
                )}
            </div>
            {!root && (
                <div className="text-center text-[var(--foreground-muted)] pt-1">
                    {loadingFM ? "Loading Feature Model..." : "Upload a Feature Model to start configuring generation."}
                </div>
            )}
        </section>
    );
}
