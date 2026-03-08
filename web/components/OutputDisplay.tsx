"use client"

import { Button } from "@/components/ui/button";
import { GenerationOutput } from "@/app/types";

interface OutputDisplayProps {
    output: GenerationOutput;
    status: string;
    downloadFilename: string;
}

export function OutputDisplay({ output, status, downloadFilename }: OutputDisplayProps) {
    const runSuffix = `_${output.metadata.run_id}`;

    const downloadCSV = () => {
        const blob = new Blob([output.output_content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${downloadFilename}${runSuffix}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const downloadMetadata = () => {
        const blob = new Blob([JSON.stringify(output.metadata, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${downloadFilename}${runSuffix}_metadata.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="elegant-card p-6 mt-8">
            <div className="flex justify-between items-center">
                <h3 className="section-heading text-xl">Output</h3>
                <div className="flex items-center gap-2">
                    <Button
                        onClick={downloadCSV}
                        className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all"
                    >
                        Download CSV
                    </Button>
                    <Button
                        onClick={downloadMetadata}
                        variant="outline"
                        className="rounded-lg transition-all"
                    >
                        Download Metadata
                    </Button>
                </div>
            </div>

            <p className="text-[var(--foreground-muted)] text-sm whitespace-pre-line">{status}</p>
        </div>
    );
}
