"use client"

import { Button } from "@/components/ui/button";
import { Results } from "@/app/types";

interface ResultsDisplayProps {
    results: Results;
    status: string;
    downloadFilename: string;
}

export function ResultsDisplay({ results, status, downloadFilename }: ResultsDisplayProps) {
    const runSuffix = `_${results.report.run_id}`;

    const downloadCSV = () => {
        const blob = new Blob([results.output_content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${downloadFilename}${runSuffix}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const downloadReport = () => {
        const blob = new Blob([JSON.stringify(results.report, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${downloadFilename}${runSuffix}_report.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="elegant-card p-6 mt-8">
            <div className="flex justify-between items-center">
                <h3 className="section-heading text-xl">Results</h3>
                <div className="flex items-center gap-2">
                    <Button
                        onClick={downloadCSV}
                        className="bg-[var(--purple)] hover:bg-[var(--purple-dark)] text-white rounded-lg transition-all"
                    >
                        Download CSV
                    </Button>
                    <Button
                        onClick={downloadReport}
                        variant="outline"
                        className="rounded-lg transition-all"
                    >
                        Download Report
                    </Button>
                </div>
            </div>

            <p className="text-[var(--foreground-muted)] text-sm whitespace-pre-line">{status}</p>
        </div>
    );
}
