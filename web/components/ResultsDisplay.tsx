import { Button } from "@/components/ui/button";
import { DownloadIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Results } from "@/app/types";

interface ResultsDisplayProps {
    results: Results;
    status: string;
    downloadFilename?: string;
}

export function ResultsDisplay({ results, status, downloadFilename }: ResultsDisplayProps) {
    const handleDownload = () => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const defaultName = `synthline_output_${timestamp}.${results.output_format}`;
        const filename = downloadFilename ? `${downloadFilename}.${results.output_format}` : defaultName;

        if (results?.output_path) {
            window.open(`/api/files/${encodeURIComponent(results.output_path)}`, '_blank');
        } else if (results?.output_content) {
            const blob = new Blob([results.output_content], { type: results.output_format === 'json' ? 'application/json' : 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };

    return (
        <Card className="bg-[#121212] border-[#1E1E1E] overflow-hidden shadow-md">
            <div className="border-b border-[#2A2A2A] p-4">
                <h3 className="text-lg font-medium text-[#8A2BE2]">Results</h3>
            </div>

            <div className="p-6">
                <div className="bg-[#1A1A1A] border-[#2A2A2A] p-4 rounded">
                    <div className="flex justify-between items-center mb-4">
                        <div>
                            <p className="text-white">{status}</p>
                            {results.output_path && (
                                <p className="text-zinc-400 text-sm">Output saved to: {results.output_path}</p>
                            )}
                        </div>
                        <Button
                            variant="ghost"
                            className="text-[#8A2BE2] hover:bg-[#8A2BE2]/10"
                            onClick={handleDownload}
                        >
                            <DownloadIcon className="h-4 w-4 mr-2" />
                            Download
                        </Button>
                    </div>
                </div>
            </div>
        </Card>
    );
}
