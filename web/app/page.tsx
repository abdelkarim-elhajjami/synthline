"use client"

import { useState, useEffect } from "react";
import { v4 as uuidv4 } from 'uuid';
import { DownloadIcon } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Progress } from "@/components/ui/progress";
import { Input } from "@/components/ui/input";

interface FormData {
  label: string;
  label_definition: string;
  domain: string;
  language: string;
  output_format: "CSV" | "JSON";
  temperature: number;
  top_p: number;
  total_samples: number;
  samples_per_prompt: number;
  llm: string;
  specification_format: string;
  specification_level: string;
  stakeholder: string;
}

interface Sample {
  text: string;
  label: string;
  domain: string;
  language: string;
}

interface Results {
  samples: Sample[];
  output_path: string;
  fewer_samples_received?: boolean;
}

// Constants for form options
const SPECIFICATION_FORMATS = ["NL", "Constrained NL", "Use Case", "User Story"];
const SPECIFICATION_LEVELS = ["High", "Detailed"];
const STAKEHOLDERS = ["End Users", "Business Managers", "Developers", "Regulatory Bodies"];
const LLM_OPTIONS = [
  { value: "gpt-4o", label: "gpt-4o" },
  { value: "deepseek-chat", label: "deepseek-chat" }
];

export default function SynthlineApp() {
  // Initial form state
  const initialFormState: FormData = {
    label: "",
    label_definition: "",
    domain: "",
    language: "",
    output_format: "CSV",
    temperature: 0.6,
    top_p: 0.9,
    total_samples: 10,
    samples_per_prompt: 5,
    llm: "gpt-4o",
    specification_format: "",
    specification_level: "",
    stakeholder: "",
  };

  // State
  const [formData, setFormData] = useState<FormData>(initialFormState);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [currentPrompt, setCurrentPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Results | null>(null);
  const [connectionId] = useState(() => uuidv4());
  
  // WebSocket handling
  useEffect(() => {
    let reconnectAttempt = 0;
    let ws: WebSocket | null = null;
    
    const connectWebSocket = () => {
      // Close existing connection if any
      if (ws) ws.close();
      
      // Create new connection
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/${connectionId}`;
      
      console.log('Connecting WebSocket to:', wsUrl);
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempt = 0; // Reset reconnect counter on successful connection
      };
      
      ws.onmessage = (event) => {
        try {
          console.log('WebSocket message received:', event.data); // Add for debugging
          const data = JSON.parse(event.data);
          if (data.type === 'progress') {
            console.log(`Progress update: ${data.progress}%`);
            setProgress(data.progress);
          } else if (data.type === 'complete') {
            setProgress(100);
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        
        // Only attempt to reconnect if we haven't tried too many times
        if (reconnectAttempt < 5) {
          reconnectAttempt++;
          setTimeout(connectWebSocket, 1000); // Wait 1 second before reconnecting
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };
    
    // Initial connection
    connectWebSocket();
    
    // Cleanup
    return () => {
      if (ws) {
        ws.close();
      }
    };
  }, [connectionId]);

  // Handlers
  const handleInputChange = <K extends keyof FormData>(field: K, value: FormData[K]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const validateRequiredFields = (fields: Array<keyof FormData>): string => {
    const missingFields = fields.filter(field => !formData[field]);
    return missingFields.length > 0 
      ? `Please fill in: ${missingFields.join(', ')}` 
      : '';
  };

  const handlePreviewPrompt = async () => {
    const previewRequired: Array<keyof FormData> = [
      'label', 'label_definition', 'domain', 'language', 
      'specification_format', 'specification_level', 'stakeholder'
    ];
    
    const errorMessage = validateRequiredFields(previewRequired);
    if (errorMessage) {
      setError(errorMessage);
      return;
    }
    
    setPreviewLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/preview-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          feature_values: {
            ...formData,
            count: formData.samples_per_prompt
          }
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to generate prompt preview');
      }
      
      const data = await response.json();
      setCurrentPrompt(data.prompt);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate prompt preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleGenerate = async () => {
    const requiredFields: Array<keyof FormData> = [
      'label', 'label_definition', 'domain', 'language',
      'specification_format', 'specification_level', 'stakeholder'
    ];
    
    const errorMessage = validateRequiredFields(requiredFields);
    if (errorMessage) {
      setError(errorMessage);
      return;
    }
    
    setResults(null);
    setIsLoading(true);
    setStatus("Generating...");
    setError("");
    setProgress(0);
    
    try {
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          feature_values: formData,
          connection_id: connectionId
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Generation failed');
      }
      
      const data = await response.json();
      setResults(data);
      setStatus(`Generation complete! ${data.samples.length} samples generated`);
      
      if (data.fewer_samples_received) {
        setStatus(prev => prev + " (Note: Fewer samples were received than requested due to token limits)");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during generation');
      setStatus("Generation failed");
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownload = () => {
    if (!results?.output_path) return;
    window.open(`/api/download?path=${encodeURIComponent(results.output_path)}`, '_blank');
  };

  // Component to render option buttons to reduce repetition
  const OptionButtons = ({ 
    options, 
    selected, 
    fieldName 
  }: { 
    options: string[], 
    selected: string, 
    fieldName: keyof FormData 
  }) => {
    return (
      <div className="flex flex-wrap gap-2 mt-2">
        {options.map((item) => {
          // Determine the classes based on selection and loading states
          const selectedClass = selected === item 
            ? "bg-[#8A2BE2] border-[#8A2BE2] text-white" 
            : "bg-[#1A1A1A] border-[#2A2A2A] text-zinc-300 hover:bg-[#222222] hover:border-[#8A2BE2]";
          
          const disabledClass = isLoading ? 'opacity-50 pointer-events-none' : '';
          
          return (
            <div
              key={item}
              className={`border rounded-lg px-4 py-2 text-sm cursor-pointer transition-colors ${selectedClass} ${disabledClass}`}
              onClick={isLoading ? undefined : () => handleInputChange(fieldName, item)}
            >
              {item}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-[#0A0A0A] text-white">
      <div className="container mx-auto py-8 max-w-5xl">
        {/* Header */}
        <header className="mb-12 text-center">
          <h1 className="text-6xl font-bold mb-2 text-[#8A2BE2]">Synthline</h1>
          <p className="text-zinc-400 text-xl">Generate High-Quality Synthetic Data for Requirements Engineering</p>
        </header>

        <div className="space-y-6">
          {/* Classification */}
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
                    disabled={isLoading}
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
                    disabled={isLoading}
                  />
                </div>
              </div>
            </Card>
          </section>

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          {/* Requirements Artifact */}
          <section className="space-y-4">
            <h2 className="text-2xl font-medium text-[#8A2BE2]">Requirements Artifact</h2>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label className="text-white text-base font-medium">Specification Format</Label>
                  <OptionButtons 
                    options={SPECIFICATION_FORMATS} 
                    selected={formData.specification_format} 
                    fieldName="specification_format" 
                  />
                </div>

                <div className="h-px bg-[#2A2A2A] my-4"></div>

                <div className="space-y-2">
                  <Label className="text-white text-base font-medium">Specification Level</Label>
                  <OptionButtons 
                    options={SPECIFICATION_LEVELS} 
                    selected={formData.specification_level} 
                    fieldName="specification_level" 
                  />
                </div>

                <div className="h-px bg-[#2A2A2A] my-4"></div>

                <div className="space-y-2">
                  <Label className="text-white text-base font-medium">Stakeholder</Label>
                  <OptionButtons 
                    options={STAKEHOLDERS} 
                    selected={formData.stakeholder} 
                    fieldName="stakeholder" 
                  />
                </div>

                <div className="h-px bg-[#2A2A2A] my-4"></div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label className="text-white text-base font-medium">Domain</Label>
                    <Input 
                      type="text"
                      placeholder="e.g., Banking"
                      className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                      value={formData.domain}
                      onChange={(e) => handleInputChange('domain', e.target.value)}
                      required
                      disabled={isLoading}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label className="text-white text-base font-medium">Language</Label>
                    <Input 
                      type="text"
                      placeholder="e.g., English"
                      className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                      value={formData.language}
                      onChange={(e) => handleInputChange('language', e.target.value)}
                      required
                      disabled={isLoading}
                    />
                  </div>
                </div>
              </div>
            </Card>
          </section>

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          {/* Generator Settings */}
          <section className="space-y-4">
            <h2 className="text-2xl font-medium text-[#8A2BE2]">Generator</h2>
            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
              <Label className="text-white mb-4 block text-base font-medium">LLM Settings</Label>

              <div className="mb-6">
                <Select 
                  value={formData.llm} 
                  onValueChange={(value) => handleInputChange('llm', value)}
                  disabled={isLoading}
                >
                  <SelectTrigger className="bg-[#1A1A1A] border-[#2A2A2A] text-white">
                    <SelectValue placeholder="Select Model" />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1A1A1A] border-[#2A2A2A] text-white">
                    {LLM_OPTIONS.map(option => (
                      <SelectItem key={option.value} value={option.value}>{option.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-white">Temperature</Label>
                    <span className="text-zinc-400 text-sm">{formData.temperature}</span>
                  </div>
                  <Slider 
                    value={[formData.temperature]} 
                    onValueChange={(values) => handleInputChange('temperature', values[0])} 
                    max={2} 
                    step={0.1} 
                    className="py-4"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-zinc-500">Controls randomness (0=deterministic, 2=random)</p>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between">
                    <Label className="text-white">Top P</Label>
                    <span className="text-zinc-400 text-sm">{formData.top_p}</span>
                  </div>
                  <Slider 
                    value={[formData.top_p]} 
                    onValueChange={(values) => handleInputChange('top_p', values[0])} 
                    max={1} 
                    step={0.1} 
                    className="py-4"
                    disabled={isLoading}
                  />
                  <p className="text-xs text-zinc-500">Controls diversity (0=focused, 1=diverse)</p>
                </div>
              </div>
            </Card>

            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
              <Label className="text-white mb-4 block text-base font-medium">Prompt Settings</Label>
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label className="text-white text-base font-medium">Samples Per Prompt</Label>
                  <Input
                    type="number"
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                    value={formData.samples_per_prompt}
                    onChange={(e) => handleInputChange('samples_per_prompt', Number(e.target.value))}
                    required
                    disabled={isLoading}
                  />
                  <p className="text-xs text-zinc-500">Specify the number of samples to generate per prompt</p>
                </div>

                <div className="h-px bg-[#2A2A2A] my-4"></div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <Label className="text-white text-base font-medium">Prompt Preview</Label>
                    <Button
                      variant="outline"
                      size="sm"
                      className="bg-[#1A1A1A] border-[#2A2A2A] text-zinc-300 hover:bg-[#222222] hover:text-white"
                      onClick={handlePreviewPrompt}
                      disabled={previewLoading || isLoading}
                    >
                      {previewLoading ? 'Loading...' : 'Preview'}
                    </Button>
                  </div>
                  <Textarea
                    value={currentPrompt}
                    readOnly
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white h-40 resize-none"
                    placeholder="Click 'Preview' to see the prompt based on your configuration"
                  />
                </div>
              </div>
            </Card>
          </section>

          <div className="h-px bg-[#2A2A2A] my-8"></div>

          {/* Output Settings */}
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
                    disabled={isLoading}
                  />
                </div>

                <div className="space-y-2">
                  <Label className="text-white text-base font-medium">File Format</Label>
                  <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4">
                    <RadioGroup 
                      value={formData.output_format} 
                      onValueChange={(value) => handleInputChange('output_format', value as "CSV" | "JSON")} 
                      className="flex space-x-4"
                      disabled={isLoading}
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

          {/* Error message */}
          {error && (
            <div className="bg-red-900/30 border border-red-500 text-red-200 p-4 rounded-md">
              {error}
            </div>
          )}

          {/* Action Buttons */}
          <div className="space-y-4">
            <Button
              onClick={handleGenerate}
              disabled={isLoading}
              className="w-full py-6 text-lg font-medium bg-[#8A2BE2] hover:bg-opacity-80 text-white transition-all"
            >
              {isLoading ? 'Generating...' : 'Generate Data'}
            </Button>

            {/* Progress bar with percentage */}
            {isLoading && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-[#2A2A2A]" />
              </div>
            )}
          </div>

          {/* Results Area */}
          {results && (
            <Card className="bg-[#121212] border-[#1E1E1E] overflow-hidden shadow-md">
              <div className="border-b border-[#2A2A2A] p-4">
                <h3 className="text-lg font-medium text-[#8A2BE2]">Results</h3>
              </div>

              <div className="p-6">
                <div className="bg-[#1A1A1A] border-[#2A2A2A] p-4 rounded">
                  <div className="flex justify-between items-center">
                    <div>
                      <p className="text-white">{status}</p>
                      <p className="text-zinc-400 text-sm">Output saved to: {results.output_path}</p>
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
          )}
        </div>
      </div>
    </div>
  )
}