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
  prompt_approach: string;
  pace_iterations: number;
  pace_actors: number;
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

// Constants
const SPECIFICATION_FORMATS = ["NL", "Constrained NL", "Use Case", "User Story"];
const SPECIFICATION_LEVELS = ["High", "Detailed"];
const STAKEHOLDERS = ["End Users", "Business Managers", "Developers", "Regulatory Bodies"];
const LLM_OPTIONS = [
  { value: "gpt-4o", label: "gpt-4o" },
  { value: "deepseek-chat", label: "deepseek-chat" }
];
const REQUIRED_FIELDS: (keyof FormData)[] = [
  'label', 'label_definition', 'domain', 'language',
  'specification_format', 'specification_level', 'stakeholder'
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
    prompt_approach: "Default",
    pace_iterations: 4,
    pace_actors: 2
  };

  // State
  const [formData, setFormData] = useState<FormData>(initialFormState);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("");
  const [currentPrompt, setCurrentPrompt] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isOptimizingPrompt, setIsOptimizingPrompt] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Results | null>(null);
  const [connectionId] = useState(() => uuidv4());
  const [optimizationSuccess, setOptimizationSuccess] = useState<string | null>(null);
  const [isPromptOptimized, setIsPromptOptimized] = useState(false);
  
  // WebSocket handling
  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.hostname}:8000/ws/${connectionId}`;
    const ws = new WebSocket(wsUrl);
    
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        switch (data.type) {
          case 'progress':
            setProgress(data.progress);
            break;
          
          case 'prompt_update':
            setCurrentPrompt(data.prompt);
            break;
          
          case 'optimize_complete':
            setIsOptimizingPrompt(false);
            setProgress(100);
            
            const optimizedPrompt = data.optimized_prompt;
            
            setCurrentPrompt(optimizedPrompt);
            setIsPromptOptimized(true);
            
            setOptimizationSuccess(`Prompt optimization complete!`);
            setTimeout(() => setOptimizationSuccess(null), 10000);
            break;
          
          case 'error':
            setError(data.message);
            setIsOptimizingPrompt(false);
            break;
          
          case 'complete':
            setProgress(100);
            break;
        }
      } catch (error) {
        console.error("WebSocket message parsing error:", error, event.data);
      }
    };
    
    return () => ws.close();
  }, [connectionId]);

  // Automatic prompt preview
  useEffect(() => {
    const missingFields = REQUIRED_FIELDS.filter(field => !formData[field]);
    if (missingFields.length === 0) {
      generatePromptPreview();
    } else {
      setCurrentPrompt("");
    }
  }, [
    formData.label, 
    formData.label_definition, 
    formData.domain, 
    formData.language,
    formData.specification_format, 
    formData.specification_level, 
    formData.stakeholder,
    formData.samples_per_prompt
  ]);

  const generatePromptPreview = async () => {
    try {
      const response = await fetch('/api/preview-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: formData })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Preview generation failed:', errorData);
        return;
      }
      
      const data = await response.json();
      setCurrentPrompt(data.prompt);
    } catch (err) {
      console.error('Preview generation failed:', err);
    }
  };

  // Handlers
  const validateForm = (): string => {
    const missingFields = REQUIRED_FIELDS.filter(field => !formData[field]);
    return missingFields.length > 0 
      ? `Please fill in: ${missingFields.join(', ')}` 
      : '';
  };
  
  const handleInputChange = <K extends keyof FormData>(field: K, value: FormData[K]) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleOptimizePrompt = async () => {
    const errorMessage = validateForm();
    if (errorMessage) {
      setError(errorMessage);
      return;
    }
    
    setIsOptimizingPrompt(true);
    setResults(null);
    setError("");
    setProgress(0);
    setStatus("Optimizing...");
    
    try {
      const response = await fetch('/api/optimize-prompt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: formData,
          connection_id: connectionId
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prompt optimization failed');
      }
      
      // The actual results will come via WebSocket,
      // so we just acknowledge that the optimization is running
      await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred during prompt optimization');
      setIsOptimizingPrompt(false);
    }
  };

  const handleGenerate = async () => {
    const errorMessage = validateForm();
    if (errorMessage) {
      setError(errorMessage);
      return;
    }
    
    setResults(null);
    setIsGenerating(true);
    setStatus("Generating...");
    setError("");
    setProgress(0);
    
    try {
      // Include optimized prompt if available
      const requestData = {
        features: {
          ...formData,
          optimized_prompt: formData.prompt_approach === "PACE" && isPromptOptimized 
            ? currentPrompt 
            : undefined
        },
        connection_id: connectionId
      };
      
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
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
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!results?.output_path) return;
    window.open(`/api/download?path=${encodeURIComponent(results.output_path)}`, '_blank');
  };

  // Component to render option buttons
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
          
          const disabledClass = isGenerating || isOptimizingPrompt ? 'opacity-50 pointer-events-none' : '';
          
          return (
            <div
              key={item}
              className={`border rounded-lg px-4 py-2 text-sm cursor-pointer transition-colors ${selectedClass} ${disabledClass}`}
              onClick={() => handleInputChange(fieldName, item)}
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
        <header className="mb-6 text-center">
          <h1 className="text-6xl font-bold mb-2 text-[#8A2BE2]">Synthline</h1>
          <p className="text-zinc-400 text-xl">Generate High-Quality Synthetic Data for Requirements Engineering</p>
        </header>

        {/* Error message */}
        {error && (
          <div className="bg-red-900/30 border border-red-500 text-red-200 p-4 rounded-md mb-6 sticky top-2 z-10">
            {error}
          </div>
        )}

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
                    disabled={isGenerating || isOptimizingPrompt}
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
                    disabled={isGenerating || isOptimizingPrompt}
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
                      disabled={isGenerating || isOptimizingPrompt}
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
                      disabled={isGenerating || isOptimizingPrompt}
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
                  disabled={isGenerating || isOptimizingPrompt}
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
                    disabled={isGenerating || isOptimizingPrompt}
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
                    disabled={isGenerating || isOptimizingPrompt}
                  />
                  <p className="text-xs text-zinc-500">Controls diversity (0=focused, 1=diverse)</p>
                </div>
              </div>
            </Card>

            <Card className="bg-[#121212] border-[#1E1E1E] p-6 shadow-md">
              <Label className="text-white mb-4 block text-base font-medium">Prompt Settings</Label>
              <div className="space-y-6">
                <div className="space-y-2">
                  <Label className="text-white text-sm font-medium">Samples Per Prompt</Label>
                  <Input
                    type="number"
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white"
                    value={formData.samples_per_prompt}
                    onChange={(e) => handleInputChange('samples_per_prompt', Math.max(1, Number(e.target.value)))}
                    min={1}
                    required
                    disabled={isGenerating || isOptimizingPrompt}
                  />
                  <p className="text-xs text-zinc-500">Specify the number of samples to generate per prompt</p>
                </div>

                <div className="h-px bg-[#2A2A2A] my-4"></div>

                <div className="space-y-2">
                  <Label className="text-white text-sm font-medium">Prompt Approach</Label>
                  <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4">
                    <RadioGroup 
                      value={formData.prompt_approach}
                      onValueChange={(value) => handleInputChange('prompt_approach', value)}
                      className="space-y-2"
                      disabled={isGenerating || isOptimizingPrompt}
                    >
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="Default" id="default" className="text-[#8A2BE2]" />
                        <Label htmlFor="default" className="text-white">Default</Label>
                      </div>
                      <div className="flex items-center space-x-2">
                        <RadioGroupItem value="PACE" id="pace" className="text-[#8A2BE2]" />
                        <Label htmlFor="pace" className="text-white">PACE Optimization</Label>
                      </div>
                    </RadioGroup>
                  </div>
                </div>

                {formData.prompt_approach === "PACE" && (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label className="text-white">Iterations</Label>
                        <span className="text-zinc-400 text-sm">{formData.pace_iterations}</span>
                      </div>
                      <Slider 
                        value={[formData.pace_iterations]} 
                        onValueChange={(values) => handleInputChange('pace_iterations', values[0])} 
                        min={1}
                        max={10} 
                        step={1} 
                        className="py-4"
                        disabled={isGenerating || isOptimizingPrompt}
                      />
                      <p className="text-xs text-zinc-500">More iterations may yield better prompts but take longer</p>
                    </div>

                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <Label className="text-white">Actor-Critic Pairs</Label>
                        <span className="text-zinc-400 text-sm">{formData.pace_actors}</span>
                      </div>
                      <Slider 
                        value={[formData.pace_actors]} 
                        onValueChange={(values) => handleInputChange('pace_actors', values[0])} 
                        min={1}
                        max={10} 
                        step={1} 
                        className="py-4"
                        disabled={isGenerating || isOptimizingPrompt}
                      />
                      <p className="text-xs text-zinc-500">More pairs provide diverse feedback but increase costs</p>
                    </div>

                    <Button 
                      onClick={handleOptimizePrompt}
                      disabled={isOptimizingPrompt || isGenerating}
                      className="w-full bg-[#8A2BE2] hover:bg-opacity-80 text-white"
                    >
                      {isOptimizingPrompt ? "Optimizing..." : "Optimize Prompt"}
                    </Button>

                    {/* Progress bar for optimization */}
                    {isOptimizingPrompt && (
                      <div className="space-y-2">
                        <div className="flex justify-end">
                          <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
                        </div>
                        <Progress value={progress} className="h-2 bg-[#2A2A2A]" />
                      </div>
                    )}
                    
                    {/* Optimization success notification */}
                    {optimizationSuccess && (
                      <div className="bg-green-900/30 border border-green-500 text-green-200 p-3 rounded-md">
                        {optimizationSuccess}
                      </div>
                    )}
                  </div>
                )}

                <div>
                  <Label className="text-white text-sm font-medium">Prompt Preview</Label>
                  <Textarea
                    value={currentPrompt}
                    className="bg-[#1A1A1A] border-[#2A2A2A] text-white h-40 resize-none"
                    placeholder="Prompt preview once all required fields are filled"
                    readOnly
                    disabled={isGenerating || isOptimizingPrompt}
                  />
                  <p className="text-xs text-zinc-500 mt-2 italic">
                  {formData.prompt_approach === "PACE" 
                      ? "This is the optimized prompt that will be used for generation"
                      : "This prompt will be used for generation"}
                  </p>
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

          {/* Action Buttons */}
          <div className="space-y-4">
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || isOptimizingPrompt}
              className="w-full py-6 text-lg font-medium bg-[#8A2BE2] hover:bg-opacity-80 text-white transition-all"
            >
              {isGenerating ? 'Generating...' : 'Generate Data'}
            </Button>

            {/* Progress bar with percentage */}
            {isGenerating && (
              <div className="space-y-2">
                <div className="flex justify-end">
                  <span className="text-sm font-medium text-white">{Math.round(progress)}%</span>
                </div>
                <Progress value={progress} className="h-2 bg-[#2A2A2A]" />
              </div>
            )}
          </div>

          {/* Results */}
          {results && (
            <Card className="bg-[#121212] border-[#1E1E1E] overflow-hidden shadow-md">
              <div className="border-b border-[#2A2A2A] p-4">
                <h3 className="text-lg font-medium text-[#8A2BE2]">Results</h3>
              </div>

              <div className="p-6">
                <div className="bg-[#1A1A1A] border-[#2A2A2A] p-4 rounded">
                  <div className="flex justify-between items-center mb-4">
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