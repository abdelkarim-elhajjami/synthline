# Synthline

Synthline is a tool for generating high-quality synthetic data for requirements engineering. It leverages large language models (LLMs) to create diverse, customizable requirement samples according to specified attributes.

## Overview

Synthline generates high-quality synthetic data for training and evaluating AI models in Requirements Engineering. It uses large language models (gpt-4.1-nano-2025-04-14, DeepSeek) to produce realistic requirements with configurable properties.

### Key Features

- **LLM-Powered Generation**: Using gpt-4.1-nano-2025-04-14 and DeepSeek models
- **Highly Configurable**: Control all aspects of generated requirements
- **Multiple Output Formats**: Export as JSON or CSV
- **Web Interface**: Intuitive UI for configuration

## New Feature: PACE Prompt Optimization

Synthline now includes PACE (Prompt Actor-Critic Editing) for prompt optimization (https://aclanthology.org/2024.findings-acl.436/). This technique improves the quality and relevance of generated requirements by:

- Using multiple "actors" to generate candidate outputs
- Employing "critics" to evaluate these outputs and provide feedback
- Iteratively refining prompts based on collected feedback
- Measuring diversity to select the best performing prompt

The PACE approach helps create more diverse, accurate, and domain-specific requirements with minimal manual intervention.

To use PACE:
1. Select "PACE Optimization" in the prompt approach settings
2. Configure the number of iterations and actor-critic pairs
3. Click "Optimize Prompt" before generating data

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- API Keys:
  - [OpenAI API Key](https://platform.openai.com/) (for gpt-4.1-nano-2025-04-14)
  - [DeepSeek API Key](https://www.deepseek.com/) (for DeepSeek models)
- Local LLM (Optional):
  - [Ollama](https://ollama.com/) installed and running
  - Pull a model: `ollama pull ministral-3:14b`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/abdelkarim-elhajjami/Synthline.git
   cd synthline
   ```

2. Create environment files:

   For the engine:
   ```bash
   # engine/.env
   OPENAI_API_KEY=your_openai_api_key
   DEEPSEEK_API_KEY=your_deepseek_api_key
   
   # For Local LLM (Ollama)
   OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
   ```

3. Start the application using the provided script:
   ```bash
   ./deploy.sh
   ```

4. Access the web interface at [http://localhost:3000](http://localhost:3000)

## Configuration Options

Synthline offers various configuration options for generating requirements:

### Classification

- **Label**: The type/category of requirement
- **Label Definition**: Description of what the label means

### Requirements Artifact

- **Specification Format**: NL, Constrained NL, Use Case, User Story
- **Specification Level**: High, Detailed
- **Stakeholder**: End Users, Business Managers, Developers, Regulatory Bodies
- **Domain**: Application domain (e.g., Healthcare, Finance)
- **Language**: Natural language (e.g., English, Spanish)

### Generator Settings

- **LLM**: Select the language model (gpt-4.1-nano-2025-04-14, DeepSeek)
- **Temperature**: Controls randomness (0-2)
- **Top P**: Controls diversity (0-1)
- **Samples Per Prompt**: Number of samples in each LLM request

### Output Settings

- **Total Samples**: Total number of samples to generate
- **File Format**: JSON or CSV output format

## Usage Guide

1. **Configuration**: Set up your generation parameters in the web interface
2. **Preview**: Preview the prompt that will be sent to the LLM
3. **Generate**: Start the generation process
4. **Download**: Get your generated samples as JSON or CSV 