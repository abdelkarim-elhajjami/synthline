# Synthline

<div align="center">
  <h3>Generate High-Quality Synthetic Data for Requirements Engineering</h3>
</div>

## Overview

Synthline generates high-quality synthetic data for training and evaluating AI models in Requirements Engineering. It uses large language models (GPT-4o, DeepSeek) to produce realistic requirements with configurable properties.

### Key Features

- **LLM-Powered Generation**: Using GPT-4o and DeepSeek models
- **Highly Configurable**: Control all aspects of generated requirements
- **Multiple Output Formats**: Export as JSON or CSV
- **Web Interface**: Intuitive UI for configuration

## Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- API Keys:
  - [OpenAI API Key](https://platform.openai.com/) (for GPT-4o)
  - [DeepSeek API Key](https://www.deepseek.com/) (for DeepSeek models)

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

- **LLM**: Select the language model (GPT-4o, DeepSeek)
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