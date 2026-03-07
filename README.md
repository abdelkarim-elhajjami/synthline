# Synthline
Synthline is a Product Line approach for generating synthetic Requirements Engineering (RE) data using Large Language Models (LLMs). This repository contains the implementation code and datasets from our research paper "Synthline: A Product Line Approach for Synthetic Requirements Engineering Data Generation using Large Language Models".

Synthline addresses the data scarcity challenge in Requirements Engineering by providing a systematic approach to generate synthetic data. Our framework leverages LLMs and a Feature Model to enable controlled generation of synthetic data for various RE use cases.

### Key Features
- Product Line approach using Feature Models for systematic data generation
- Support for multiple LLMs (GPT-4o and DeepSeek-V3)
- Configurable generation parameters
- Support for various requirement types, specification formats, and domains
- Evaluation tools for data diversity and classifiers performance

## Installation
1. Clone the repository.
2. Install required dependencies (requirements.txt). Note: Python 3.10 or higher is required.

## Project Structure
```
synthline/
├── src/                         # Source code
│   ├── fm.py                    # Feature Model implementation 
│   ├── generator.py             # Data generation logic
│   ├── gui.py                   # GUI interface
│   ├── llm_client.py            # LLM API client
│   ├── output.py                # Output handling
│   ├── promptline.py            # Prompt generation
│   └── run.py                   # Main entry point
│
├── train_classifiers/           # Classifiers training and evaluation  
│   ├── config.py                # Training configuration
│   ├── data_processor.py        # Data processing utilities
│   ├── evaluate_classifiers.py  # Classifiers evaluation
│   ├── evaluate_diversity.py    # Diversity evaluation
│   └── train.py                 # Classifiers training
│
├── output/                      # Generated datasets
└── requirements.txt             # Project dependencies
```

## Usage
1. Configure your API keys in `src/run.py`.
2. Run the Synthline GUI.