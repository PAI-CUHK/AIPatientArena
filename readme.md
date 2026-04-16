# AIPatient Arena

## A Doctor-Patient Interactive Framework for Evaluating LLM-based Doctor Performance in Clinical Consultations

### Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Contributing](#contributing)


## Project Overview

AIPatient Arena is a comprehensive framework designed to evaluate the performance of LLM-based doctors in clinical consultations through interactive doctor-patient dialogues. This framework provides a standardized environment for testing and comparing different LLM models in a clinical setting, focusing on their ability to ask appropriate questions, gather relevant information, and make accurate diagnoses.

Key features:
- Interactive doctor-patient dialogue simulation
- Multiple evaluation dimensions (questioning skills, information coverage, robustness, ethics, etc.)
- Standardized evaluation metrics
- Support for multiple LLM models
- Neo4j database integration for patient data management


## Installation

### Prerequisites

- Python 3.8+
- Neo4j Database
- OpenAI API key (for GPT models)
- Required Python packages

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PAI-CUHK/AIPatientArena.git
   cd AIPatientArena
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Neo4j Database**:
   - Create a Neo4j database instance
   - Update the connection details in `secrets.txt`:
     ```
     uri,neo4j+s://your-neo4j-uri
     user,neo4j
     password,your-neo4j-password
     ```

4. **Configure API Keys**:
   - Add your OpenAI API key to `secrets.txt`:
     ```
     open_ai_key,sk-your-openai-key
     base_url,https://api.openai.com/v1
     ```

## Quick Start

To run the benchmark with a sample doctor module:

```bash
python src/main.py --doctor_module AIdoctor --doctor_class Doctor  --output_filename data/dialogue/gpt5.jsonl --max_questions 15 --doctor_model_name gpt5
```

To evaluate the performance of a doctor model, you can use the following command:

```bash
python src/evaluate.py --input gpt5.jsonl --output gpt5.jsonl
```

## Contributing

We welcome contributions to the AIPatient Arena project. To contribute:

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

Please ensure your code follows the project's coding standards and includes appropriate tests.
