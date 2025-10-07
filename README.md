# ARGOS: Ontology-Based Access Control for Agent Systems

This repository contains the implementation of ARGOS (Access Realignment through Guided Ontological Structures), an advanced ontology-based access control system designed for agent systems. This implementation provides SQL query analysis, AST tree manipulation, and ontological reasoning capabilities for enforcing access control policies.

## Overview

ARGOS implements a sophisticated access control mechanism that leverages ontological knowledge to make authorization decisions for database queries. The system parses SQL queries into Abstract Syntax Trees (AST), maps them to ontological instances, and applies reasoning to determine access permissions based on predefined policies.

## Key Features

- **SQL Query Analysis**: Advanced parsing and analysis of SQL queries using SQLGlot
- **AST Tree Manipulation**: Custom tree structures for representing and manipulating query components
- **Ontological Reasoning**: Integration with OWL ontologies for knowledge representation and reasoning
- **Access Control Policies**: Flexible policy framework for defining access control rules
- **Agent System Integration**: Designed specifically for multi-agent environments

## Project Structure

```
OBAC-based-Agent-Systems/
├── main.py                          # Main entry point
├── pyproject.toml                   # Project configuration and dependencies
├── requirements.txt                 # Installation requirements
├── README.md                        # This file
├── uv.lock                         # Lock file for dependencies
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── DBImporter.py               # Database import utilities
│   ├── prune.py                    # Data pruning operations
│   ├── term_evals.py               # Term evaluation functions
│   │
│   ├── data/                       # Data structures and models
│   │   └── ASTTree.py              # AST tree node definitions
│   │
│   └── operators/                  # Core operational modules
│       ├── astObject.py            # SQL AST object operations
│       ├── astTree.py              # AST tree manipulation
│       └── ontologyInstance.py     # Ontology instantiation and reasoning
│
├── data/                           # Datasets and experimental results
│   ├── test_dataset_v4.csv         # Test dataset version 4
│   └── test_dataset_result_v*.csv  # Various versioned result files
│
├── experiments/                    # Jupyter notebooks and experimental code
│
└── ontology_file/                 # Ontology definitions and instances
    ├── ARGOS.rdf                   # Main ARGOS ontology
    ├── ARGOSv3.rdf                 # ARGOS ontology version 3
    ├── finance_instance.properties # Financial domain instances
    ├── financial_instances.rdf     # Financial instances (RDF format)

```

## Installation

### Prerequisites

- Python 3.13 or higher
- pip package manager

### Install Dependencies

1. Clone the repository:
```bash
git clone https://github.com/nipdep/OBAC-based-Agent-Systems.git
cd OBAC-based-Agent-Systems
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Alternatively, if you're using uv:
```bash
uv sync
```

### Dependencies

The project requires the following main dependencies:
- **owlready2** (>=0.48): OWL ontology manipulation and reasoning
- **pandas** (>=2.3.1): Data manipulation and analysis
- **sqlglot** (>=27.7.0): SQL parsing and analysis

## Usage

### Basic Usage

```python
from src.operators.ontologyInstance import OntologyOperator
from src.operators.astObject import SqlglotOperator

# Initialize the ontology operator with your ontology file
onto_op = OntologyOperator("ontology_file/ARGOS.rdf")

# Parse a SQL query
sql_query = "SELECT name, salary FROM employees WHERE department = 'IT'"
sql_op = SqlglotOperator(sql_query)

# Process the query through the access control system
# (Additional implementation details in the source code)
```

### Running Experiments

The `experiments/` directory contains Jupyter notebooks demonstrating various aspects of the system:

1. **Abstract Data Operation Flow**: Main workflow and processing pipeline
2. **AST Analysis**: Analysis of Abstract Syntax Trees
3. **Ontology Testing**: Ontological reasoning and policy evaluation
4. **SQL Query Analysis**: Query parsing and manipulation

## Datasets

The `data/` directory contains various datasets used for testing and evaluation:

- **Sample Datasets**: `sample_data.csv`, `sample_data_mini.csv`
- **Test Datasets**: Various versions (`test_dataset_v2.csv` through `test_dataset_v4.csv`)
- **Results**: Experimental results from different runs and configurations
- **Versioned Results**: Multiple result files tracking different experimental configurations

## Architecture

### Core Components

1. **SqlglotOperator**: Handles SQL parsing and AST manipulation
2. **ASTTreeOperator**: Manages custom tree structures for query representation
3. **OntologyOperator**: Manages ontological reasoning and instance creation
4. **TreeNode**: Represents nodes in the AST tree structure

### Workflow

1. SQL queries are parsed into AST representations
2. AST nodes are mapped to ontological instances
3. Access control policies are evaluated using ontological reasoning
4. Authorization decisions are made based on policy evaluation results

## Research Citation

If you use this implementation in your research, please cite:

```bibtex

```

## Contributing

Contributions are welcome! Please ensure that any contributions maintain the existing code structure and include appropriate tests.

## License

This project is part of ongoing research in ontology-based access control systems. Please contact the authors for licensing information.

## Contact

For questions regarding this implementation or research collaboration, please contact the project maintainers through the GitHub repository.
