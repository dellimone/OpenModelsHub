# OpenModelsHub

A FAIR-compliant metadata management system for machine learning research assets.

## Overview

OpenModelsHub helps researchers document, track, and share ML models, datasets, and experiments with complete provenance and reproducibility metadata following FAIR principles.

## Quick Start

```bash
# Clone and navigate
git clone git@github.com:dellimone/OpenModelsHub.git
cd OpenModelsHub/ProofOfConcept/src

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

# Initialize database
python manage.py migrate

# Run server
python manage.py runserver
```

Visit http://localhost:8000/

## Documentation

- [Data Models](Documentation/data-model/) - Entity specifications
- [Data Management Plan](Documentation/data-plan/data-plan.md) - Complete lifecycle documentation
- [FAIR Compliance](Documentation/fair-compliance-assessment.md) - FAIR principles implementation
- [Metadata Schemas](Documentation/metadata/) - XML schemas and standards

## Project Structure

```bash
OpenModelsHub/
├── Documentation/      # Models, schemas, and standards
├── ProofOfConcept/    # Django implementation
└── OpenModelsHub.pdf  # Project overview
```
