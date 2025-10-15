# OpenModelsHub - Django Demo

## Project Structure

```
OpenModelsHub/ProofOfConcept/src/
├── openmodelshub_project/   # Django project settings
│   ├── settings.py          # Main configuration
│   ├── urls.py              # URL routing
│   ├── wsgi.py              # WSGI configuration
│   └── asgi.py              # ASGI configuration
├── frontend/                # Web interface app
│   ├── views.py             # View functions
│   ├── urls.py              # URL patterns
│   └── templates/           # HTML templates
├── research/                # Research context models
│   ├── models.py            # Organization, Researcher
│   └── admin.py             # Admin interface
├── assets/                  # ML assets models
│   ├── models.py            # Model, Dataset, Experiment
│   └── admin.py             # Admin interface
├── resources/               # Computational resources
│   ├── models.py            # Resources, Utilization
│   └── admin.py             # Admin interface
├── manage.py                # Django management script
├── test_models.py           # Comprehensive model tests
└── db.sqlite3               # SQLite database (created after migration)
```

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

## Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to the project directory
cd OpenModelsHub/Demo/src

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Database Setup

The application uses SQLite by default.

```bash
# Create migrations (if not already created)
python manage.py makemigrations

# Apply migrations to create database tables
python manage.py migrate
```

### 4. Create Test Data (Optional)

Run the test script to populate the database with sample data:

```bash
python test_models.py
```

This will create:
- 1 Organization (AI Research Institute)
- 1 Researcher (Alice Johnson)
- 4 Datasets (including version chain)
- 1 Model (Climate Detection ResNet Model)
- 1 Experiment (Climate Detection Training Run)
- 3 Computational Resources (Local, Cloud, HPC)
- Resource utilization with environmental & cost data
- Hyperparameters, metrics, and checkpoints

### 5. Create Admin User (Optional)

To access the Django admin interface:

```bash
python manage.py createsuperuser
```

Follow the prompts to create your admin account.

## Running the Server

### Start Development Server

```bash
python manage.py runserver
```

The server will start at **http://localhost:8000/**

### Available URLs

#### Main Pages
- **Home**: http://localhost:8000/
- **Models**: http://localhost:8000/models/
- **Datasets**: http://localhost:8000/datasets/
- **Experiments**: http://localhost:8000/experiments/
- **Researchers**: http://localhost:8000/researchers/
- **Organizations**: http://localhost:8000/organizations/


#### Admin & Search
- **Django Admin**: http://localhost:8000/admin/
- **Search**: http://localhost:8000/search/