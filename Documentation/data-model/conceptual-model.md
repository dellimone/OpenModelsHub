# Conceptual Data Model

### Overview

OpenModelsHub is a comprehensive platform designed to manage machine learning assets, with a focus on computational resource tracking, environmental impact, cost monitoring, and reproducibility.

This document outlines the core business concepts, entities, and relationships that form the foundation of the system.

## Business Context

**Domain**: Machine Learning Research and Asset Management

**Objective**: Enable researchers to efficiently manage, discover, replicate, and reuse machine learning assets, while also tracking computational costs and environmental impact.

**Key Stakeholders**:
- **ML Researchers**: Create and use models, datasets, experiments
- **Data Scientists**: Apply models in downstream tasks
- **Research Institutions**: Manage computational resources and projects

**Guiding Principles**:
- **FAIR Data**: Findable, Accessible, Interoperable, Reusable assets
- **Model Reproducibility**: Ensure experiments and results can be reliably reproduced
- **Resource Awareness**: Mandatory tracking of computational costs and environmental impact
- **Standards Compliance**: Adherence to international metadata standards

## Data Model Overview

![OpenModelsHub Logical Data Model](../Assets/logical_model/logical-model.svg)

## Entities

### ML Assets
The central assets of the machine learning workflow:

- **MLAsset (Base)**: Common properties shared by all ML assets (ID, metadata, versioning)
- **Model**: Trained machine learning models with performance metrics and deployment info
- **Dataset**: Collections of data used for training, validation, or testing
- **Experiment**: Complete training runs that produce models from datasets

### Experiment Tracking
Entities that track the detailed execution and configuration of experiments:

- **Hyperparameter**: Individual experiment configuration parameters (learning rate, batch size, etc.)
- **ExperimentDataset**: How an experiment uses a dataset, including split definition and preprocessing
- **ExperimentMetric**: Time-series metrics logged during training (loss, accuracy per epoch)
- **Checkpoint**: Saved model states during training for resume and rollback

### Resource Management 
Tracks computational resources and their impact:

- **ComputationalResource**: Base class for computing infrastructure
  - **LocalResource**: On-premises hardware (workstations, local servers)
  - **CloudResource**: Cloud provider instances (AWS, Azure, GCP)
  - **HPCResource**: High-performance computing cluster resources
- **ResourceUtilization**: Actual usage metrics with integrated cost and environmental tracking

### Research Context
Connects assets to their creators and institutional context:

- **Researcher**: Individual contributors with expertise and affiliations
- **Organization**: Institutions that provide resources and governance

## Attributes

### MLAsset (Base) - Shared foundation for all ML assets

- **Identity**: Unique system ID (UUID), Persistent identifier (DOI, etc.), Name, description, version
- **Ownership**: Creator (Researcher), Organization, License and access rights
- **Temporal**: Creation and update timestamps
- **Versioning**: Parent version (simple parent-child relationship), Version notes
- **Integrity**: Checksum with algorithm specification

### Model - Trained ML models

- **File Information**: File path, size, format (PyTorch, TensorFlow, ONNX, etc.)
- **Architecture**: Framework and version, Model type (classification, regression, generation, etc.), Architecture description, Input/output schemas
- **Performance**: Inference time, Model size
- **Training**: Produced by which Experiment

### Dataset - Training/validation/test data

- **File Structure**: File paths (can be multiple files), Total size, Format (CSV, JSON, Parquet, etc.), Schema definition
- **Statistics**: Number of records and features, Target column, Data types per column, Missing value counts
- **Column Classification**: Categorical columns, Numerical columns
- **Privacy & Ethics**: Privacy level (public/confidential/restricted), Ethical considerations
- **Collection**: Collection method, Sampling strategy
- **Relationships**: Used by Experiments (via ExperimentDataset)

### Experiment - ML training/evaluation runs

- **Classification**: Experiment type (training, validation, testing, etc.), Status (pending, running, completed, failed)
- **Timing**: Start time, end time, duration
- **Configuration**: Config parameters (JSON), Random seed for reproducibility, Reproducibility hash
- **Code & Environment**: Code repository URL, Commit hash, Environment specification
- **Relationships**: Uses base_model (optional, for fine-tuning), Produces Model(s), Uses Dataset(s) via ExperimentDataset, Has Hyperparameter(s), Logs ExperimentMetric(s), Saves Checkpoint(s), Uses ComputationalResource(s) via ResourceUtilization

### Hyperparameter - Queryable experiment parameters

- **Parameter Identity**: Parameter name (e.g., "learning_rate", "batch_size"), Parameter value (stored as string for flexibility), Parameter type (float, int, string, boolean, list, dict)
- **Metadata**: Display order (for UI presentation), Is tunable (if part of hyperparameter search)

### ExperimentDataset - Dataset usage with split definition

- **Association**: Which Experiment, Which Dataset
- **Role**: How dataset is used (training/validation/testing)
- **Split Definition**: Split percentage, Number of records, Indices file path (which records), Random seed (reproducibility)
- **Preprocessing**: Transformations applied, Data augmentation enabled

### ExperimentMetric - Time-series training metrics

- **Metric Identity**: Metric name (e.g., "train_loss", "val_accuracy", "f1_score"), Step (epoch or training step number), Timestamp, Value
- **Classification**: Metric type (loss, accuracy, precision, recall, custom)

### Checkpoint - Training snapshots

- **Identification**: Checkpoint name (e.g., "epoch_10", "best_val_loss"), Step (epoch or training step), Saved timestamp
- **File Information**: File path, size, Checksum with algorithm
- **Metadata**: Metrics snapshot (metrics at this checkpoint), Is best (best checkpoint based on metric), Is final (training complete), Notes
- **Relationships**: Belongs to Experiment, Can become published Model

### ComputationalResource (Base) - Computing infrastructure

- **Identity**: Name, location, availability status
- **Resource Type**: Type: LOCAL, CLOUD, HPC
- **Common Specifications**: CPU cores, Memory (GB), GPU count and model
- **Cost**: Pricing model, Cost per hour, Currency
- **Environmental**: Energy efficiency, Carbon intensity (kg CO2/kWh)

### LocalResource - On-premises hardware

- **Network**: Hostname, MAC addresses, Local IP address
- **Physical**: Power consumption (watts), Cooling type, Chassis type, Rack location, Asset tag

### CloudResource - Cloud provider instances

- **Provider Information**: Cloud provider (AWS, Azure, GCP, etc.), Provider region, Availability zone, Account ID
- **Instance Details**: Instance type, Instance family, Virtualization type, Tenancy (shared/dedicated)
- **Pricing**: Pricing model (on-demand, reserved, spot), Spot price
- **Auto-scaling**: Auto-scaling enabled, Min/max instances

### HPCResource - HPC cluster resources

- **Cluster**: Cluster name, Total nodes, Node configuration
- **Scheduler**: Scheduler type (Slurm, PBS, LSF, SGE), Scheduler version, Partition names, Queue limits
- **Resource Policy**: Allocation policy (exclusive/shared), Max job duration
- **Interconnect**: Interconnect type (InfiniBand, Ethernet), Parallel filesystem (Lustre, GPFS, BeeGFS)

### ResourceUtilization - Resource tracking

- **Association**: Which Experiment, Which ComputationalResource
- **Timing**: Start time, end time, duration
- **Usage Metrics**: CPU utilization (%), GPU utilization (%), Memory usage (GB), Storage usage (GB), Network I/O (GB)
- **Cost Tracking** (integrated), Compute cost, Storage cost, Network cost, Total cost, Currency
- **Environmental Impact** (integrated), Carbon footprint (kg CO2 equivalent), Carbon footprint uncertainty range (lower/upper bounds), Energy consumption (kWh), Water usage (liters, optional), Carbon intensity location, Carbon intensity value (kg CO2/kWh)
- **Analytics**: Peak usage metrics (JSON), Average usage metrics (JSON)

### Researcher - Individual contributors

- **Identity**: First name, last name, Email, ORCID ID (persistent researcher identifier)
- **Expertise**: Expertise areas, Research interests
- **Affiliations**: Affiliated with Organizations (M:M)

### Organization - Research institutions

- **Identity**: Name, Type (university, research institute, corporation, government, non-profit), Location, Website
- **Governance**: Policies
- **Resources**: Provides ComputationalResources (M:M)

## Relationships

### Core Workflow Relationships

**Training Flow**:
- Experiment uses Dataset(s) via ExperimentDataset (1..M:0..M)
  - Captures role (training/validation/testing)
  - Split definition (percentage, records, seed)
  - Preprocessing applied

- Experiment uses base_model  (0..1:0..M) 
  - For fine-tuning scenarios
  - Clear directionality: experiment uses existing model

- Experiment produces Model(s) (0..M:1)
  - Clear directionality: experiment generates new models
  - One experiment can produce multiple models (checkpoints becoming models)

**Experiment Tracking**:
- Experiment has Hyperparameter(s) (1..M:1)
  - Each parameter is a separate queryable record

- Experiment logs ExperimentMetric(s) (1..M:1)
  - Time-series metrics with step/timestamp

- Experiment saves Checkpoint(s) (0..M:1)
  - Training snapshots for resume and rollback

**Versioning**:
- MLAsset has parent_version (0:1)
  - Simple parent-child versioning (v1 → v2 → v3)
  - Can query version trees recursively
  - Version notes document changes

**Attribution**:
- Researcher creates MLAsset(s) (0..M:1..M)
  - Each researcher can create zero or many assets
  - Each asset can be created by more researchers

### Resource Relationships

**Allocation**:
- Organization provides ComputationalResource(s) (M:M via ResourceAllocation)
  - Quota hours
  - Cost allocation percentage
  - Access priority
  - Start/end dates

**Usage**:
- Experiment uses ComputationalResource(s) (M:M via ResourceUtilization)
  - Tracks actual usage, cost, and environmental impact
  - One experiment can use multiple resources
  - One resource used by multiple experiments

### Research Context Relationships

**Affiliation**:
- Researcher affiliated with Organization(s) (M:M via OrganizationAffiliation)
  - Role, start/end dates
  - Primary affiliation flag

## Rules and Constraints

### Asset Management Rules

**Unique Identity**
- Every asset must have a globally unique, persistent identifier
- System ID (UUID) + persistent ID (DOI, etc.)

**Complete Provenance**
- All assets must have traceable lineage to source data and creators
- Built-in provenance via created_by, created_at, parent_version
- Training lineage via Experiment → Dataset relationships

**Metadata Standards**
- Assets must comply with relevant international metadata standards
- Metadata validation ensures completeness and consistency

**Version Control**
- Asset versions use simple parent-child relationships via parent_version FK
- Version notes document changes
- Recursive queries enable version tree traversal

**Integrity**
- All files must have checksums with algorithm specification
- Checksum validation ensures file integrity

### Experiment Tracking Rules

**Metric Logging**
- All training metrics must include step number and timestamp
- Unique constraint: (experiment, metric_name, step)

**Checkpoint Integrity**
- All checkpoints must have checksums with algorithm
- Checkpoints linked to parent experiment
- Can designate "best" checkpoint based on metrics

**Hyperparameter Completeness**
- Core hyperparameters should be tracked as separate Hyperparameter records
- Enables efficient querying by parameter values
- Unique constraint: (experiment, parameter_name)

**Configuration Tracking**
- Random seed required for reproducibility
- Environment specification required for replication

**Split Reproducibility**
- ExperimentDataset must include random seed for reproducibility
- Indices file documents exact records in split
- Split definition is experiment-specific

**Experiment-Model Clarity**
- Experiments can optionally use a base_model (for fine-tuning)
- Experiments produce Model(s)

**Reproducibility**
- Experiments must include complete environment specifications
- Code repository and commit hash required
- Random seeds for deterministic execution

### Resource Tracking Rules

**Mandatory Monitoring**
- All computational resource utilization must be tracked
- Usage metrics required: CPU, GPU, memory, storage

**Unified Tracking**
- Cost and environmental impact tracked together in ResourceUtilization

**Environmental Impact**
- Carbon footprint required with uncertainty bounds
- Energy consumption required
- Geographic location for carbon intensity

**Type Specificity**
- Resources must specify type: Local, Cloud, or HPC
- Type-specific attributes in specialized subclasses

**Resource Allocation**
- Organizations can share resources via ResourceAllocation
- Quota and cost allocation tracked
- Supports multi-tenant and shared HPC scenarios


### Research Integrity Rules

**Attribution**
- All contributions must be properly attributed to researchers
- ORCID IDs recommended for persistent identification

**FAIR Principles**
- Assets must be Findable, Accessible, Interoperable, and Reusable
- Metadata standards compliance enforced

**Institutional Governance**
- Organizations can define policies
- Resource allocations enforce quotas and priorities

---

*Next: [logical-model.md](logical-model.md) for detailed entity specifications*
