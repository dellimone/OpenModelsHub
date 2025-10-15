# Data Management Plan

## Overview

Unlike traditional scientific datasets (e.g., observational or survey data), OpenModelsHub focuses on **meta-research data**: structured records that document the *inputs, processes, and outputs* of machine learning experiments. The goal is to ensure **FAIRness**, **traceability**, and **reproducibility** across the entire research lifecycle.

## Data Life Cycle – UK Data Archive

### Phase 1 — Create

**Objective:**
Capture and register machine learning assets as digital research objects with complete, validated metadata and provenance.

**Activities:**

- Capture experiment configurations, metrics, and resource logs from training systems
- Assign **persistent identifiers** (DOIs for published assets, UUIDs for system tracking, ORCIDs for researchers)
- Validate metadata completeness and schema conformance at creation time
- Record creator and institutional ownership details with ORCID linkage
- Ensure ethical and legal compliance (licensing, privacy level classification)
- Generate checksums for integrity verification
- Establish access rights
- Normalize metadata for consistency (metric naming, parameter typing, environment specifications)

---

### Phase 2 — Process

*Once assets are created with complete metadata, they are enriched with relationships and provenance chains to enhance discoverability and enable lineage tracking.*

**Objective:**
Enrich metadata with relationships, provenance chains, and cross-references to ensure interoperability and discoverability.

**Activities:**

- Establish **provenance relationships** (experiment → dataset, experiment → model, model → base_model)
- Harmonize computational usage metrics across different resource types
- Document preprocessing transformations and data augmentation strategies in ExperimentDataset associations
- Enrich assets with subject classifications and keywords for discoverability
- Cross-reference with external repositories and code repositories (Git commits, DOIs)

---

### Phase 3 — Analyse

*Enriched metadata with complete provenance enables meta-level analysis capabilities for comparative studies and institutional monitoring.*

**Objective:**
Enable meta-level analysis capabilities by providing structured access to experiment metrics, resource utilization, and provenance data.

**Activities:**

- Support **comparative queries** across experiments (model performance vs. computational cost, framework comparisons)
- Provide reproducibility metrics (environment specification completeness, configuration consistency)
- Enable **resource impact analysis**:
  - Energy consumption trends across experiments and resource types
  - Carbon footprint analysis by geographic location and time period
  - Cost efficiency evaluations (performance per dollar/kWh)
- Offer **analytical views** for institutional monitoring:
  - Model performance trends and distributions
  - Resource utilization patterns and efficiency
  - Environmental sustainability dashboards
  - Dataset reuse and citation metrics
- Support hyperparameter exploration and outcome correlation analysis

---

### Phase 4 — Preserve

*To enable long-term analysis and reuse, assets must be preserved in durable formats with guaranteed integrity and accessibility.*

**Objective:**
Ensure long-term preservation, integrity, and accessibility of all machine learning assets and metadata through tiered storage and format standardization.

**Activities:**

- Archive curated datasets, models, and metadata in institutional or national repositories (e.g., Zenodo)
- Store assets in **durable, non-proprietary formats**:
  - **Metadata:** JSON-LD, XML (ensuring standard compliance)
  - **Models:** ONNX, PyTorch (.pt), TensorFlow SavedModel
  - **Datasets:** Parquet, HDF5, CSV
  - **Configurations:** YAML, JSON
- Implement **tiered storage architecture**:
  - **Active tier:** High-performance storage (SSD) for frequently accessed assets
  - **Cold tier:** Cost-efficient storage (HDD, object storage) for infrequently accessed data
  - **Archival tier:** Durable, geo-redundant storage for long-term preservation (institutional repositories, cloud archival services)
- Verify data integrity through periodic checksum validation
- Maintain **redundant storage** across multiple geographic locations
- Ensure complete provenance chains documenting all transformations and version lineage
- Export enriched metadata in JSON-LD format with full standard compliance

**Metadata Retention:**

To ensure metadata accessibility even when data files are deleted, OpenModelsHub implements a comprehensive metadata persistence strategy:

- **Metadata-only records**: When data files or model binaries are removed, complete metadata records persist indefinitely in the repository database

- **Citation continuity**: Persistent identifiers (DOIs) continue to resolve to metadata records, enabling citation of deleted assets

- **Asset lifecycle states**:
  - **ACTIVE**: Full metadata + accessible data files
  - **DEPRECATED**: Full metadata + data files + deprecation notice with successor information
  - **DELETED**: Metadata-only record + no data file access

---

### Phase 5 — Share

*Preserved assets with validated metadata are ready for dissemination through public catalogs, APIs, and standard interfaces.*

**Objective:**
Disseminate preserved assets and metadata for discovery, reuse, and citation under FAIR and ethical principles.

**Activities:**

- Publish ML assets and metadata via OpenModelsHub's **public catalogue and REST API**:
  - Query and filter by metadata fields (framework, model type, date range, organization)
  - Full-text search across descriptions and keywords
- Assign appropriate **licenses** (MIT, Apache 2.0, Creative Commons, Open Data Commons)
- Configure access rights based on privacy, ethics, and institutional policies
- Expose metadata through standard interfaces (REST API)
- Link datasets, experiments, and models with associated publications and code repositories
- Generate citation information in standard formats (BibTeX, RIS, CSL-JSON)

**External Catalog and Registry Integration:**

To enable discoverability through established research infrastructure, OpenModelsHub integrates with external catalogs and registries:

- **DataCite DOI Registration**: DOI resolution directs to OpenModelsHub landing pages with full metadata display
- **Repository Registry Listing**: OpenModelsHub registered in re3data.org (Registry of Research Data Repositories)
- **Web Discoverability**: Enables indexing by Google Dataset Search, and academic search engines
- **Disciplinary Catalogs**: Model registration in community platforms (Papers with Code model index, Hugging Face Model Hub when applicable) and Dataset cross-referencing with OpenML, UCI ML Repository.


---

### Phase 6 — Reuse

*Shared assets enable replication, benchmarking, and meta-research, with usage metrics feeding back into asset creation and enhancement.*

**Objective:**
Enable other researchers to reuse ML assets for replication, benchmarking, or meta-research, with tracking that feeds back into the lifecycle.

**Activities:**

- Allow replication of experiments using preserved configurations and environments
- Enable search across all metadata fields:
  - By framework (PyTorch, TensorFlow, etc.)
  - By model type (classification, generation, etc.)
  - By dataset format (CSV, Parquet, etc.)
  - By organization, researcher, or time period
- Support browsing by provenance relationships:
  - View experiment → model lineage
  - Explore dataset → experiment usage
  - Navigate version history trees
- Link derived works to original assets through version lineage (parent_version)
- Ensure code and configuration availability for reproducibility
- Track and report **reuse metrics** that inform future asset creation (Phase 1):
  - Asset downloads and access patterns
  - Citation counts and trends
  - Derived experiments and models
  - Cross-institutional usage statistics
  - Dataset reuse frequency and combinations


---

*For entity specifications, see [logical-model.md](../data-model/logical-model.md). For metadata standards, see [standards-reference.md](../metadata/standards-reference.md).*
