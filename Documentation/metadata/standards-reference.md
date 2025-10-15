# Metadata Standards Reference

## Overview

OpenModelsHub integrates 11 international metadata standards to ensure FAIR (Findable, Accessible, Interoperable, Reusable) compliance for ML research assets.

## Standards Catalog

### Foundation Standards

These universal metadata standards provide base functionality for asset identification, attribution, and provenance.

| Standard | Version | Authority | Purpose | Namespace | Official Link |
|----------|---------|-----------|---------|-----------|---------------|
| **Dublin Core (DCMI Terms)** | 2020-01-20 | Dublin Core Metadata Initiative | Minimal, reusable metadata terms for digital resources | `http://purl.org/dc/terms/` | [dublincore.org](https://www.dublincore.org/specifications/dublin-core/dcmi-terms/) |
| **Schema.org Dataset** | Rolling | Schema.org Community | Web-discoverable dataset metadata with search engine compatibility | `http://schema.org/` | [schema.org/Dataset](https://schema.org/Dataset) |
| **DataCite Metadata** | 4.6 (2024) | DataCite Consortium | Citation metadata and DOI assignment for research data | N/A (XML Schema) | [DataCite Schema 4.6](https://datacite-metadata-schema.readthedocs.io/en/4.6/) |
| **W3C DCAT** | 2.0 (2020-02-04) | W3C (Recommendation) | Data catalog vocabulary for dataset distribution | `http://www.w3.org/ns/dcat#` | [W3C DCAT 2](https://www.w3.org/TR/vocab-dcat-2/) |
| **W3C PROV-O** | 1.0 (2013-04-30) | W3C (Recommendation) | Provenance ontology for tracking origin and transformation | `http://www.w3.org/ns/prov#` | [W3C PROV-O](https://www.w3.org/TR/prov-o/) |

### ML-Specific Standards

Domain-specific vocabularies for machine learning models, datasets, experiments, and training workflows.

| Standard | Version | Authority | Purpose | Namespace | Official Link |
|----------|---------|-----------|---------|-----------|---------------|
| **Croissant** | 1.0 (2024-03-01) | MLCommons | ML dataset metadata extension to Schema.org | `http://mlcommons.org/croissant/` | [Croissant Spec](https://docs.mlcommons.org/croissant/docs/croissant-spec.html) |
| **ML-Schema** | 2016-10-17 | W3C ML Schema Community Group | Ontology for ML artifacts (algorithms, models, datasets, runs) | `http://www.w3.org/ns/mls#` | [ML-Schema](https://ml-schema.github.io/documentation/ML%20Schema.html) |
| **FAIR4ML** | 0.1.0 (2024-10-27) | RDA FAIR4ML Interest Group | FAIR-aligned ML model metadata for findability and reuse | `https://w3id.org/fair4ml#` | [FAIR4ML v0.1.0](https://rda-fair4ml.github.io/FAIR4ML-schema/release/0.1.0/index.html) |
| **MEX Vocabulary** | 1.0.2 (2015) | AKSW Research Group | Lightweight experiment metadata (extends PROV-O) | `http://mex.aksw.org/mex-core` | [MEX GitHub](https://github.com/mexplatform/mex-vocabulary) |

### Infrastructure Standards

Computational resource vocabulary for hardware, clusters, and environmental impact tracking.

| Standard | Version | Authority | Purpose | Namespace | Official Link |
|----------|---------|-----------|---------|-----------|---------------|
| **HPC Ontology** | 1.1.0 (2022-12-09) | HPC-FAIR Initiative | Semantic framework for HPC resources, hardware specs, and compute environments | `https://hpc-fair.github.io/ontology#` | [HPC Ontology](https://hpc-fair.github.io/ontology/) |

## Entity-to-Standards Coverage Matrix

This matrix shows which standards apply to each OpenModelsHub entity based on verified vocabulary coverage.

| Entity                    | Dublin Core | Schema.org | DataCite | DCAT | PROV-O | Croissant | ML-Schema | FAIR4ML | MEX | HPC Ontology |
| ------------------------- | :---------: | :--------: | :------: | :--: | :----: | :-------: | :-------: | :-----: | :-: | :----------: |
| **MLAsset (Base)**        |      ✓      |      ✓     |     ✓    |   ✓  |    ✓   |     —     |     —     |    —    |  —  |       —      |
| **Model**                 |      ✓      |      ✓     |     ✓    |   —  |    ✓   |     —     |     ✓     |    ✓    |  —  |       —      |
| **Dataset**               |      ✓      |      ✓     |     ✓    |   ✓  |    ✓   |     ✓     |     ✓     |    —    |  —  |       —      |
| **Experiment**            |      ✓      |      —     |     ✓    |   —  |    ✓   |     —     |     ✓     |    ✓    |  ✓  |       —      |
| **Hyperparameter**        |      —      |      —     |     —    |   —  |    —   |     —     |     ✓     |    —    |  ✓  |       —      |
| **ExperimentMetric**      |      —      |      —     |     —    |   —  |    —   |     —     |     ✓     |    —    |  ✓  |       —      |
| **Checkpoint**            |      —      |      —     |     —    |   —  |    ✓   |     —     |     ✓     |    —    |  ✓  |       —      |
| **ExperimentDataset**     |      —      |      —     |     —    |   —  |    —   |     ✓     |     ✓     |    —    |  ✓  |       —      |
| **ComputationalResource** |      —      |      —     |     —    |   —  |    —   |     —     |     —     |    —    |  —  |       ✓      |
| **LocalResource**         |      —      |      —     |     —    |   —  |    —   |     —     |     —     |    —    |  —  |       ✓      |
| **CloudResource**         |      —      |      —     |     —    |   —  |    —   |     —     |     —     |    —    |  —  |       ✓      |
| **HPCResource**           |      —      |      —     |     —    |   —  |    —   |     —     |     —     |    —    |  —  |       ✓      |
| **ResourceUtilization**   |      —      |      —     |     —    |   —  |    ✓   |     —     |     —     |    —    |  ✓  |       ✓      |
| **Researcher**            |      ✓      |      ✓     |     ✓    |   —  |    ✓   |     —     |     —     |    —    |  —  |       —      |
| **Organization**          |      ✓      |      ✓     |     ✓    |   —  |    ✓   |     —     |     —     |    —    |  —  |       —      |


## Standard Descriptions

### Dublin Core (DCMI Terms)

**Description:**
A universal metadata standard defining 15 core elements for describing digital resources, widely used for cross-domain interoperability.

**Key Vocabulary:** 
`title`, `creator`, `description`, `date`, `identifier`, `publisher`, `rights`, `subject`

**Use in OpenModelsHub:**
Provides foundational metadata for all ML assets (MLAsset base class). Essential for basic resource description and cross-repository interoperability.

### Schema.org Dataset

**Description:**
A web-oriented metadata vocabulary for structured data, enabling search engines to index and interpret datasets. The Dataset class standardizes descriptions of data collections, distributions, and coverage

**Key Classes & Properties**: 
`Dataset`, `Person`, `Organization`, `name`, `description`, `identifier`, `creator`, `distribution`, `contentSize`, `encodingFormat`

**Use in OpenModelsHub:**
Facilitates web discoverability. Provides base vocabulary extended by Croissant for ML datasets.

### DataCite Metadata Schema

**Description:**
A metadata standard for research data citation, providing required and recommended properties data formally citable and trackable.

**Required Properties:** 
`Identifier`, `Creator`, `Title`, `Publisher`, `PublicationYear`, `ResourceType`

**Additional Properties:**
`Subject`, `Contributor`, `Date`, `RelatedIdentifier`, `Description`, `Rights`

**Use in OpenModelsHub:**
DOI assignment and research citation. Supports dataset and model publication with persistent identifiers.

### W3C DCAT

**Description:**
A W3C standard for describing datasets, distributions, and data services, enabling interoperability between web-based data catalogs.

**Key Classes & Properties**:
`Dataset`, `Distribution`, `Catalog`, `distribution`, `byteSize`, `mediaType`, `keyword`, `theme`, `accessRights`

**Use in OpenModelsHub:**
Dataset cataloging and distribution metadata. Links datasets to downloadable distributions.

### W3C PROV-O

**Description:**
A W3C ontology for representing provenance, defining Entities, Activities, and Agents to capture how digital objects are created, modified, and used.

**Key Classes & Relationship:**
`Entity`, `Activity`, `Agent`, `wasGeneratedBy`, `used`, `wasAttributedTo`, `wasDerivedFrom`, `wasAssociatedWith`, `startedAtTime`, `endedAtTime`

**Use in OpenModelsHub:**
Records complete provenance chains for ML experiments, linking Datasets/Models to Experiments and generated Models.

### Croissant

**Description:**
An MLCommons metadata format extending Schema.org Dataset, capturing dataset structure at file, record and features including transformations and relationships.

**Key Classes & Properties**:
`RecordSet`, `Field`, `Split`, `url`, `encodingFormat`, `recordCount`, `dataType`

**Use in OpenModelsHub:**
Represents internal dataset features, labels, splits, types, and preprocessing—enabling automated, reproducible data loading and ML experimentation.

### ML-Schema

**Description:**
A W3C Community ontology for ML experiments, defining Experiments, Runs, Algorithms, Models, Data, Evaluations, and HyperParameters, with structured relationships linking datasets, algorithms, runs, models, and evaluations.

**Key Classes & Properties**: 
`Run`, `Model`, `Algorithm`, `Dataset`, `HyperParameter`, `hasInput`, `hasOutput`, `executes`, `realizes`, `implements`, `hasHyperParameter`, `definedOn`

**Use in OpenModelsHub:**
Captures complete ML experiment workflows, connecting Datasets, Algorithms, Runs, Models, and Evaluations. Treats hyperparameters as first-class entities.

### FAIR4ML

**Description:**
A metadata schema for FAIR ML models, extending Schema.org and CodeMeta.

**Key Classes & Properties:** 
`MLModel`, `MLModelEvaluation`, `trainedOn`, `validatedOn`, `testedOn`, `fineTunedFrom`

**Use in OpenModelsHub:**
FAIR compliance assessment and comprehensive model documentation. Supports ethical AI and environmental impact tracking.


### MEX Vocabulary

**Description:**
A lightweight vocabulary for ML experiment metadata, extending PROV-O with three layers: mexcore, mexalgo, mexperf

**Key Classes & Properties:** 
`Experiment`, `ExperimentConfiguration`, `Execution`, `Dataset`, `Model`, `Phase`, `Algorithm`, `Tool`, `HyperParameter`, `LearningMethod`, `PerformanceMeasure`, `ExecutionPerformance`

**Use in OpenModelsHub:** Experiment tracking with performance measures. Provides hyperparameter and measurement vocabulary.

### HPC Ontology

**Description:**
A semantic framework for describing high-performance computing resources, hardware, and compute environments, including systems, CPUs/GPUs, memory, storage, and execution metrics.

**Key Classes & Properties:**
`Cluster`, `Hardware`, `Processor`, `Accelerator`, `Memory`, `Computer`, `Server`,`cpuCoreCount`, `cpuFrequency`, `memorySize`, `gpuMemorySizePerNode`, `processorPeakPerformance`

**Use in OpenModelsHub:**
Tracks resources across Local, Cloud, and HPC systems. Records hardware specifications, utilization metrics and power consumption for energy and carbon footprint analysis.

---

*For detailed attribute mappings, see [integration-mapping.md](integration-mapping.md). For schema implementation, see [schemas/omh-unified-schema.xsd](schemas/omh-unified-schema.xsd).*
