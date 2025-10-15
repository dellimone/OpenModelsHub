# Metadata Integration Mapping

## Integration Architecture

OpenModelsHub uses a **layered integration** approach:

1. **Foundation Layer**: Dublin Core, Schema.org, DataCite, DCAT, and PROV-O provide universal metadata for all assets
2. **ML Layer**: Croissant, ML-Schema, FAIR4ML, MLModel, and MEX add domain-specific vocabulary
3. **Infrastructure Layer**: HPC Ontology describes computational resources and usage

This multi-standard approach ensures:
- **Broad Interoperability**: Foundation standards enable integration with general-purpose repositories
- **Domain Specificity**: ML standards capture specialized training, performance, and reproducibility metadata

The `omh:` namespace prefix identifies OpenModelsHub-specific extensions while maintaining standard compliance.

**Namespace Strategy:**
```
dc:       http://purl.org/dc/elements/1.1/
dct:      http://purl.org/dc/terms/
schema:   http://schema.org/
dcat:     http://www.w3.org/ns/dcat#
prov:     http://www.w3.org/ns/prov#
cr:       http://mlcommons.org/croissant/
mls:      http://www.w3.org/ns/mls#
fair4ml:  https://w3id.org/fair4ml#
mex:      http://mex.aksw.org/mex-core
hpc:      https://hpc-fair.github.io/ontology#
omh:      https://openmodelshub.org/ontology#
```

## Foundation Entity Mappings

### MLAsset (Base Class)

All ML assets inherit these verified standard mappings:

| OpenModelsHub Attribute | Dublin Core | Schema.org | DataCite | PROV-O |
|-------------------------|-------------|------------|----------|--------|
| `id` | — | `@id` | `Identifier` | `prov:Entity` |
| `persistent_identifier` | `dc:identifier` | `schema:identifier` | `datacite:identifier` | — |
| `name` | `dc:title` | `schema:name` | `datacite:title` | — |
| `description` | `dc:description` | `schema:description` | `datacite:description` | — |
| `version` | — | `schema:version` | `datacite:version` | — |
| `created_at` | `dc:date` | `schema:dateCreated` | `datacite:date` | — |
| `updated_at` | `dct:modified` | `schema:dateModified` | — | — |
| `created_by` | `dc:creator` | `schema:creator` | `datacite:creator` | `prov:wasAttributedTo` |
| `organization` | `dc:publisher` | `schema:publisher` | `datacite:publisher` | — |
| `license` | `dc:rights` | `schema:license` | `datacite:rights` | — |
| `subjects` | `dc:subject` | `schema:keywords` | `datacite:subject` | — |
| `access_rights` | `dct:accessRights` | — | `datacite:rights` | — |
| `parent_version` | `dct:isVersionOf` | `schema:isBasedOn` | `datacite:relatedIdentifier` | `prov:wasDerivedFrom` |

### Researcher

| OpenModelsHub Attribute | Dublin Core | Schema.org | DataCite | PROV-O |
|-------------------------|-------------|------------|----------|--------|
| Researcher entity | `dc:creator` | `schema:Person` | `datacite:creator` | `prov:Agent` |
| `id` | — | `schema:@id` | `datacite:Identifier` | — |
| `first_name` | — | `schema:givenName` | `datacite:givenName` | — |
| `last_name` | — | `schema:familyName` | `datacite:familyName` | — |
| `email` | — | `schema:email` | — | — |
| `orcid_id` | `dc:identifier` | `schema:identifier` | `datacite:nameIdentifier` (scheme=ORCID) | — |
| `expertise_areas` | — | `schema:knowsAbout` | — | — |
| `research_interests` | — | `schema:knowsAbout` | — | — |
| Relationship to Organizations | — | `schema:memberOf` | `datacite:affiliation` | — |
| Relationship to ML Assets | `dc:creator` | `schema:creator` | `datacite:creator` | `prov:wasAttributedTo` |


### Organization

| OpenModelsHub Attribute | Dublin Core | Schema.org | DataCite | PROV-O |
|-------------------------|-------------|------------|----------|--------|
| Organization entity | `dc:publisher` | `schema:Organization` | `datacite:publisher` / `datacite:contributor` | — |
| `id` | — | `schema:@id` | `datacite:Identifier` | — |
| `name` | `dc:publisher` | `schema:name` | `datacite:publisher` | — |
| `type` | — | `schema:@type` (Organization types) | `datacite:contributorType` | — |
| `location` | — | `schema:address` | — | — |
| `website` | — | `schema:url` | — | — |
| `policies` | — | — | — | — |
| Relationship to Researchers | — | `schema:member` | — | — |
| Relationship to Resources | — | `schema:ownerOf` | — | — |

## ML-Specific Entity Mappings

### Model

| OpenModelsHub Attribute | ML-Schema | FAIR4ML | MEX | PROV-O |
|-------------------------|-----------|---------|--------------|--------|
| Model entity | `mls:Model` | `fair4ml:MLModel` | `mex:Model` | `prov:Entity` |
| `architecture` | `mls:algorithm` | `fair4ml:modelCategory` | `mex:Algorithm` | — |
| `framework` | `mls:software` | `fair4ml:softwareRequirements` | `mex:Tool` | — |
| `framework_version` | — | — | — | — |
| `model_type` | — | `fair4ml:mlTask` | — | — |
| `input_schema` | — | Related to `fair4ml:usageInstructions` | — | — |
| `output_schema` | — | Related to `fair4ml:usageInstructions` | — | — |
| `training_datasets` | `mls:dataset` | `fair4ml:trainedOn` | `om:mlFeatures.featureSources` | `prov:used` |


### Dataset

| OpenModelsHub Attribute | Schema.org | DCAT | Croissant | ML-Schema | 
|-------------------------|------------|------|-----------|-----------|
| Dataset entity | `schema:Dataset` | `dcat:Dataset` | Extends `schema:Dataset` | `mls:Dataset` | — | — |
| `file_paths` | `schema:distribution` | `dcat:distribution` | `cr:url` | — | — | — |
| `total_size_bytes` | `schema:contentSize` | `dcat:byteSize` | — | — | — | — |
| `format` | `schema:encodingFormat` | `dcat:mediaType` | `cr:encodingFormat` | — | — | — |
| `schema` | `schema:variableMeasured` | — | `cr:RecordSet` | — | — | — |
| `num_records` | — | — | `cr:recordCount` | `mls:DatasetCharacteristic` | — | — |
| `num_features` | — | — | `cr:Field` | `mls:DatasetCharacteristic` | — | — |
| `target_column` | — | — | `cr:Field` | — | — | — |
| `data_types` | `schema:variableMeasured` | — | `cr:dataType` | — | — | — |
| `categorical_columns` | — | — | `cr:Field` | — | — | — |
| `numerical_columns` | — | — | `cr:Field` | — | — | — |


### Experiment

| OpenModelsHub Attribute | ML-Schema | FAIR4ML | MEX | PROV-O |
|-------------------------|-----------|---------|-----|--------|
| Experiment entity | `mls:Run` | `fair4ml:TrainingRun` | `mex:Execution` | `prov:Activity` |
| `experiment_type` | — | — | `mex:ApplicationContext` | — |
| `start_time` | — | — | — | `prov:startedAtTime` |
| `end_time` | — | — | — | `prov:endedAtTime` |
| `code_repository_url` | — | Inherits `schema:codeRepository` | — | — |
| `environment_specification` | — | `fair4ml:softwareRequirements` | — | — |
| `base_model`relationship | `mls:Model` | `fair4ml:fineTunedFrom` | — | `prov:used` |
| `datasets` relationship | `mls:hasInput` | — | — | `prov:used` |
| `produced_models` relationship | `mls:hasOutput` | — |`prov:wasGeneratedBy` | — |
| `hyperparameters` relationship | `mls:hasHyperParameter` | — | Via `mex:HyperParameter` | — | — |
| `metrics` relationship | `mls:hasQuality` | — | `mexperf:PerformanceMeasure` | — |
| `checkpoints` relationship | — | — | — | `prov:wasGeneratedBy` |
| `resource_utilizations` relationship | — | — | `mex:Execution` | `prov:Activity` |

## Support Entity Mappings

### Hyperparameter

| OpenModelsHub Attribute | ML-Schema | MEX |
|-------------------------|-----------|-----|
| Hyperparameter entity | `mls:HyperParameter` | `mex:HyperParameter` |
| Relationship to Experiment | `mls:hasHyperParameter` | `mex:HyperParameterCollection` |

### ExperimentMetric

| OpenModelsHub Attribute | ML-Schema | MEX | PROV-O |
|-------------------------|-----------|-----|--------|
| Metric entity | `mls:Evaluation` + `mls:Measure` | `mexperf:PerformanceMeasure` | — |
| `metric_name` | `mls:Measure` | — | — |
| `value` | `mls:hasValue` | — | — |
| `timestamp` | — | — | `prov:generatedAtTime` |
| `metric_type` | `mls:Measure` | `mexperf` | — |



### Checkpoint

| OpenModelsHub Attribute | PROV-O | ML-Schema |
|-------------------------|--------|-----------|
| Checkpoint entity | `prov:Entity` |  `mls:Model` |
| `checkpoint_name` | — | — |
| `saved_at` | `prov:generatedAtTime` | — |
| `metrics_snapshot` | — | `mls:Evaluation` |
| Relationship to Experiment | `prov:wasGeneratedBy` | — |
| Relationship to Model | — |  `mls:Model` |


### ExperimentDataset

| OpenModelsHub Attribute | Croissant | ML-Schema | MEX |
|-------------------------|-----------|-----------|-----|
| ExperimentDataset entity | `cr:Split` | `mls:DatasetCharacteristic` | `mex:Phase` |
| `role` | — | Via `mls:hasInput` | `mex:Phase` |
| `split_percentage` | Via `cr:Split` | — | — |
| `num_records` | `cr:recordCount` | `mls:DatasetCharacteristic` | — |
| `random_seed` | Via `cr:Split` | — | — |
| `preprocessing_applied` | — | `mls:DataTransformation` | — |
| `augmentation_enabled` | — | `mls:DataTransformation` | — |
| Relationship to Experiment | — | `mls:hasInput` | `mex:Execution` |
| Relationship to Dataset | Parent `cr:Dataset` | `mls:dataset` | — |

### ComputationalResource (Base Class)

| OpenModelsHub Attribute | HPC Ontology |
|-------------------------|--------------|
| Resource entity | `hpc:Hardware` / `hpc:Computer` / `hpc:Server` |
| `resource_type` | Determines subclass: LOCAL → `hpc:Workstation`/`hpc:Server`, CLOUD → `hpc:Computer`, HPC → `hpc:Cluster` |
| `cpu_cores` | `hpc:cpuCoreCount` |
| `memory_gb` | `hpc:memorySize` |
| `gpu_count` | Via `hpc:Accelerator` count |
| `gpu_model` | Via `hpc:Accelerator` properties |
| `energy_efficiency` | `hpc:powerEfficiency` |

### LocalResource

| OpenModelsHub Attribute | HPC Ontology |
|-------------------------|--------------|
| LocalResource entity | `hpc:Workstation` / `hpc:Server` |
| `hostname` | Via `hpc:Computer` identifier |
| `power_consumption_watts` | `hpc:power` |

### CloudResource

| OpenModelsHub Attribute | HPC Ontology |
|-------------------------|--------------|
| CloudResource entity | `hpc:Computer` |
| `provider_region` | `hpc:country ` |

### HPCResource

| OpenModelsHub Attribute | HPC Ontology |
|-------------------------|--------------|
| HPCResource entity | `hpc:Cluster` / `hpc:SuperComputer` |
| `cluster_name` | Via `hpc:Cluster` identifier |
| `total_nodes` | `hpc:computeNodeCount` |
| `node_configuration` | `hpc:systemArchitecture` |
| `max_job_duration` | Related to `hpc:maxExecutionTime` |
| `interconnect_type` | `hpc:gpuInterconnect` |

### ResourceUtilization

| OpenModelsHub Attribute | HPC Ontology |
|-------------------------|--------------|
| ResourceUtilization entity | Related to `hpc:Hardware` usage |
| `resource` | → `hpc:Computer` / `hpc:Cluster` |
| `duration_seconds` | `hpc:averageExecutionTime` |
| `cpu_utilization_percent` | `hpc:streamingProcessorUtilizationRate` |
| `gpu_utilization_percent` | `hpc:streamingProcessorUtilizationRate` (for GPU) |
| `memory_usage_gb` | `hpc:memoryOccupancy` |
| `storage_usage_gb` | Related to `hpc:harddriveSize` |
| `network_io_gb` | `hpc:dataTransferSize` |
| `energy_consumption_kwh` | Related to `hpc:power` |
| `peak_usage_metrics` | `hpc:maxExecutionTime`, various utilization properties |
| `average_usage_metrics` | `hpc:averageExecutionTime`, `hpc:memoryThroughputRate` |

---

*For standards descriptions, see [standards-reference.md](standards-reference.md). For schema implementation, see [schemas/omh-schema.xsd](schemas/omh-schema.xsd).*
