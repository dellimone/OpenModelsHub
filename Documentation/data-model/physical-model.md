# Physical Data Model

## Introduction

This document translates the logical model into concrete Django ORM implementations, providing complete code examples with primary keys, foreign keys, indexes, and Django-specific patterns.

### Design Principles

- **Abstract Base Classes**: Shared functionality through inheritance
- **Multi-table Inheritance**: Resource specialization (Local/Cloud/HPC)
- **JSON Fields**: Flexible storage for variable configuration data
- **UUID Primary Keys**: Global uniqueness for distributed systems


## Core Django Patterns

### Enumerations and Choices


```python
from django.db import models

class AccessRights(models.TextChoices):
    PUBLIC = 'PUBLIC', 'Public'
    REGISTERED = 'REGISTERED', 'Registered Users'
    RESTRICTED = 'RESTRICTED', 'Restricted Access'
    EMBARGOED = 'EMBARGOED', 'Embargoed'

class ChecksumAlgorithm(models.TextChoices):
    MD5 = 'MD5', 'MD5 (legacy)'
    SHA1 = 'SHA1', 'SHA-1 (legacy)'
    SHA256 = 'SHA256', 'SHA-256'
    SHA512 = 'SHA512', 'SHA-512'

class ModelFormat(models.TextChoices):
    PYTORCH = 'PYTORCH', 'PyTorch'
    TENSORFLOW_SAVEDMODEL = 'TENSORFLOW_SAVEDMODEL', 'TensorFlow SavedModel'
    TENSORFLOW_H5 = 'TENSORFLOW_H5', 'TensorFlow H5'
    ONNX = 'ONNX', 'ONNX'
    KERAS = 'KERAS', 'Keras'
    SCIKIT_LEARN = 'SCIKIT_LEARN', 'Scikit-learn'
    XGBOOST = 'XGBOOST', 'XGBoost'
    LIGHTGBM = 'LIGHTGBM', 'LightGBM'
    OTHER = 'OTHER', 'Other'

class Framework(models.TextChoices):
    PYTORCH = 'PYTORCH', 'PyTorch'
    TENSORFLOW = 'TENSORFLOW', 'TensorFlow'
    KERAS = 'KERAS', 'Keras'
    SCIKIT_LEARN = 'SCIKIT_LEARN', 'Scikit-learn'
    JAX = 'JAX', 'JAX'
    MXNET = 'MXNET', 'MXNet'
    XGBOOST = 'XGBOOST', 'XGBoost'
    LIGHTGBM = 'LIGHTGBM', 'LightGBM'
    CATBOOST = 'CATBOOST', 'CatBoost'
    HUGGINGFACE = 'HUGGINGFACE', 'Hugging Face'
    OTHER = 'OTHER', 'Other'

class ModelType(models.TextChoices):
    CLASSIFICATION = 'CLASSIFICATION', 'Classification'
    REGRESSION = 'REGRESSION', 'Regression'
    CLUSTERING = 'CLUSTERING', 'Clustering'
    GENERATION = 'GENERATION', 'Generation'
    TRANSLATION = 'TRANSLATION', 'Translation'
    SUMMARIZATION = 'SUMMARIZATION', 'Summarization'
    QUESTION_ANSWERING = 'QUESTION_ANSWERING', 'Question Answering'
    OBJECT_DETECTION = 'OBJECT_DETECTION', 'Object Detection'
    IMAGE_SEGMENTATION = 'IMAGE_SEGMENTATION', 'Image Segmentation'
    SPEECH_RECOGNITION = 'SPEECH_RECOGNITION', 'Speech Recognition'
    REINFORCEMENT_LEARNING = 'REINFORCEMENT_LEARNING', 'Reinforcement Learning'
    OTHER = 'OTHER', 'Other'

class DatasetFormat(models.TextChoices):
    CSV = 'CSV', 'CSV'
    JSON = 'JSON', 'JSON'
    JSONL = 'JSONL', 'JSON Lines'
    PARQUET = 'PARQUET', 'Parquet'
    HDF5 = 'HDF5', 'HDF5'
    ARROW = 'ARROW', 'Arrow'
    AVRO = 'AVRO', 'Avro'
    TFRECORD = 'TFRECORD', 'TFRecord'
    PICKLE = 'PICKLE', 'Pickle'
    NPY = 'NPY', 'NumPy'
    IMAGE_FOLDER = 'IMAGE_FOLDER', 'Image Folder'
    TEXT_FILES = 'TEXT_FILES', 'Text Files'
    AUDIO_FILES = 'AUDIO_FILES', 'Audio Files'
    OTHER = 'OTHER', 'Other'

class PrivacyLevel(models.TextChoices):
    PUBLIC = 'PUBLIC', 'Public'
    INTERNAL = 'INTERNAL', 'Internal'
    CONFIDENTIAL = 'CONFIDENTIAL', 'Confidential'
    RESTRICTED = 'RESTRICTED', 'Restricted'
    ANONYMIZED = 'ANONYMIZED', 'Anonymized'

class ExperimentType(models.TextChoices):
    TRAINING = 'TRAINING', 'Training'
    VALIDATION = 'VALIDATION', 'Validation'
    TESTING = 'TESTING', 'Testing'
    HYPERPARAMETER_TUNING = 'HYPERPARAMETER_TUNING', 'Hyperparameter Tuning'
    FINE_TUNING = 'FINE_TUNING', 'Fine-tuning'
    TRANSFER_LEARNING = 'TRANSFER_LEARNING', 'Transfer Learning'
    BENCHMARK = 'BENCHMARK', 'Benchmark'
    OTHER = 'OTHER', 'Other'

class ExperimentStatus(models.TextChoices):
    PENDING = 'PENDING', 'Pending'
    RUNNING = 'RUNNING', 'Running'
    COMPLETED = 'COMPLETED', 'Completed'
    FAILED = 'FAILED', 'Failed'
    CANCELLED = 'CANCELLED', 'Cancelled'
    PAUSED = 'PAUSED', 'Paused'

class ParameterType(models.TextChoices):
    FLOAT = 'FLOAT', 'Float'
    INTEGER = 'INTEGER', 'Integer'
    STRING = 'STRING', 'String'
    BOOLEAN = 'BOOLEAN', 'Boolean'
    LIST = 'LIST', 'List'
    DICT = 'DICT', 'Dictionary'

class MetricType(models.TextChoices):
    LOSS = 'LOSS', 'Loss'
    ACCURACY = 'ACCURACY', 'Accuracy'
    PRECISION = 'PRECISION', 'Precision'
    RECALL = 'RECALL', 'Recall'
    F1 = 'F1', 'F1 Score'
    AUC = 'AUC', 'AUC'
    MAE = 'MAE', 'Mean Absolute Error'
    MSE = 'MSE', 'Mean Squared Error'
    RMSE = 'RMSE', 'Root Mean Squared Error'
    R2 = 'R2', 'R-squared'
    PERPLEXITY = 'PERPLEXITY', 'Perplexity'
    BLEU = 'BLEU', 'BLEU Score'
    CUSTOM = 'CUSTOM', 'Custom'

class SplitType(models.TextChoices):
    TRAINING = 'TRAINING', 'Training'
    VALIDATION = 'VALIDATION', 'Validation'
    TESTING = 'TESTING', 'Testing'
    HOLDOUT = 'HOLDOUT', 'Holdout'
    CUSTOM = 'CUSTOM', 'Custom'

class ResourceType(models.TextChoices):
    LOCAL = 'LOCAL', 'Local'
    CLOUD = 'CLOUD', 'Cloud'
    HPC = 'HPC', 'HPC'

class Availability(models.TextChoices):
    AVAILABLE = 'AVAILABLE', 'Available'
    BUSY = 'BUSY', 'Busy'
    MAINTENANCE = 'MAINTENANCE', 'Maintenance'
    OFFLINE = 'OFFLINE', 'Offline'

class CoolingType(models.TextChoices):
    AIR = 'AIR', 'Air'
    LIQUID = 'LIQUID', 'Liquid'
    PASSIVE = 'PASSIVE', 'Passive'
    HYBRID = 'HYBRID', 'Hybrid'

class ChassisType(models.TextChoices):
    DESKTOP = 'DESKTOP', 'Desktop'
    TOWER = 'TOWER', 'Tower'
    RACK = 'RACK', 'Rack'
    BLADE = 'BLADE', 'Blade'

class CloudProvider(models.TextChoices):
    AWS = 'AWS', 'Amazon Web Services'
    AZURE = 'AZURE', 'Microsoft Azure'
    GCP = 'GCP', 'Google Cloud Platform'
    IBM_CLOUD = 'IBM_CLOUD', 'IBM Cloud'
    ORACLE_CLOUD = 'ORACLE_CLOUD', 'Oracle Cloud'
    ALIBABA_CLOUD = 'ALIBABA_CLOUD', 'Alibaba Cloud'
    DIGITALOCEAN = 'DIGITALOCEAN', 'DigitalOcean'
    OTHER = 'OTHER', 'Other'

class VirtualizationType(models.TextChoices):
    HVM = 'HVM', 'Hardware Virtual Machine'
    PV = 'PV', 'Paravirtual'
    CONTAINER = 'CONTAINER', 'Container'

class Tenancy(models.TextChoices):
    SHARED = 'SHARED', 'Shared'
    DEDICATED = 'DEDICATED', 'Dedicated'
    HOST = 'HOST', 'Dedicated Host'

class CloudPricing(models.TextChoices):
    ON_DEMAND = 'ON_DEMAND', 'On-Demand'
    RESERVED = 'RESERVED', 'Reserved'
    SPOT = 'SPOT', 'Spot'
    SAVINGS_PLAN = 'SAVINGS_PLAN', 'Savings Plan'

class SchedulerType(models.TextChoices):
    SLURM = 'SLURM', 'Slurm'
    PBS = 'PBS', 'PBS'
    LSF = 'LSF', 'LSF'
    SGE = 'SGE', 'SGE'
    TORQUE = 'TORQUE', 'TORQUE'

class AllocationPolicy(models.TextChoices):
    EXCLUSIVE = 'EXCLUSIVE', 'Exclusive'
    SHARED = 'SHARED', 'Shared'

class InterconnectType(models.TextChoices):
    INFINIBAND = 'INFINIBAND', 'InfiniBand'
    ETHERNET = 'ETHERNET', 'Ethernet'
    OMNI_PATH = 'OMNI_PATH', 'Omni-Path'
    PROPRIETARY = 'PROPRIETARY', 'Proprietary'

class ParallelFilesystem(models.TextChoices):
    LUSTRE = 'LUSTRE', 'Lustre'
    GPFS = 'GPFS', 'GPFS'
    BEEGFS = 'BEEGFS', 'BeeGFS'
    NFS = 'NFS', 'NFS'
    CEPH = 'CEPH', 'Ceph'

class TrainingRole(models.TextChoices):
    TRAINING = 'TRAINING', 'Training'
    VALIDATION = 'VALIDATION', 'Validation'
    TESTING = 'TESTING', 'Testing'
    HOLDOUT = 'HOLDOUT', 'Holdout'

class OrganizationType(models.TextChoices):
    UNIVERSITY = 'UNIVERSITY', 'University'
    RESEARCH_INSTITUTE = 'RESEARCH_INSTITUTE', 'Research Institute'
    CORPORATION = 'CORPORATION', 'Corporation'
    GOVERNMENT = 'GOVERNMENT', 'Government'
    NON_PROFIT = 'NON_PROFIT', 'Non-Profit'
    CONSORTIUM = 'CONSORTIUM', 'Consortium'
```

---

### MLAsset Abstract Base Class

The foundation for all ML assets using Django's abstract base class pattern:

```python
import uuid
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class MLAsset(models.Model):
    """
    Abstract base class for all ML assets (Models, Datasets, Experiments).
    Provides common identity, metadata, versioning, and ownership functionality.
    """
    # Identity
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        help_text="System-generated unique identifier"
    )
    persistent_identifier = models.CharField(
        max_length=255,
        unique=True,
        help_text="External persistent ID (DOI, ARK, etc.)"
    )

    # Basic info
    name = models.CharField(
        max_length=255,
        db_index=True,
        help_text="Human-readable asset name"
    )
    description = models.TextField(
        help_text="Detailed description of the asset"
    )
    version = models.CharField(
        max_length=50,
        help_text="Version identifier (e.g., v2.1.0)"
    )

    # Timestamps
    created_at = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Creation timestamp"
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Last modification timestamp"
    )

    # Ownership
    created_by = models.ForeignKey(
        'Researcher',
        on_delete=models.PROTECT,
        related_name='%(class)s_created',
        help_text="Asset creator"
    )
    organization = models.ForeignKey(
        'Organization',
        on_delete=models.PROTECT,
        related_name='%(class)s_assets',
        help_text="Owning institution"
    )

    # Versioning (simplified parent-child)
    parent_version = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name='child_versions',
        help_text="Parent version in lineage"
    )
    version_notes = models.TextField(
        blank=True,
        help_text="Description of version changes"
    )

    # Metadata
    license = models.CharField(
        max_length=100,
        help_text="Usage license (SPDX identifier preferred)"
    )
    subjects = models.JSONField(
        default=list,
        help_text="Classification keywords and topics"
    )
    access_rights = models.CharField(
        max_length=20,
        choices=AccessRights.choices,
        help_text="Access level"
    )

    # Integrity
    checksum = models.CharField(
        max_length=128,
        help_text="File integrity hash"
    )
    checksum_algorithm = models.CharField(
        max_length=20,
        choices=ChecksumAlgorithm.choices,
        default=ChecksumAlgorithm.SHA256,
        help_text="Hash algorithm used"
    )

    class Meta:
        abstract = True
        indexes = [
            models.Index(fields=['created_at', 'access_rights']),
            models.Index(fields=['organization', 'created_at']),
        ]
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name} v{self.version}"

    def get_version_lineage(self):
        """
        Recursively retrieve version lineage from root to this asset.
        Returns list of assets from oldest to newest.
        """
        lineage = []
        current = self
        while current:
            lineage.insert(0, current)
            current = current.parent_version
        return lineage
```

---

### Concrete Implementations

#### Model

```python
class Model(MLAsset):
    """
    Trained machine learning model with performance metrics and deployment info.
    """
    # File info
    model_file_path = models.CharField(
        max_length=500,
        help_text="Location of model file"
    )
    model_file_size = models.BigIntegerField(
        validators=[MinValueValidator(1)],
        help_text="Size in bytes"
    )
    model_format = models.CharField(
        max_length=30,
        choices=ModelFormat.choices,
        help_text="Model file format"
    )

    # Architecture
    architecture = models.TextField(
        help_text="Model architecture description"
    )
    framework = models.CharField(
        max_length=30,
        choices=Framework.choices,
        help_text="ML framework used"
    )
    framework_version = models.CharField(
        max_length=20,
        help_text="Framework version"
    )
    model_type = models.CharField(
        max_length=40,
        choices=ModelType.choices,
        help_text="ML task type"
    )

    # Schemas
    input_schema = models.JSONField(
        null=True,
        blank=True,
        help_text="Expected input format specification"
    )
    output_schema = models.JSONField(
        null=True,
        blank=True,
        help_text="Output format specification"
    )

    # Performance
    inference_time_ms = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="Average prediction time (milliseconds)"
    )
    model_size_mb = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="Model memory footprint (megabytes)"
    )

    # Relationships
    produced_by = models.ForeignKey(
        'Experiment',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='produced_models',
        help_text="Experiment that created this model"
    )

    class Meta:
        indexes = [
            models.Index(fields=['framework', 'model_type']),
            models.Index(fields=['model_format']),
        ]
```

#### Dataset

```python
class Dataset(MLAsset):
    """
    Collection of data used for ML training, validation, or testing.
    """
    # File and structure
    file_paths = models.JSONField(
        default=list,
        help_text="List of dataset file locations"
    )
    total_size_bytes = models.BigIntegerField(
        validators=[MinValueValidator(1)],
        help_text="Combined size of all files"
    )
    format = models.CharField(
        max_length=20,
        choices=DatasetFormat.choices,
        help_text="Primary data format"
    )
    schema = models.JSONField(
        null=True,
        blank=True,
        help_text="Data structure definition"
    )

    # Statistics
    num_records = models.BigIntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Total number of data points"
    )
    num_features = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Number of input columns"
    )
    target_column = models.CharField(
        max_length=100,
        blank=True,
        help_text="Column with labels/outcomes"
    )

    # Data analysis
    data_types = models.JSONField(
        default=dict,
        help_text="Column type mapping {col: type}"
    )
    missing_values_count = models.JSONField(
        default=dict,
        help_text="Missing data per column {col: count}"
    )
    categorical_columns = models.JSONField(
        default=list,
        help_text="Non-numeric classification columns"
    )
    numerical_columns = models.JSONField(
        default=list,
        help_text="Numeric data columns"
    )

    # Privacy
    privacy_level = models.CharField(
        max_length=20,
        choices=PrivacyLevel.choices,
        help_text="Data sensitivity level"
    )
    ethical_considerations = models.TextField(
        blank=True,
        help_text="Bias, fairness, and ethical notes"
    )

    # Collection
    collection_method = models.CharField(
        max_length=255,
        blank=True,
        help_text="How data was gathered"
    )
    sampling_strategy = models.CharField(
        max_length=255,
        blank=True,
        help_text="Data sampling approach"
    )

    class Meta:
        indexes = [
            models.Index(fields=['format']),
            models.Index(fields=['privacy_level']),
        ]
```

#### Experiment

```python
class Experiment(MLAsset):
    """
    Complete ML training or evaluation run with full reproducibility information.
    """
    # Classification
    experiment_type = models.CharField(
        max_length=30,
        choices=ExperimentType.choices,
        help_text="Purpose of experiment"
    )
    status = models.CharField(
        max_length=20,
        choices=ExperimentStatus.choices,
        default=ExperimentStatus.PENDING,
        db_index=True,
        help_text="Execution state"
    )

    # Timing
    start_time = models.DateTimeField(
        null=True,
        blank=True,
        db_index=True,
        help_text="When experiment began"
    )
    end_time = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When experiment finished"
    )
    duration_seconds = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Total execution time"
    )

    # Configuration
    config_parameters = models.JSONField(
        default=dict,
        help_text="General experiment settings"
    )
    random_seed = models.IntegerField(
        null=True,
        blank=True,
        help_text="Reproducibility seed value"
    )
    reproducibility_hash = models.CharField(
        max_length=64,
        blank=True,
        help_text="Unique fingerprint for reproduction"
    )

    # Code & Environment
    code_repository_url = models.URLField(
        blank=True,
        help_text="Link to source code"
    )
    code_commit_hash = models.CharField(
        max_length=40,
        blank=True,
        help_text="Exact code version (Git SHA)"
    )
    environment_specification = models.JSONField(
        default=dict,
        help_text="Software dependencies and versions"
    )

    # Relationships
    base_model = models.ForeignKey(
        'Model',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='finetuning_experiments',
        help_text="Base model for fine-tuning (optional)"
    )
    datasets = models.ManyToManyField(
        'Dataset',
        through='ExperimentDataset',
        related_name='experiments'
    )

    class Meta:
        indexes = [
            models.Index(fields=['status', 'start_time']),
            models.Index(fields=['experiment_type']),
        ]
```

---

## Experiment Support Entities

### Hyperparameter

```python
class Hyperparameter(models.Model):
    """
    Individual experiment configuration parameter for efficient querying.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE,
        related_name='hyperparameters'
    )

    parameter_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Parameter name (e.g., learning_rate)"
    )
    parameter_value = models.CharField(
        max_length=500,
        help_text="Parameter value (stored as string)"
    )
    parameter_type = models.CharField(
        max_length=20,
        choices=ParameterType.choices,
        help_text="Data type"
    )

    display_order = models.IntegerField(
        default=0,
        help_text="UI presentation order"
    )
    is_tunable = models.BooleanField(
        default=False,
        help_text="If part of hyperparameter search space"
    )

    class Meta:
        unique_together = [['experiment', 'parameter_name']]
        indexes = [
            models.Index(fields=['parameter_name', 'parameter_value']),
        ]
        ordering = ['display_order', 'parameter_name']

    def __str__(self):
        return f"{self.parameter_name}={self.parameter_value}"
```

### ExperimentMetric

```python
class ExperimentMetric(models.Model):
    """
    Time-series metrics logged during training (loss, accuracy, etc.).
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE,
        related_name='metrics'
    )

    metric_name = models.CharField(
        max_length=100,
        db_index=True,
        help_text="Metric identifier (e.g., train_loss)"
    )
    step = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="Epoch or training step"
    )
    timestamp = models.DateTimeField(
        auto_now_add=True,
        help_text="When metric was recorded"
    )
    value = models.FloatField(
        help_text="Metric value"
    )

    metric_type = models.CharField(
        max_length=20,
        choices=MetricType.choices,
        default=MetricType.CUSTOM,
        help_text="Type of metric"
    )

    class Meta:
        unique_together = [['experiment', 'metric_name', 'step']]
        indexes = [
            models.Index(fields=['experiment', 'metric_name', 'step']),
            models.Index(fields=['experiment', 'timestamp']),
        ]
        ordering = ['step', 'timestamp']

    def __str__(self):
        return f"{self.metric_name}@step{self.step}: {self.value}"
```

### Checkpoint

```python
class Checkpoint(models.Model):
    """
    Saved model states during training for resume and rollback.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE,
        related_name='checkpoints'
    )
    model = models.ForeignKey(
        'Model',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='source_checkpoints',
        help_text="If checkpoint becomes published model"
    )

    checkpoint_name = models.CharField(
        max_length=100,
        help_text="Checkpoint identifier"
    )
    step = models.IntegerField(
        validators=[MinValueValidator(0)],
        help_text="Epoch or training step"
    )
    saved_at = models.DateTimeField(
        auto_now_add=True,
        help_text="When checkpoint was saved"
    )

    file_path = models.CharField(
        max_length=500,
        help_text="Location of checkpoint file"
    )
    file_size_bytes = models.BigIntegerField(
        validators=[MinValueValidator(1)],
        help_text="Checkpoint file size"
    )
    checksum = models.CharField(
        max_length=128,
        help_text="File integrity hash"
    )
    checksum_algorithm = models.CharField(
        max_length=20,
        choices=ChecksumAlgorithm.choices,
        default=ChecksumAlgorithm.SHA256
    )

    metrics_snapshot = models.JSONField(
        default=dict,
        help_text="Metrics at this checkpoint"
    )
    is_best = models.BooleanField(
        default=False,
        db_index=True,
        help_text="Best checkpoint based on metric"
    )
    is_final = models.BooleanField(
        default=False,
        help_text="Final checkpoint (training complete)"
    )
    notes = models.TextField(
        blank=True,
        help_text="Additional notes"
    )

    class Meta:
        indexes = [
            models.Index(fields=['experiment', 'step']),
            models.Index(fields=['experiment', 'is_best']),
        ]
        ordering = ['step']

    def __str__(self):
        return f"{self.checkpoint_name} (step {self.step})"
```

---

## Resource Management

### ComputationalResource (Base with Multi-table Inheritance)

```python
class ComputationalResource(models.Model):
    """
    Base class for computing infrastructure. Specialized via multi-table inheritance.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    name = models.CharField(
        max_length=255,
        help_text="Human-readable resource name"
    )
    resource_type = models.CharField(
        max_length=20,
        choices=ResourceType.choices,
        help_text="Type of resource"
    )
    location = models.CharField(
        max_length=255,
        help_text="Physical or logical location"
    )
    availability = models.CharField(
        max_length=20,
        choices=Availability.choices,
        default=Availability.AVAILABLE,
        help_text="Current status"
    )

    # Common specifications
    cpu_cores = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Number of CPU cores"
    )
    memory_gb = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Total system memory (GB)"
    )
    gpu_count = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of GPU devices"
    )
    gpu_model = models.CharField(
        max_length=100,
        blank=True,
        help_text="GPU model name"
    )

    # Cost
    pricing_model = models.CharField(
        max_length=50,
        help_text="How costs are calculated"
    )
    cost_per_hour = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Hourly usage cost"
    )
    currency = models.CharField(
        max_length=3,
        default='USD',
        help_text="Currency code (ISO 4217)"
    )

    # Environmental
    energy_efficiency = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)],
        help_text="Power usage effectiveness (0-1)"
    )
    carbon_intensity = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="kg CO2 per kWh"
    )

    # Relationships
    organizations = models.ManyToManyField(
        'Organization',
        through='ResourceAllocation',
        related_name='computational_resources'
    )

    class Meta:
        indexes = [
            models.Index(fields=['resource_type', 'availability']),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_resource_type_display()})"
```

### LocalResource

```python
class LocalResource(ComputationalResource):
    """
    On-premises hardware resources (workstations, servers, local clusters).
    """
    hostname = models.CharField(
        max_length=255,
        unique=True,
        help_text="Network hostname"
    )
    mac_addresses = models.JSONField(
        default=list,
        help_text="Network interface MAC addresses"
    )
    local_ip_address = models.GenericIPAddressField(
        null=True,
        blank=True,
        help_text="Local network IP"
    )

    power_consumption_watts = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(1)],
        help_text="Typical power consumption"
    )
    cooling_type = models.CharField(
        max_length=20,
        choices=CoolingType.choices,
        help_text="Cooling method"
    )
    chassis_type = models.CharField(
        max_length=20,
        choices=ChassisType.choices,
        help_text="Physical form factor"
    )
    rack_location = models.CharField(
        max_length=100,
        blank=True,
        help_text="Physical rack position"
    )
    asset_tag = models.CharField(
        max_length=50,
        blank=True,
        help_text="Organizational asset ID"
    )

    def __str__(self):
        return f"Local: {self.hostname}"
```

### CloudResource

```python
class CloudResource(ComputationalResource):
    """
    Virtual resources provided by cloud service providers.
    """
    cloud_provider = models.CharField(
        max_length=50,
        choices=CloudProvider.choices,
        help_text="Cloud provider"
    )
    provider_region = models.CharField(
        max_length=50,
        help_text="Geographic region"
    )
    availability_zone = models.CharField(
        max_length=50,
        blank=True,
        help_text="Specific availability zone"
    )
    account_id = models.CharField(
        max_length=100,
        blank=True,
        help_text="Cloud account ID"
    )

    instance_type = models.CharField(
        max_length=100,
        help_text="Provider-specific instance type"
    )
    instance_family = models.CharField(
        max_length=50,
        help_text="Instance category"
    )
    virtualization_type = models.CharField(
        max_length=20,
        choices=VirtualizationType.choices,
        help_text="Virtualization technology"
    )
    tenancy = models.CharField(
        max_length=20,
        choices=Tenancy.choices,
        help_text="Tenancy model"
    )

    pricing_model = models.CharField(
        max_length=20,
        choices=CloudPricing.choices,
        help_text="Pricing strategy"
    )
    spot_price = models.DecimalField(
        max_digits=10,
        decimal_places=4,
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Spot instance price"
    )

    auto_scaling_enabled = models.BooleanField(
        default=False,
        help_text="Auto-scaling active"
    )
    min_instances = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1)],
        help_text="Min instance count"
    )
    max_instances = models.IntegerField(
        default=1,
        validators=[MinValueValidator(1)],
        help_text="Max instance count"
    )

    class Meta:
        indexes = [
            models.Index(fields=['cloud_provider', 'provider_region']),
            models.Index(fields=['instance_type']),
        ]

    def __str__(self):
        return f"{self.get_cloud_provider_display()}: {self.instance_type}"
```

### HPCResource

```python
class HPCResource(ComputationalResource):
    """
    High-performance computing cluster resources with job scheduling.
    """
    cluster_name = models.CharField(
        max_length=100,
        help_text="HPC cluster identifier"
    )
    total_nodes = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Total compute nodes"
    )
    node_configuration = models.JSONField(
        default=dict,
        help_text="Per-node hardware specification"
    )

    scheduler_type = models.CharField(
        max_length=20,
        choices=SchedulerType.choices,
        help_text="Job scheduler"
    )
    scheduler_version = models.CharField(
        max_length=50,
        help_text="Scheduler software version"
    )
    partition_names = models.JSONField(
        default=list,
        help_text="Available job partitions"
    )
    queue_limits = models.JSONField(
        default=dict,
        help_text="Per-queue resource limits"
    )

    allocation_policy = models.CharField(
        max_length=20,
        choices=AllocationPolicy.choices,
        help_text="Node allocation strategy"
    )
    max_job_duration = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Maximum job runtime (hours)"
    )

    interconnect_type = models.CharField(
        max_length=20,
        choices=InterconnectType.choices,
        help_text="Network interconnect type"
    )
    parallel_filesystem = models.CharField(
        max_length=20,
        choices=ParallelFilesystem.choices,
        help_text="Shared parallel filesystem"
    )

    class Meta:
        indexes = [
            models.Index(fields=['cluster_name']),
            models.Index(fields=['scheduler_type']),
        ]

    def __str__(self):
        return f"HPC: {self.cluster_name}"
```

### ResourceUtilization (Consolidated)

```python
class ResourceUtilization(models.Model):
    """
    Tracks actual resource consumption during experiments.
    Consolidates usage metrics, cost tracking, and environmental impact.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    resource = models.ForeignKey(
        'ComputationalResource',
        on_delete=models.CASCADE,
        related_name='utilizations'
    )
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE,
        related_name='resource_utilizations'
    )

    # Timing
    start_time = models.DateTimeField(
        help_text="When usage began"
    )
    end_time = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When usage ended"
    )
    duration_seconds = models.IntegerField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0)],
        help_text="Total usage time"
    )

    # Usage metrics
    cpu_utilization_percent = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Average CPU usage"
    )
    gpu_utilization_percent = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Average GPU usage"
    )
    memory_usage_gb = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Peak memory consumption"
    )
    storage_usage_gb = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Disk space consumed"
    )
    network_io_gb = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Data transfer volume"
    )

    # Cost tracking (integrated)
    compute_cost = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Processing costs"
    )
    storage_cost = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Data storage costs"
    )
    network_cost = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        default=0.0,
        validators=[MinValueValidator(0)],
        help_text="Data transfer costs"
    )
    total_cost = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        validators=[MinValueValidator(0)],
        help_text="Sum of all costs"
    )
    currency = models.CharField(
        max_length=3,
        help_text="Currency code (ISO 4217)"
    )

    # Environmental impact (integrated)
    carbon_footprint_kg = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="CO2 equivalent emissions"
    )
    carbon_footprint_lower = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="Lower bound (uncertainty)"
    )
    carbon_footprint_upper = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="Upper bound (uncertainty)"
    )
    energy_consumption_kwh = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Total energy used"
    )
    water_usage_liters = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(0.0)],
        help_text="Cooling water consumed"
    )
    carbon_intensity_location = models.CharField(
        max_length=100,
        help_text="Geographic location for calculations"
    )
    carbon_intensity_value = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="kg CO2/kWh for location"
    )

    # Analytics
    peak_usage_metrics = models.JSONField(
        default=dict,
        help_text="Maximum resource consumption"
    )
    average_usage_metrics = models.JSONField(
        default=dict,
        help_text="Mean resource consumption"
    )

    class Meta:
        indexes = [
            models.Index(fields=['experiment', 'start_time']),
            models.Index(fields=['resource', 'start_time']),
        ]

    def __str__(self):
        return f"{self.experiment.name} on {self.resource.name}"
```

---

## Research Context

### Researcher

```python
class Researcher(models.Model):
    """
    Individuals who create and contribute to ML assets.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    first_name = models.CharField(
        max_length=100,
        help_text="Given name"
    )
    last_name = models.CharField(
        max_length=100,
        help_text="Family name"
    )
    email = models.EmailField(
        unique=True,
        help_text="Contact email address"
    )
    orcid_id = models.CharField(
        max_length=19,
        unique=True,
        null=True,
        blank=True,
        help_text="Persistent researcher ID (ORCID)"
    )

    expertise_areas = models.JSONField(
        default=list,
        help_text="Research specializations"
    )
    research_interests = models.JSONField(
        default=list,
        help_text="Current focus areas"
    )

    organizations = models.ManyToManyField(
        'Organization',
        through='OrganizationAffiliation',
        related_name='researchers'
    )

    def __str__(self):
        return f"{self.first_name} {self.last_name}"

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"
```

### Organization

```python
class Organization(models.Model):
    """
    Institutions that support ML research and provide computational resources.
    """
    id = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False
    )
    name = models.CharField(
        max_length=255,
        unique=True,
        help_text="Official organization name"
    )
    type = models.CharField(
        max_length=50,
        choices=OrganizationType.choices,
        help_text="Organization category"
    )
    location = models.CharField(
        max_length=255,
        help_text="Geographic location"
    )
    website = models.URLField(
        blank=True,
        help_text="Official website"
    )

    policies = models.JSONField(
        default=list,
        help_text="Research and usage policies"
    )

    class Meta:
        indexes = [
            models.Index(fields=['type']),
        ]

    def __str__(self):
        return self.name
```

---

## Association Classes

### ExperimentDataset

```python
class ExperimentDataset(models.Model):
    """
    Captures relationship between experiments and datasets with split definition.
    """
    experiment = models.ForeignKey(
        'Experiment',
        on_delete=models.CASCADE
    )
    dataset = models.ForeignKey(
        'Dataset',
        on_delete=models.CASCADE
    )

    # Role
    role = models.CharField(
        max_length=20,
        choices=TrainingRole.choices,
        help_text="Usage role (training/validation/testing)"
    )

    # Split definition
    split_percentage = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Percentage of dataset used"
    )
    num_records = models.IntegerField(
        validators=[MinValueValidator(1)],
        help_text="Number of records in split"
    )
    indices_file_path = models.CharField(
        max_length=500,
        blank=True,
        help_text="File with record indices/IDs"
    )
    random_seed = models.IntegerField(
        help_text="Seed for split generation (reproducibility)"
    )

    # Preprocessing
    preprocessing_applied = models.JSONField(
        default=list,
        help_text="Transformations applied"
    )
    augmentation_enabled = models.BooleanField(
        default=False,
        help_text="Whether augmentation used"
    )

    class Meta:
        unique_together = [['experiment', 'dataset', 'role']]

    def __str__(self):
        return f"{self.experiment.name} - {self.dataset.name} ({self.role})"
```

### OrganizationAffiliation

```python
class OrganizationAffiliation(models.Model):
    """
    Represents researcher-organization relationships over time.
    """
    researcher = models.ForeignKey(
        'Researcher',
        on_delete=models.CASCADE
    )
    organization = models.ForeignKey(
        'Organization',
        on_delete=models.CASCADE
    )

    role = models.CharField(
        max_length=100,
        help_text="Position/role within organization"
    )
    start_date = models.DateField(
        help_text="Beginning of affiliation"
    )
    end_date = models.DateField(
        null=True,
        blank=True,
        help_text="End of affiliation (optional)"
    )
    is_primary = models.BooleanField(
        default=False,
        help_text="Primary affiliation flag"
    )

    class Meta:
        unique_together = [['researcher', 'organization', 'start_date']]

    def __str__(self):
        return f"{self.researcher.full_name} @ {self.organization.name}"
```

### ResourceAllocation

```python
class ResourceAllocation(models.Model):
    """
    Represents shared resource allocation from organizations.
    """
    organization = models.ForeignKey(
        'Organization',
        on_delete=models.CASCADE
    )
    resource = models.ForeignKey(
        'ComputationalResource',
        on_delete=models.CASCADE
    )

    quota_hours = models.FloatField(
        validators=[MinValueValidator(0.0)],
        help_text="Allocated hours"
    )
    cost_allocation_percent = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Cost share percentage"
    )
    access_priority = models.IntegerField(
        default=0,
        help_text="Access priority level"
    )

    start_date = models.DateField(
        help_text="Allocation start date"
    )
    end_date = models.DateField(
        null=True,
        blank=True,
        help_text="Allocation end date (optional)"
    )

    class Meta:
        unique_together = [['organization', 'resource', 'start_date']]

    def __str__(self):
        return f"{self.organization.name} - {self.resource.name}"
```


---

*See [conceptual-model.md](conceptual-model.md) for business context and [logical-model.md](logical-model.md) for detailed entity specifications.*
