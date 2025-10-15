
import uuid
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

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

class TrainingRole(models.TextChoices):
    TRAINING = 'TRAINING', 'Training'
    VALIDATION = 'VALIDATION', 'Validation'
    TESTING = 'TESTING', 'Testing'
    HOLDOUT = 'HOLDOUT', 'Holdout'

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
        'research.Researcher',
        on_delete=models.PROTECT,
        related_name='%(class)s_created',
        help_text="Asset creator"
    )
    organization = models.ForeignKey(
        'research.Organization',
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
