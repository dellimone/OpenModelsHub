from django.contrib import admin
from .models import (
    Model, Dataset, Experiment,
    Hyperparameter, ExperimentMetric, Checkpoint, ExperimentDataset
)


class ExperimentDatasetInline(admin.TabularInline):
    """Inline admin for experiment-dataset relationships"""
    model = ExperimentDataset
    extra = 1
    fields = ['dataset', 'role', 'split_percentage', 'num_records', 'augmentation_enabled']


class HyperparameterInline(admin.TabularInline):
    """Inline admin for hyperparameters"""
    model = Hyperparameter
    extra = 1
    fields = ['parameter_name', 'parameter_value', 'parameter_type', 'is_tunable']


@admin.register(Model)
class ModelAdmin(admin.ModelAdmin):
    """Admin interface for Model"""
    list_display = ['name', 'version', 'model_type', 'framework', 'created_by', 'created_at']
    list_filter = ['model_type', 'framework', 'access_rights', 'created_at']
    search_fields = ['name', 'description', 'persistent_identifier']
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Identity', {
            'fields': ('id', 'persistent_identifier', 'name', 'description', 'version')
        }),
        ('Classification', {
            'fields': ('model_type', 'framework', 'framework_version', 'model_format')
        }),
        ('Files', {
            'fields': ('model_file_path', 'model_file_size')
        }),
        ('Architecture', {
            'fields': ('architecture', 'input_schema', 'output_schema')
        }),
        ('Performance', {
            'fields': ('inference_time_ms', 'model_size_mb')
        }),
        ('Relationships', {
            'fields': ('produced_by',)
        }),
        ('Ownership', {
            'fields': ('created_by', 'organization', 'license', 'access_rights')
        }),
        ('Metadata', {
            'fields': ('subjects', 'checksum', 'checksum_algorithm'),
            'classes': ('collapse',)
        }),
        ('Versioning', {
            'fields': ('parent_version', 'version_notes'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    """Admin interface for Dataset"""
    list_display = ['name', 'version', 'format', 'num_records', 'privacy_level', 'created_by', 'created_at']
    list_filter = ['format', 'privacy_level', 'access_rights', 'created_at']
    search_fields = ['name', 'description', 'persistent_identifier']
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Identity', {
            'fields': ('id', 'persistent_identifier', 'name', 'description', 'version')
        }),
        ('Files and Format', {
            'fields': ('file_paths', 'total_size_bytes', 'format', 'schema')
        }),
        ('Statistics', {
            'fields': ('num_records', 'num_features', 'target_column')
        }),
        ('Data Analysis', {
            'fields': ('data_types', 'missing_values_count', 'categorical_columns', 'numerical_columns'),
            'classes': ('collapse',)
        }),
        ('Privacy and Ethics', {
            'fields': ('privacy_level', 'ethical_considerations')
        }),
        ('Collection', {
            'fields': ('collection_method', 'sampling_strategy')
        }),
        ('Ownership', {
            'fields': ('created_by', 'organization', 'license', 'access_rights')
        }),
        ('Metadata', {
            'fields': ('subjects', 'checksum', 'checksum_algorithm'),
            'classes': ('collapse',)
        }),
        ('Versioning', {
            'fields': ('parent_version', 'version_notes'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )


@admin.register(Experiment)
class ExperimentAdmin(admin.ModelAdmin):
    """Admin interface for Experiment"""
    list_display = ['name', 'version', 'experiment_type', 'status', 'start_time', 'created_by']
    list_filter = ['experiment_type', 'status', 'created_at']
    search_fields = ['name', 'description', 'persistent_identifier']
    readonly_fields = ['id', 'created_at', 'updated_at']
    date_hierarchy = 'start_time'

    fieldsets = (
        ('Identity', {
            'fields': ('id', 'persistent_identifier', 'name', 'description', 'version')
        }),
        ('Classification', {
            'fields': ('experiment_type', 'status')
        }),
        ('Timing', {
            'fields': ('start_time', 'end_time', 'duration_seconds')
        }),
        ('Configuration', {
            'fields': ('config_parameters', 'random_seed', 'reproducibility_hash')
        }),
        ('Code Tracking', {
            'fields': ('code_repository_url', 'code_commit_hash', 'environment_specification')
        }),
        ('Relationships', {
            'fields': ('base_model',)
        }),
        ('Ownership', {
            'fields': ('created_by', 'organization', 'license', 'access_rights')
        }),
        ('Metadata', {
            'fields': ('subjects', 'checksum', 'checksum_algorithm'),
            'classes': ('collapse',)
        }),
        ('Versioning', {
            'fields': ('parent_version', 'version_notes'),
            'classes': ('collapse',)
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    inlines = [ExperimentDatasetInline, HyperparameterInline]


@admin.register(Hyperparameter)
class HyperparameterAdmin(admin.ModelAdmin):
    """Admin interface for Hyperparameter"""
    list_display = ['experiment', 'parameter_name', 'parameter_value', 'parameter_type', 'is_tunable']
    list_filter = ['parameter_type', 'is_tunable']
    search_fields = ['experiment__name', 'parameter_name']


@admin.register(ExperimentMetric)
class ExperimentMetricAdmin(admin.ModelAdmin):
    """Admin interface for ExperimentMetric"""
    list_display = ['experiment', 'metric_name', 'step', 'value', 'metric_type', 'timestamp']
    list_filter = ['metric_type', 'timestamp']
    search_fields = ['experiment__name', 'metric_name']
    date_hierarchy = 'timestamp'


@admin.register(Checkpoint)
class CheckpointAdmin(admin.ModelAdmin):
    """Admin interface for Checkpoint"""
    list_display = ['experiment', 'checkpoint_name', 'step', 'is_best', 'is_final', 'saved_at']
    list_filter = ['is_best', 'is_final', 'saved_at']
    search_fields = ['experiment__name', 'checkpoint_name']
    readonly_fields = ['saved_at']
    date_hierarchy = 'saved_at'


@admin.register(ExperimentDataset)
class ExperimentDatasetAdmin(admin.ModelAdmin):
    """Admin interface for ExperimentDataset association"""
    list_display = ['experiment', 'dataset', 'role', 'split_percentage', 'num_records', 'augmentation_enabled']
    list_filter = ['role', 'augmentation_enabled']
    search_fields = ['experiment__name', 'dataset__name']
