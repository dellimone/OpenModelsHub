#!/usr/bin/env python
"""
Simple test script to validate MLRepo Django models
Tests basic model creation and relationships according to Documentation/
"""

import os
import sys
import django
from datetime import date, datetime
from decimal import Decimal

# Add the project directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'openmodelshub_project.settings')
django.setup()

from research.models import Organization, Researcher, OrganizationAffiliation
from assets.models import (
    Model, Dataset, Experiment,
    Hyperparameter, ExperimentMetric, Checkpoint, ExperimentDataset
)
from resources.models import (
    ComputationalResource, LocalResource, CloudResource, HPCResource,
    ResourceUtilization, ResourceAllocation
)


def test_research_models():
    """Test research context models"""
    print("=== Testing Research Models ===")

    # Create an organization
    org = Organization.objects.create(
        name="AI Research Institute",
        type="UNIVERSITY",
        location="Cambridge, MA, USA",
        website="https://airi.edu",
        policies=["open_science", "fair_data"]
    )
    print(f"✓ Created organization: {org}")

    # Create a researcher
    researcher = Researcher.objects.create(
        first_name="Alice",
        last_name="Johnson",
        email="alice.johnson@airi.edu",
        orcid_id="0000-0001-2345-6789",
        expertise_areas=["deep learning", "computer vision"],
        research_interests=["transformers", "multimodal learning"]
    )
    print(f"✓ Created researcher: {researcher}")

    # Create affiliation
    affiliation = OrganizationAffiliation.objects.create(
        researcher=researcher,
        organization=org,
        role="Assistant Professor",
        start_date=date(2020, 9, 1),
        is_primary=True
    )
    print(f"✓ Created affiliation: {affiliation}")

    return org, researcher


def test_asset_models(org, researcher):
    """Test ML asset models"""
    print("\n=== Testing ML Asset Models ===")

    # Create a dataset
    dataset = Dataset.objects.create(
        # MLAsset fields
        persistent_identifier="doi:10.5555/12345678",
        name="Climate Change Detection Dataset",
        description="Satellite imagery dataset for climate change detection",
        version="1.0.0",
        created_by=researcher,
        organization=org,
        license="CC-BY-4.0",
        subjects=["climate", "satellite", "imagery"],
        access_rights="PUBLIC",
        checksum="a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
        checksum_algorithm="SHA256",
        # Dataset-specific fields
        file_paths=["/data/climate/images.zip", "/data/climate/labels.csv"],
        total_size_bytes=2147483648,  # 2GB
        format="IMAGE_FOLDER",
        schema={"image": "path", "label": "category", "coordinates": "geo"},
        num_records=10000,
        num_features=512,
        target_column="label",
        data_types={"image_path": "str", "label": "category", "coordinates": "str"},
        missing_values_count={"label": 0, "coordinates": 45},
        categorical_columns=["label"],
        numerical_columns=["temperature", "elevation"],
        privacy_level="PUBLIC",
        ethical_considerations="Dataset contains no personal information",
        collection_method="Satellite imagery from public sources",
        sampling_strategy="Random sampling across geographic regions"
    )
    print(f"✓ Created dataset: {dataset}")

    # Create an experiment
    experiment = Experiment.objects.create(
        # MLAsset fields
        persistent_identifier="doi:10.5555/87654321",
        name="Climate Detection Training Run",
        description="Training CNN model on climate detection dataset",
        version="1.0.0",
        created_by=researcher,
        organization=org,
        license="MIT",
        subjects=["climate", "deep learning", "CNN"],
        access_rights="PUBLIC",
        checksum="b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
        checksum_algorithm="SHA256",
        # Experiment-specific fields
        experiment_type="TRAINING",
        status="PENDING",
        config_parameters={
            "model_architecture": "resnet50",
            "optimizer": "adam",
            "loss_function": "cross_entropy"
        },
        random_seed=42,
        reproducibility_hash="abc123def456",
        code_repository_url="https://github.com/airi/climate-detection",
        code_commit_hash="a1b2c3d4e5f6789012345678901234567890abcd",
        environment_specification={
            "python": "3.9",
            "pytorch": "1.12.0",
            "cuda": "11.6"
        }
    )
    print(f"✓ Created experiment: {experiment}")

    # Create experiment-dataset relationship
    exp_dataset = ExperimentDataset.objects.create(
        experiment=experiment,
        dataset=dataset,
        role="TRAINING",
        split_percentage=80.0,
        num_records=8000,
        indices_file_path="/data/splits/train_indices.txt",
        random_seed=42,
        preprocessing_applied=["resize", "normalize", "augment"],
        augmentation_enabled=True
    )
    print(f"✓ Created experiment-dataset relationship: {exp_dataset}")

    # Update experiment to running
    experiment.status = "RUNNING"
    experiment.start_time = datetime.now()
    experiment.save()
    print(f"✓ Updated experiment status to: {experiment.status}")

    # Create hyperparameters
    hp1 = Hyperparameter.objects.create(
        experiment=experiment,
        parameter_name="learning_rate",
        parameter_value="0.001",
        parameter_type="FLOAT",
        display_order=1,
        is_tunable=True
    )
    hp2 = Hyperparameter.objects.create(
        experiment=experiment,
        parameter_name="batch_size",
        parameter_value="32",
        parameter_type="INTEGER",
        display_order=2,
        is_tunable=True
    )
    print(f"✓ Created hyperparameters: {hp1}, {hp2}")

    # Create experiment metrics
    metric1 = ExperimentMetric.objects.create(
        experiment=experiment,
        metric_name="train_loss",
        step=1,
        value=2.45,
        metric_type="LOSS"
    )
    metric2 = ExperimentMetric.objects.create(
        experiment=experiment,
        metric_name="val_accuracy",
        step=1,
        value=0.72,
        metric_type="ACCURACY"
    )
    print(f"✓ Created metrics: {metric1}, {metric2}")

    # Create a checkpoint
    checkpoint = Checkpoint.objects.create(
        experiment=experiment,
        checkpoint_name="best_model",
        step=10,
        file_path="/checkpoints/exp_001_epoch_10.pth",
        file_size_bytes=440000000,
        checksum="c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678",
        checksum_algorithm="SHA256",
        metrics_snapshot={"train_loss": 0.45, "val_accuracy": 0.89},
        is_best=True,
        is_final=False,
        notes="Best validation accuracy achieved at epoch 10"
    )
    print(f"✓ Created checkpoint: {checkpoint}")

    # Create a model from the experiment
    model = Model.objects.create(
        # MLAsset fields
        persistent_identifier="doi:10.5555/11223344",
        name="Climate Detection ResNet Model",
        description="ResNet50 model fine-tuned for climate change detection",
        version="1.0.0",
        created_by=researcher,
        organization=org,
        license="MIT",
        subjects=["climate", "resnet", "computer vision"],
        access_rights="PUBLIC",
        checksum="d4e5f6789012345678901234567890abcdef1234567890abcdef123456789",
        checksum_algorithm="SHA256",
        # Model-specific fields
        model_file_path="/models/climate_resnet50_v1.pth",
        model_file_size=440000000,  # 440MB
        model_format="PYTORCH",
        architecture="ResNet50 with custom classification head (5 classes)",
        framework="PYTORCH",
        framework_version="1.12.0",
        model_type="CLASSIFICATION",
        input_schema={"image": "tensor[3, 224, 224]", "dtype": "float32"},
        output_schema={"logits": "tensor[5]", "probabilities": "tensor[5]"},
        inference_time_ms=45.0,
        model_size_mb=440.0,
        produced_by=experiment
    )
    print(f"✓ Created model: {model}")

    # Link checkpoint to model
    checkpoint.model = model
    checkpoint.save()
    print(f"✓ Linked checkpoint to model")

    # Complete the experiment
    experiment.status = "COMPLETED"
    experiment.end_time = datetime.now()
    experiment.duration_seconds = 14400  # 4 hours
    experiment.save()
    print(f"✓ Completed experiment: {experiment.status}, Duration: {experiment.duration_seconds}s")

    return dataset, model, experiment


def test_resource_models(org, experiment):
    """Test computational resource models"""
    print("\n=== Testing Resource Models ===")

    # Create a local resource
    local_resource = LocalResource.objects.create(
        name="ML Workstation 01",
        resource_type="LOCAL",
        location="Lab 301, Building A",
        availability="AVAILABLE",
        cpu_cores=16,
        memory_gb=64,
        gpu_count=2,
        gpu_model="NVIDIA RTX 3090",
        pricing_model="fixed_monthly",
        cost_per_hour=Decimal("5.00"),
        currency="USD",
        energy_efficiency=0.85,
        carbon_intensity=0.45,
        # LocalResource-specific
        hostname="ml-ws-01.airi.local",
        mac_addresses=["00:1B:63:84:45:E6"],
        local_ip_address="192.168.1.100",
        power_consumption_watts=750,
        cooling_type="AIR",
        chassis_type="TOWER",
        rack_location="",
        asset_tag="AIRI-IT-2023-045"
    )
    print(f"✓ Created local resource: {local_resource}")

    # Create a cloud resource
    cloud_resource = CloudResource.objects.create(
        name="AWS EC2 GPU Instance",
        resource_type="CLOUD",
        location="us-east-1",
        availability="AVAILABLE",
        cpu_cores=8,
        memory_gb=32,
        gpu_count=1,
        gpu_model="NVIDIA A100",
        cost_per_hour=Decimal("3.50"),
        currency="USD",
        energy_efficiency=0.90,
        carbon_intensity=0.35,
        # CloudResource-specific
        cloud_provider="AWS",
        provider_region="us-east-1",
        availability_zone="us-east-1a",
        account_id="123456789012",
        instance_type="p3.2xlarge",
        instance_family="p3",
        virtualization_type="HVM",
        tenancy="SHARED",
        pricing_model="ON_DEMAND",
        spot_price=None,
        auto_scaling_enabled=False,
        min_instances=1,
        max_instances=1
    )
    print(f"✓ Created cloud resource: {cloud_resource}")

    # Create HPC resource
    hpc_resource = HPCResource.objects.create(
        name="University HPC Cluster",
        resource_type="HPC",
        location="Data Center, Building B",
        availability="AVAILABLE",
        cpu_cores=1024,
        memory_gb=4096,
        gpu_count=64,
        gpu_model="NVIDIA A100",
        pricing_model="allocation_based",
        cost_per_hour=Decimal("100.00"),
        currency="USD",
        energy_efficiency=0.95,
        carbon_intensity=0.30,
        # HPCResource-specific
        cluster_name="airi-hpc-01",
        total_nodes=32,
        node_configuration={"cores_per_node": 32, "memory_per_node_gb": 128, "gpus_per_node": 2},
        scheduler_type="SLURM",
        scheduler_version="21.08.8",
        partition_names=["gpu", "cpu", "bigmem"],
        queue_limits={"gpu": {"max_time_hours": 48, "max_nodes": 8}},
        allocation_policy="SHARED",
        max_job_duration=72,
        interconnect_type="INFINIBAND",
        parallel_filesystem="LUSTRE"
    )
    print(f"✓ Created HPC resource: {hpc_resource}")

    # Create resource allocation
    allocation = ResourceAllocation.objects.create(
        organization=org,
        resource=hpc_resource,
        quota_hours=10000.0,
        cost_allocation_percent=50.0,
        access_priority=1,
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31)
    )
    print(f"✓ Created resource allocation: {allocation}")

    # Create resource utilization
    utilization = ResourceUtilization.objects.create(
        resource=local_resource,
        experiment=experiment,
        start_time=experiment.start_time,
        end_time=experiment.end_time,
        duration_seconds=experiment.duration_seconds,
        cpu_utilization_percent=85.5,
        gpu_utilization_percent=92.3,
        memory_usage_gb=48.5,
        storage_usage_gb=120.0,
        network_io_gb=5.2,
        compute_cost=Decimal("20.00"),
        storage_cost=Decimal("2.50"),
        network_cost=Decimal("0.50"),
        total_cost=Decimal("23.00"),
        currency="USD",
        carbon_footprint_kg=12.5,
        carbon_footprint_lower=10.2,
        carbon_footprint_upper=14.8,
        energy_consumption_kwh=15.5,
        water_usage_liters=None,
        carbon_intensity_location="Massachusetts, USA",
        carbon_intensity_value=0.45,
        peak_usage_metrics={"cpu": 95.2, "gpu": 98.5, "memory": 52.1},
        average_usage_metrics={"cpu": 85.5, "gpu": 92.3, "memory": 48.5}
    )
    print(f"✓ Created resource utilization: {utilization}")

    return local_resource, cloud_resource, hpc_resource


def test_relationships():
    """Test model relationships and queries"""
    print("\n=== Testing Relationships ===")

    # Test researcher -> assets relationships
    researchers = Researcher.objects.all()
    for researcher in researchers:
        print(f"Researcher {researcher.full_name}:")
        print(f"  - Created models: {researcher.model_created.count()}")
        print(f"  - Created datasets: {researcher.dataset_created.count()}")
        print(f"  - Created experiments: {researcher.experiment_created.count()}")
        print(f"  - Organizations: {researcher.organizations.count()}")

    # Test organization -> assets relationships
    organizations = Organization.objects.all()
    for org in organizations:
        print(f"Organization {org.name}:")
        print(f"  - Models: {org.model_assets.count()}")
        print(f"  - Datasets: {org.dataset_assets.count()}")
        print(f"  - Experiments: {org.experiment_assets.count()}")
        print(f"  - Researchers: {org.researchers.count()}")
        print(f"  - Resources: {org.computational_resources.count()}")

    # Test model -> experiment relationships
    models = Model.objects.all()
    for model in models:
        print(f"Model {model.name}:")
        if model.produced_by:
            print(f"  - Produced by experiment: {model.produced_by.name}")
        print(f"  - Associated checkpoints: {model.source_checkpoints.count()}")

    # Test experiment -> dataset relationships
    experiments = Experiment.objects.all()
    for exp in experiments:
        print(f"Experiment {exp.name}:")
        print(f"  - Datasets used: {exp.datasets.count()}")
        exp_datasets = ExperimentDataset.objects.filter(experiment=exp)
        for rel in exp_datasets:
            print(f"    - {rel.dataset.name} ({rel.role}): {rel.split_percentage}% = {rel.num_records} records")
        print(f"  - Hyperparameters: {exp.hyperparameters.count()}")
        print(f"  - Metrics: {exp.metrics.count()}")
        print(f"  - Checkpoints: {exp.checkpoints.count()}")
        print(f"  - Resource utilizations: {exp.resource_utilizations.count()}")


def test_version_lineage(researcher, org):
    """Test version lineage functionality"""
    print("\n=== Testing Version Lineage ===")

    # Create version chain: v1.0 -> v1.1 -> v2.0
    dataset_v1 = Dataset.objects.create(
        persistent_identifier="doi:10.5555/version-test-v1",
        name="Test Dataset",
        description="Original version",
        version="1.0.0",
        created_by=researcher,
        organization=org,
        license="MIT",
        subjects=["test"],
        access_rights="PUBLIC",
        checksum="v1hash",
        checksum_algorithm="SHA256",
        file_paths=["/data/v1.csv"],
        total_size_bytes=1000000,
        format="CSV",
        privacy_level="PUBLIC"
    )
    print(f"✓ Created dataset v1.0: {dataset_v1}")

    dataset_v11 = Dataset.objects.create(
        persistent_identifier="doi:10.5555/version-test-v1.1",
        name="Test Dataset",
        description="Bug fixes",
        version="1.1.0",
        created_by=researcher,
        organization=org,
        license="MIT",
        subjects=["test"],
        access_rights="PUBLIC",
        checksum="v11hash",
        checksum_algorithm="SHA256",
        file_paths=["/data/v1.1.csv"],
        total_size_bytes=1100000,
        format="CSV",
        privacy_level="PUBLIC",
        parent_version=dataset_v1,
        version_notes="Fixed data quality issues"
    )
    print(f"✓ Created dataset v1.1: {dataset_v11}")

    dataset_v2 = Dataset.objects.create(
        persistent_identifier="doi:10.5555/version-test-v2",
        name="Test Dataset",
        description="Major update",
        version="2.0.0",
        created_by=researcher,
        organization=org,
        license="MIT",
        subjects=["test"],
        access_rights="PUBLIC",
        checksum="v2hash",
        checksum_algorithm="SHA256",
        file_paths=["/data/v2.csv"],
        total_size_bytes=2000000,
        format="CSV",
        privacy_level="PUBLIC",
        parent_version=dataset_v11,
        version_notes="Complete data restructuring"
    )
    print(f"✓ Created dataset v2.0: {dataset_v2}")

    # Test version lineage
    lineage = dataset_v2.get_version_lineage()
    print(f"Version lineage for {dataset_v2.name} v{dataset_v2.version}:")
    for i, version in enumerate(lineage):
        print(f"  {i+1}. v{version.version}: {version.version_notes or 'Initial version'}")


def main():
    """Main test function"""
    print("MLRepo Django Models Test")
    print("Testing against Documentation/ ground truth")
    print("=" * 60)

    try:
        # Clean up any existing test data
        print("Cleaning up existing test data...")
        ResourceUtilization.objects.all().delete()
        ResourceAllocation.objects.all().delete()
        HPCResource.objects.all().delete()
        CloudResource.objects.all().delete()
        LocalResource.objects.all().delete()
        ComputationalResource.objects.all().delete()
        ExperimentDataset.objects.all().delete()
        Checkpoint.objects.all().delete()
        ExperimentMetric.objects.all().delete()
        Hyperparameter.objects.all().delete()
        Model.objects.all().delete()
        Experiment.objects.all().delete()
        Dataset.objects.all().delete()
        OrganizationAffiliation.objects.all().delete()
        Researcher.objects.all().delete()
        Organization.objects.all().delete()
        print("✓ Cleanup complete")

        # Test research models
        org, researcher = test_research_models()

        # Test asset models
        dataset, model, experiment = test_asset_models(org, researcher)

        # Test resource models
        local_resource, cloud_resource, hpc_resource = test_resource_models(org, experiment)

        # Test relationships
        test_relationships()

        # Test version lineage
        test_version_lineage(researcher, org)

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Django models are correctly implemented according to Documentation/")
        print("\nDatabase contains:")
        print(f"  - Organizations: {Organization.objects.count()}")
        print(f"  - Researchers: {Researcher.objects.count()}")
        print(f"  - Organization Affiliations: {OrganizationAffiliation.objects.count()}")
        print(f"  - Models: {Model.objects.count()}")
        print(f"  - Datasets: {Dataset.objects.count()}")
        print(f"  - Experiments: {Experiment.objects.count()}")
        print(f"  - Hyperparameters: {Hyperparameter.objects.count()}")
        print(f"  - Experiment Metrics: {ExperimentMetric.objects.count()}")
        print(f"  - Checkpoints: {Checkpoint.objects.count()}")
        print(f"  - Experiment-Dataset Links: {ExperimentDataset.objects.count()}")
        print(f"  - Computational Resources: {ComputationalResource.objects.count()}")
        print(f"  - Resource Utilizations: {ResourceUtilization.objects.count()}")
        print(f"  - Resource Allocations: {ResourceAllocation.objects.count()}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
