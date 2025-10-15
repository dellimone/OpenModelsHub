import uuid
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

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
        'research.Organization',
        through='ResourceAllocation',
        related_name='computational_resources'
    )

    class Meta:
        indexes = [
            models.Index(fields=['resource_type', 'availability']),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_resource_type_display()})"

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
        'assets.Experiment',
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

class ResourceAllocation(models.Model):
    """
    Represents shared resource allocation from organizations.
    """
    organization = models.ForeignKey(
        'research.Organization',
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