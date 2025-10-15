from django.shortcuts import render, get_object_or_404
from django.db.models import Q, Sum, Avg, Count
from django.core.paginator import Paginator
from django.contrib.postgres.search import TrigramSimilarity
from assets.models import Model, Dataset, Experiment, ExperimentDataset
from research.models import Researcher, Organization
from resources.models import ComputationalResource, ResourceUtilization


def home(request):
    """Homepage with overview statistics"""
    context = {
        'total_models': Model.objects.count(),
        'total_datasets': Dataset.objects.count(),
        'total_experiments': Experiment.objects.count(),
        'total_researchers': Researcher.objects.count(),
        'recent_models': Model.objects.order_by('-created_at')[:5],
        'recent_datasets': Dataset.objects.order_by('-created_at')[:5],
        'recent_experiments': Experiment.objects.order_by('-created_at')[:5],
    }
    return render(request, 'frontend/home.html', context)


def model_list(request):
    """List all ML models with search and filtering"""
    models = Model.objects.select_related('created_by', 'organization').order_by('-created_at')

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        models = models.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(models, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': models.count(),
    }
    return render(request, 'frontend/model_list.html', context)


def model_detail(request, model_id):
    """Detailed view of a specific ML model"""
    model = get_object_or_404(Model.objects.select_related('created_by', 'organization', 'produced_by'), id=model_id)

    # Get the experiment that produced this model
    experiment = model.produced_by

    # Get datasets used in the producing experiment
    experiment_datasets = []
    if experiment:
        experiment_datasets = ExperimentDataset.objects.filter(
            experiment=experiment
        ).select_related('dataset')

    # Get resource utilization for the producing experiment
    resource_utilizations = []
    total_carbon = 0
    total_energy = 0
    total_cost = 0
    avg_cpu = avg_memory_gb = avg_gpu = total_duration = 0

    if experiment:
        resource_utilizations = ResourceUtilization.objects.filter(
            experiment=experiment
        ).select_related('resource')

        # Calculate environmental and cost totals from ResourceUtilization
        if resource_utilizations.exists():
            totals = resource_utilizations.aggregate(
                carbon=Sum('carbon_footprint_kg'),
                energy=Sum('energy_consumption_kwh'),
                cost=Sum('total_cost'),
                avg_cpu=Avg('cpu_utilization_percent'),
                avg_memory=Avg('memory_usage_gb'),
                avg_gpu=Avg('gpu_utilization_percent'),
                duration=Sum('duration_seconds')
            )
            total_carbon = totals['carbon'] or 0
            total_energy = totals['energy'] or 0
            total_cost = totals['cost'] or 0
            avg_cpu = totals['avg_cpu'] or 0
            avg_memory_gb = totals['avg_memory'] or 0
            avg_gpu = totals['avg_gpu'] or 0
            total_duration = totals['duration'] or 0

    context = {
        'model': model,
        'experiment': experiment,
        'experiment_datasets': experiment_datasets,
        'resource_utilizations': resource_utilizations[:10],
        'total_carbon': round(total_carbon, 2),
        'total_energy': round(total_energy, 2),
        'total_cost': round(float(total_cost), 2),
        'avg_cpu': round(avg_cpu, 1),
        'avg_memory_gb': round(avg_memory_gb, 1),
        'avg_gpu': round(avg_gpu, 1),
        'total_duration': total_duration,
        'has_resource_data': resource_utilizations.exists() if resource_utilizations else False,
    }
    return render(request, 'frontend/model_detail.html', context)


def dataset_list(request):
    """List all datasets with search and filtering"""
    datasets = Dataset.objects.select_related('created_by', 'organization').order_by('-created_at')

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        datasets = datasets.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(datasets, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': datasets.count(),
    }
    return render(request, 'frontend/dataset_list.html', context)


def dataset_detail(request, dataset_id):
    """Detailed view of a specific dataset"""
    dataset = get_object_or_404(Dataset.objects.select_related('created_by', 'organization'), id=dataset_id)

    # Get experiments that use this dataset
    experiment_links = ExperimentDataset.objects.filter(
        dataset=dataset
    ).select_related('experiment')

    # Get models produced by those experiments
    experiments = [link.experiment for link in experiment_links]
    related_models = Model.objects.filter(produced_by__in=experiments).select_related('produced_by')

    context = {
        'dataset': dataset,
        'experiment_links': experiment_links,
        'related_models': related_models,
    }
    return render(request, 'frontend/dataset_detail.html', context)


def experiment_list(request):
    """List all experiments with search and filtering"""
    experiments = Experiment.objects.select_related('created_by', 'organization').order_by('-created_at')

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        experiments = experiments.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(experiments, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': experiments.count(),
    }
    return render(request, 'frontend/experiment_list.html', context)


def experiment_detail(request, experiment_id):
    """Detailed view of a specific experiment"""
    experiment = get_object_or_404(
        Experiment.objects.select_related('created_by', 'organization', 'base_model'),
        id=experiment_id
    )

    # Get datasets used in this experiment
    experiment_datasets = ExperimentDataset.objects.filter(
        experiment=experiment
    ).select_related('dataset')

    # Get hyperparameters
    hyperparameters = experiment.hyperparameters.all()

    # Get metrics
    metrics = experiment.metrics.all().order_by('metric_name', 'step')

    # Get checkpoints
    checkpoints = experiment.checkpoints.all().order_by('-step')

    # Get produced models
    produced_models = experiment.produced_models.all()

    # Get resource utilization for this experiment
    resource_utilizations = ResourceUtilization.objects.filter(
        experiment=experiment
    ).select_related('resource')

    # Calculate totals
    total_carbon = 0
    total_energy = 0
    total_cost = 0
    avg_cpu = avg_memory_gb = avg_gpu = total_duration = 0

    if resource_utilizations.exists():
        totals = resource_utilizations.aggregate(
            carbon=Sum('carbon_footprint_kg'),
            energy=Sum('energy_consumption_kwh'),
            cost=Sum('total_cost'),
            avg_cpu=Avg('cpu_utilization_percent'),
            avg_memory=Avg('memory_usage_gb'),
            avg_gpu=Avg('gpu_utilization_percent'),
            duration=Sum('duration_seconds')
        )
        total_carbon = totals['carbon'] or 0
        total_energy = totals['energy'] or 0
        total_cost = totals['cost'] or 0
        avg_cpu = totals['avg_cpu'] or 0
        avg_memory_gb = totals['avg_memory'] or 0
        avg_gpu = totals['avg_gpu'] or 0
        total_duration = totals['duration'] or 0

    context = {
        'experiment': experiment,
        'experiment_datasets': experiment_datasets,
        'hyperparameters': hyperparameters,
        'metrics': metrics,
        'checkpoints': checkpoints,
        'produced_models': produced_models,
        'resource_utilizations': resource_utilizations[:10],
        'total_carbon': round(total_carbon, 2),
        'total_energy': round(total_energy, 2),
        'total_cost': round(float(total_cost), 2),
        'avg_cpu': round(avg_cpu, 1),
        'avg_memory_gb': round(avg_memory_gb, 1),
        'avg_gpu': round(avg_gpu, 1),
        'total_duration': total_duration,
        'has_resource_data': resource_utilizations.exists(),
    }
    return render(request, 'frontend/experiment_detail.html', context)


def researcher_list(request):
    """List all researchers"""
    researchers = Researcher.objects.prefetch_related('organizations').order_by('last_name', 'first_name')

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        researchers = researchers.filter(
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(orcid_id__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(researchers, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': researchers.count(),
    }
    return render(request, 'frontend/researcher_list.html', context)


def researcher_detail(request, researcher_id):
    """Detailed view of a specific researcher"""
    researcher = get_object_or_404(
        Researcher.objects.prefetch_related('organizations'),
        id=researcher_id
    )

    # Get researcher's contributions using correct related names
    created_models = researcher.model_created.all()[:10]
    created_datasets = researcher.dataset_created.all()[:10]
    created_experiments = researcher.experiment_created.all()[:10]

    # Get organizations
    affiliations = researcher.organizationaffiliation_set.select_related('organization').order_by('-start_date')

    context = {
        'researcher': researcher,
        'created_models': created_models,
        'created_datasets': created_datasets,
        'created_experiments': created_experiments,
        'affiliations': affiliations,
        'total_models': researcher.model_created.count(),
        'total_datasets': researcher.dataset_created.count(),
        'total_experiments': researcher.experiment_created.count(),
    }
    return render(request, 'frontend/researcher_detail.html', context)


def organization_list(request):
    """List all organizations"""
    organizations = Organization.objects.order_by('name')

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        organizations = organizations.filter(
            Q(name__icontains=search_query) |
            Q(location__icontains=search_query)
        )

    # Pagination
    paginator = Paginator(organizations, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': organizations.count(),
    }
    return render(request, 'frontend/organization_list.html', context)


def search(request):
    """Global search across all asset types"""
    search_query = request.GET.get('q', '')
    results = {}

    if search_query:
        # Search models
        models = Model.objects.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        ).select_related('created_by', 'organization')[:5]

        # Search datasets
        datasets = Dataset.objects.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        ).select_related('created_by', 'organization')[:5]

        # Search experiments
        experiments = Experiment.objects.filter(
            Q(name__icontains=search_query) |
            Q(description__icontains=search_query) |
            Q(persistent_identifier__icontains=search_query)
        ).select_related('created_by', 'organization')[:5]

        # Search researchers
        researchers = Researcher.objects.filter(
            Q(first_name__icontains=search_query) |
            Q(last_name__icontains=search_query) |
            Q(email__icontains=search_query) |
            Q(orcid_id__icontains=search_query)
        )[:5]

        results = {
            'models': models,
            'datasets': datasets,
            'experiments': experiments,
            'researchers': researchers,
            'total_results': len(models) + len(datasets) + len(experiments) + len(researchers),
        }

    context = {
        'search_query': search_query,
        'results': results,
    }
    return render(request, 'frontend/search.html', context)


def resources_dashboard(request):
    """Dashboard showing all computational resources and utilization"""
    resources = ComputationalResource.objects.all()

    # Search functionality
    search_query = request.GET.get('q', '')
    if search_query:
        resources = resources.filter(
            Q(name__icontains=search_query) |
            Q(location__icontains=search_query)
        )

    # Resource statistics by type
    total_resources = ComputationalResource.objects.count()
    available_resources = ComputationalResource.objects.filter(availability='AVAILABLE').count()
    cloud_resources = ComputationalResource.objects.filter(resource_type='CLOUD').count()
    local_resources = ComputationalResource.objects.filter(resource_type='LOCAL').count()
    hpc_resources = ComputationalResource.objects.filter(resource_type='HPC').count()

    # Utilization statistics
    total_utilizations = ResourceUtilization.objects.count()

    if total_utilizations > 0:
        avg_cpu_utilization = ResourceUtilization.objects.aggregate(
            avg_cpu=Avg('cpu_utilization_percent')
        )['avg_cpu'] or 0
        avg_memory_utilization = ResourceUtilization.objects.aggregate(
            avg_memory=Avg('memory_usage_gb')
        )['avg_memory'] or 0
    else:
        avg_cpu_utilization = 0
        avg_memory_utilization = 0

    # Recent utilizations
    recent_utilizations = ResourceUtilization.objects.select_related(
        'resource', 'experiment'
    ).order_by('-start_time')[:10]

    # Pagination
    paginator = Paginator(resources, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'search_query': search_query,
        'total_count': resources.count(),
        'total_resources': total_resources,
        'available_resources': available_resources,
        'cloud_resources': cloud_resources,
        'local_resources': local_resources,
        'hpc_resources': hpc_resources,
        'total_utilizations': total_utilizations,
        'avg_cpu_utilization': round(avg_cpu_utilization, 1),
        'avg_memory_utilization': round(avg_memory_utilization, 1),
        'recent_utilizations': recent_utilizations,
    }
    return render(request, 'frontend/resources_dashboard.html', context)


def resource_detail(request, resource_id):
    """Detailed view of a specific computational resource"""
    resource = get_object_or_404(ComputationalResource, id=resource_id)

    # Get utilization history
    utilizations = ResourceUtilization.objects.filter(
        resource=resource
    ).select_related('experiment').order_by('-start_time')[:20]

    # Statistics
    total_utilizations = ResourceUtilization.objects.filter(resource=resource).count()

    if utilizations.exists():
        stats = utilizations.aggregate(
            avg_cpu=Avg('cpu_utilization_percent'),
            avg_memory=Avg('memory_usage_gb'),
            avg_gpu=Avg('gpu_utilization_percent'),
            total_carbon=Sum('carbon_footprint_kg'),
            total_energy=Sum('energy_consumption_kwh'),
            total_cost=Sum('total_cost')
        )
        avg_cpu = stats['avg_cpu'] or 0
        avg_memory = stats['avg_memory'] or 0
        avg_gpu = stats['avg_gpu'] or 0
        total_carbon = stats['total_carbon'] or 0
        total_energy = stats['total_energy'] or 0
        total_cost = stats['total_cost'] or 0
    else:
        avg_cpu = avg_memory = avg_gpu = 0
        total_carbon = total_energy = total_cost = 0

    context = {
        'resource': resource,
        'utilizations': utilizations,
        'total_utilizations': total_utilizations,
        'avg_cpu': round(avg_cpu, 1),
        'avg_memory': round(avg_memory, 1),
        'avg_gpu': round(avg_gpu, 1),
        'total_carbon': round(total_carbon, 2),
        'total_energy': round(total_energy, 2),
        'total_cost': round(float(total_cost), 2),
    }
    return render(request, 'frontend/resource_detail.html', context)


def environmental_dashboard(request):
    """Dashboard showing environmental impact across all resources"""
    from django.utils import timezone
    from datetime import timedelta

    # Get all resource utilizations (contains environmental data)
    utilizations = ResourceUtilization.objects.select_related('resource', 'experiment')

    total_records = utilizations.count()

    if total_records > 0:
        totals = utilizations.aggregate(
            total_carbon=Sum('carbon_footprint_kg'),
            total_energy=Sum('energy_consumption_kwh'),
            avg_carbon_intensity=Avg('carbon_intensity_value')
        )
        total_carbon = totals['total_carbon'] or 0
        total_energy = totals['total_energy'] or 0
        avg_carbon_intensity = totals['avg_carbon_intensity'] or 0
    else:
        total_carbon = total_energy = avg_carbon_intensity = 0

    # Recent high-impact activities
    high_impact = utilizations.filter(
        carbon_footprint_kg__gte=5.0
    ).order_by('-carbon_footprint_kg')[:10]

    # Environmental impact by resource type
    cloud_impact = utilizations.filter(resource__resource_type='CLOUD').aggregate(
        total_carbon=Sum('carbon_footprint_kg'),
        total_energy=Sum('energy_consumption_kwh')
    )
    local_impact = utilizations.filter(resource__resource_type='LOCAL').aggregate(
        total_carbon=Sum('carbon_footprint_kg'),
        total_energy=Sum('energy_consumption_kwh')
    )
    hpc_impact = utilizations.filter(resource__resource_type='HPC').aggregate(
        total_carbon=Sum('carbon_footprint_kg'),
        total_energy=Sum('energy_consumption_kwh')
    )

    # Recent environmental impacts
    recent_impacts = utilizations.order_by('-start_time')[:15]

    # Experiments with highest environmental impact
    experiments_impact = utilizations.values(
        'experiment__name', 'experiment__id'
    ).annotate(
        total_carbon=Sum('carbon_footprint_kg'),
        total_energy=Sum('energy_consumption_kwh')
    ).order_by('-total_carbon')[:10]

    # Sustainability metrics
    low_impact_count = utilizations.filter(carbon_footprint_kg__lt=1.0).count()
    medium_impact_count = utilizations.filter(
        carbon_footprint_kg__gte=1.0, carbon_footprint_kg__lt=5.0
    ).count()
    high_impact_count = utilizations.filter(carbon_footprint_kg__gte=5.0).count()

    context = {
        'total_impacts': total_records,
        'total_carbon': round(total_carbon, 2),
        'total_energy': round(total_energy, 2),
        'avg_carbon_intensity': round(avg_carbon_intensity, 4),
        'high_impact': high_impact,
        'cloud_impact': {
            'carbon': round(cloud_impact['total_carbon'] or 0, 2),
            'energy': round(cloud_impact['total_energy'] or 0, 2)
        },
        'local_impact': {
            'carbon': round(local_impact['total_carbon'] or 0, 2),
            'energy': round(local_impact['total_energy'] or 0, 2)
        },
        'hpc_impact': {
            'carbon': round(hpc_impact['total_carbon'] or 0, 2),
            'energy': round(hpc_impact['total_energy'] or 0, 2)
        },
        'recent_impacts': recent_impacts,
        'experiments_impact': experiments_impact,
        'low_impact_count': low_impact_count,
        'medium_impact_count': medium_impact_count,
        'high_impact_count': high_impact_count,
    }
    return render(request, 'frontend/environmental_dashboard.html', context)


def sustainability_report(request):
    """Comprehensive sustainability analytics and reporting"""
    from django.utils import timezone
    from datetime import timedelta

    # Time-based analysis (last 30 days)
    thirty_days_ago = timezone.now() - timedelta(days=30)

    recent_utilizations = ResourceUtilization.objects.filter(start_time__gte=thirty_days_ago)

    # Monthly trends
    if recent_utilizations.exists():
        monthly_totals = recent_utilizations.aggregate(
            carbon=Sum('carbon_footprint_kg'),
            energy=Sum('energy_consumption_kwh')
        )
        monthly_carbon = monthly_totals['carbon'] or 0
        monthly_energy = monthly_totals['energy'] or 0

        # Count unique experiments
        monthly_experiments = recent_utilizations.values('experiment').distinct().count()
    else:
        monthly_carbon = monthly_energy = 0
        monthly_experiments = 0

    # Efficiency metrics
    avg_carbon_per_experiment = monthly_carbon / max(monthly_experiments, 1)
    avg_energy_per_experiment = monthly_energy / max(monthly_experiments, 1)

    # Resource efficiency
    efficient_resources = ComputationalResource.objects.annotate(
        avg_carbon=Avg('utilizations__carbon_footprint_kg'),
        usage_count=Count('utilizations')
    ).filter(usage_count__gt=0).order_by('avg_carbon')[:10]

    # Compliance targets
    total_usage = recent_utilizations.count()
    if total_usage > 0:
        compliant_count = recent_utilizations.filter(carbon_footprint_kg__lte=5.0).count()
        compliance_rate = (compliant_count / total_usage) * 100
    else:
        compliant_count = 0
        compliance_rate = 0

    # Organization sustainability rankings
    org_sustainability = Organization.objects.annotate(
        total_carbon=Sum('model_assets__produced_by__resource_utilizations__carbon_footprint_kg'),
        model_count=Count('model_assets')
    ).filter(model_count__gt=0).order_by('total_carbon')[:10]

    context = {
        'monthly_carbon': round(monthly_carbon, 2),
        'monthly_energy': round(monthly_energy, 2),
        'monthly_experiments': monthly_experiments,
        'avg_carbon_per_experiment': round(avg_carbon_per_experiment, 2),
        'avg_energy_per_experiment': round(avg_energy_per_experiment, 2),
        'efficient_resources': efficient_resources,
        'compliance_rate': round(compliance_rate, 1),
        'org_sustainability': org_sustainability,
        'total_recent_impacts': total_usage,
        'compliant_count': compliant_count,
    }
    return render(request, 'frontend/sustainability_report.html', context)
