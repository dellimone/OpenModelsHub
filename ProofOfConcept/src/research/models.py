
import uuid
from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator

class OrganizationType(models.TextChoices):
    UNIVERSITY = 'UNIVERSITY', 'University'
    RESEARCH_INSTITUTE = 'RESEARCH_INSTITUTE', 'Research Institute'
    CORPORATION = 'CORPORATION', 'Corporation'
    GOVERNMENT = 'GOVERNMENT', 'Government'
    NON_PROFIT = 'NON_PROFIT', 'Non-Profit'
    CONSORTIUM = 'CONSORTIUM', 'Consortium'

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

