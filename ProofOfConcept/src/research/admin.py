from django.contrib import admin
from .models import Organization, Researcher, OrganizationAffiliation


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    """Admin interface for Organization model"""
    list_display = ['name', 'type', 'location']
    list_filter = ['type']
    search_fields = ['name', 'location']
    readonly_fields = ['id']

    fieldsets = (
        (None, {
            'fields': ('id', 'name', 'type')
        }),
        ('Location', {
            'fields': ('location', 'website')
        }),
        ('Policies', {
            'fields': ('policies',)
        })
    )


@admin.register(Researcher)
class ResearcherAdmin(admin.ModelAdmin):
    """Admin interface for Researcher model"""
    list_display = ['full_name', 'email', 'orcid_id']
    search_fields = ['first_name', 'last_name', 'email', 'orcid_id']
    readonly_fields = ['id']

    fieldsets = (
        (None, {
            'fields': ('id', 'first_name', 'last_name', 'email', 'orcid_id')
        }),
        ('Research Profile', {
            'fields': ('expertise_areas', 'research_interests')
        })
    )


@admin.register(OrganizationAffiliation)
class OrganizationAffiliationAdmin(admin.ModelAdmin):
    """Admin interface for OrganizationAffiliation model"""
    list_display = ['researcher', 'organization', 'role', 'start_date', 'end_date', 'is_primary']
    list_filter = ['is_primary', 'start_date']
    search_fields = ['researcher__first_name', 'researcher__last_name', 'organization__name', 'role']
    date_hierarchy = 'start_date'

    fieldsets = (
        (None, {
            'fields': ('researcher', 'organization', 'role')
        }),
        ('Duration', {
            'fields': ('start_date', 'end_date', 'is_primary')
        })
    )
