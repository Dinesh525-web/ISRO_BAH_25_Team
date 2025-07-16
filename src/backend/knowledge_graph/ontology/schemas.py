"""
Pydantic schemas for knowledge graph entities and relations.
"""
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field, validator, ConfigDict


class EntityTypeEnum(str, Enum):
    """Enumeration of entity types in MOSDAC domain."""
    SATELLITE = "satellite"
    INSTRUMENT = "instrument"
    PARAMETER = "parameter"
    LOCATION = "location"
    ORGANIZATION = "organization"
    APPLICATION = "application"
    DATA_PRODUCT = "data_product"
    MISSION = "mission"
    PHENOMENON = "phenomenon"
    ALGORITHM = "algorithm"
    PERSON = "person"
    EVENT = "event"
    TEMPORAL = "temporal"


class RelationTypeEnum(str, Enum):
    """Enumeration of relation types in MOSDAC domain."""
    HAS_INSTRUMENT = "has_instrument"
    MEASURES = "measures"
    OBSERVES = "observes"
    OPERATED_BY = "operated_by"
    USED_FOR = "used_for"
    GENERATES = "generates"
    PART_OF = "part_of"
    PROCESSES = "processes"
    DETECTS = "detects"
    LOCATED_IN = "located_in"
    DEVELOPED_BY = "developed_by"
    COLLABORATES_WITH = "collaborates_with"
    SUCCEEDS = "succeeds"
    PRECEDES = "precedes"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    CONTAINS = "contains"
    APPLIES_TO = "applies_to"


class BaseKGSchema(BaseModel):
    """Base schema for knowledge graph objects."""
    
    id: Optional[str] = Field(None, description="Neo4j node/relationship ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    source_documents: List[str] = Field(default_factory=list, description="Source document IDs")
    
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        validate_assignment=True,
        str_strip_whitespace=True,
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
        }
    )


class EntityProperties(BaseModel):
    """Entity properties schema."""
    
    # Common properties
    description: Optional[str] = Field(None, description="Entity description")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    status: Optional[str] = Field(None, description="Current status")
    
    # Satellite-specific properties
    launch_date: Optional[str] = Field(None, description="Launch date")
    orbit_type: Optional[str] = Field(None, description="Orbit type")
    mission_life: Optional[str] = Field(None, description="Mission life span")
    altitude: Optional[float] = Field(None, description="Orbital altitude in km")
    inclination: Optional[float] = Field(None, description="Orbital inclination in degrees")
    
    # Instrument-specific properties
    instrument_type: Optional[str] = Field(None, description="Type of instrument")
    resolution: Optional[str] = Field(None, description="Spatial resolution")
    spectral_bands: List[str] = Field(default_factory=list, description="Spectral bands")
    swath_width: Optional[float] = Field(None, description="Swath width in km")
    
    # Parameter-specific properties
    unit: Optional[str] = Field(None, description="Unit of measurement")
    range_min: Optional[float] = Field(None, description="Minimum value")
    range_max: Optional[float] = Field(None, description="Maximum value")
    accuracy: Optional[float] = Field(None, description="Measurement accuracy")
    frequency: Optional[str] = Field(None, description="Measurement frequency")
    
    # Location-specific properties
    latitude: Optional[float] = Field(None, ge=-90, le=90, description="Latitude")
    longitude: Optional[float] = Field(None, ge=-180, le=180, description="Longitude")
    country: Optional[str] = Field(None, description="Country")
    region: Optional[str] = Field(None, description="Geographic region")
    location_type: Optional[str] = Field(None, description="Type of location")
    
    # Organization-specific properties
    organization_type: Optional[str] = Field(None, description="Type of organization")
    established: Optional[str] = Field(None, description="Establishment date")
    website: Optional[str] = Field(None, description="Official website")
    headquarters: Optional[str] = Field(None, description="Headquarters location")
    
    # Application-specific properties
    domain: Optional[str] = Field(None, description="Application domain")
    users: List[str] = Field(default_factory=list, description="User groups")
    benefits: List[str] = Field(default_factory=list, description="Benefits")
    requirements: List[str] = Field(default_factory=list, description="Requirements")
    
    # Data product-specific properties
    data_level: Optional[str] = Field(None, description="Data processing level")
    format: Optional[str] = Field(None, description="Data format")
    coverage: Optional[str] = Field(None, description="Coverage area")
    update_frequency: Optional[str] = Field(None, description="Update frequency")
    
    # Custom properties
    custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Custom properties")


class KGEntity(BaseKGSchema):
    """Knowledge graph entity schema."""
    
    name: str = Field(..., description="Entity name")
    type: EntityTypeEnum = Field(..., description="Entity type")
    properties: EntityProperties = Field(default_factory=EntityProperties, description="Entity properties")
    embedding: Optional[List[float]] = Field(None, description="Entity embedding vector")
    
    @validator('name')
    def validate_name(cls, v):
        """Validate entity name."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Entity name cannot be empty')
        return v.strip()
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if v is not None and len(v) == 0:
            raise ValueError('Embedding vector cannot be empty')
        return v


class RelationProperties(BaseModel):
    """Relation properties schema."""
    
    # Common properties
    context: Optional[str] = Field(None, description="Context of the relation")
    strength: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relation strength")
    temporal_start: Optional[datetime] = Field(None, description="Temporal start of relation")
    temporal_end: Optional[datetime] = Field(None, description="Temporal end of relation")
    
    # Instrument-parameter relation properties
    measurement_method: Optional[str] = Field(None, description="Measurement method")
    measurement_accuracy: Optional[float] = Field(None, description="Measurement accuracy")
    measurement_frequency: Optional[str] = Field(None, description="Measurement frequency")
    
    # Satellite-location relation properties
    coverage_area: Optional[str] = Field(None, description="Coverage area")
    revisit_time: Optional[str] = Field(None, description="Revisit time")
    spatial_resolution: Optional[str] = Field(None, description="Spatial resolution")
    
    # Organization relation properties
    role: Optional[str] = Field(None, description="Role in the relation")
    responsibility: Optional[str] = Field(None, description="Responsibility")
    collaboration_type: Optional[str] = Field(None, description="Type of collaboration")
    
    # Data processing relation properties
    processing_level: Optional[str] = Field(None, description="Data processing level")
    input_type: Optional[str] = Field(None, description="Input data type")
    output_type: Optional[str] = Field(None, description="Output data type")
    processing_method: Optional[str] = Field(None, description="Processing method")
    
    # Custom properties
    custom_properties: Dict[str, Any] = Field(default_factory=dict, description="Custom properties")


class KGRelation(BaseKGSchema):
    """Knowledge graph relation schema."""
    
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relation_type: RelationTypeEnum = Field(..., description="Relation type")
    properties: RelationProperties = Field(default_factory=RelationProperties, description="Relation properties")
    
    # Optional entity information for easier access
    source_entity: Optional[KGEntity] = Field(None, description="Source entity object")
    target_entity: Optional[KGEntity] = Field(None, description="Target entity object")
    
    @validator('source_entity_id', 'target_entity_id')
    def validate_entity_ids(cls, v):
        """Validate entity IDs."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Entity ID cannot be empty')
        return v.strip()


class KGPath(BaseModel):
    """Knowledge graph path schema."""
    
    entities: List[KGEntity] = Field(..., description="Entities in the path")
    relations: List[KGRelation] = Field(..., description="Relations in the path")
    length: int = Field(..., description="Path length")
    path_type: str = Field(..., description="Type of path")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Path confidence")
    
    @validator('length')
    def validate_length(cls, v, values):
        """Validate path length."""
        if 'relations' in values and v != len(values['relations']):
            raise ValueError('Path length must match number of relations')
        return v


class KGQuery(BaseModel):
    """Knowledge graph query schema."""
    
    query_type: str = Field(..., description="Type of query")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Query filters")
    limit: int = Field(default=10, ge=1, le=1000, description="Result limit")
    offset: int = Field(default=0, ge=0, description="Result offset")
    include_properties: bool = Field(default=True, description="Include entity properties")
    include_relations: bool = Field(default=True, description="Include relations")


class KGQueryResult(BaseModel):
    """Knowledge graph query result schema."""
    
    entities: List[KGEntity] = Field(default_factory=list, description="Found entities")
    relations: List[KGRelation] = Field(default_factory=list, description="Found relations")
    paths: List[KGPath] = Field(default_factory=list, description="Found paths")
    total_count: int = Field(default=0, description="Total count of results")
    query_time: float = Field(default=0.0, description="Query execution time")
    query_info: Dict[str, Any] = Field(default_factory=dict, description="Query metadata")


class KGStatistics(BaseModel):
    """Knowledge graph statistics schema."""
    
    total_entities: int = Field(default=0, description="Total number of entities")
    total_relations: int = Field(default=0, description="Total number of relations")
    entities_by_type: Dict[str, int] = Field(default_factory=dict, description="Entity counts by type")
    relations_by_type: Dict[str, int] = Field(default_factory=dict, description="Relation counts by type")
    average_degree: float = Field(default=0.0, description="Average node degree")
    density: float = Field(default=0.0, description="Graph density")
    connected_components: int = Field(default=0, description="Number of connected components")
    clustering_coefficient: float = Field(default=0.0, description="Clustering coefficient")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class EntityExtractionResult(BaseModel):
    """Entity extraction result schema."""
    
    entities: List[KGEntity] = Field(default_factory=list, description="Extracted entities")
    extraction_method: str = Field(..., description="Extraction method used")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold")
    processing_time: float = Field(default=0.0, description="Processing time")
    source_document_id: str = Field(..., description="Source document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")


class RelationExtractionResult(BaseModel):
    """Relation extraction result schema."""
    
    relations: List[KGRelation] = Field(default_factory=list, description="Extracted relations")
    extraction_method: str = Field(..., description="Extraction method used")
    confidence_threshold: float = Field(default=0.5, description="Confidence threshold")
    processing_time: float = Field(default=0.0, description="Processing time")
    source_document_id: str = Field(..., description="Source document ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata")


class KGValidationResult(BaseModel):
    """Knowledge graph validation result schema."""
    
    is_valid: bool = Field(..., description="Overall validation result")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    entity_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Entity-specific errors")
    relation_errors: Dict[str, List[str]] = Field(default_factory=dict, description="Relation-specific errors")
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Consistency score")
    validation_time: float = Field(default=0.0, description="Validation time")


class KGBuildConfig(BaseModel):
    """Configuration for knowledge graph building."""
    
    entity_extraction_enabled: bool = Field(default=True, description="Enable entity extraction")
    relation_extraction_enabled: bool = Field(default=True, description="Enable relation extraction")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Processing batch size")
    max_entities_per_document: int = Field(default=100, ge=1, description="Max entities per document")
    max_relations_per_document: int = Field(default=200, ge=1, description="Max relations per document")
    enable_validation: bool = Field(default=True, description="Enable validation")
    enable_deduplication: bool = Field(default=True, description="Enable deduplication")
    parallel_processing: bool = Field(default=False, description="Enable parallel processing")
    processing_timeout: int = Field(default=300, ge=1, description="Processing timeout in seconds")


class KGExportConfig(BaseModel):
    """Configuration for knowledge graph export."""
    
    format: str = Field(..., description="Export format")
    include_entities: bool = Field(default=True, description="Include entities")
    include_relations: bool = Field(default=True, description="Include relations")
    include_properties: bool = Field(default=True, description="Include properties")
    include_metadata: bool = Field(default=False, description="Include metadata")
    filter_by_type: List[str] = Field(default_factory=list, description="Filter by entity types")
    filter_by_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Filter by confidence")
    output_path: str = Field(..., description="Output file path")
    compression: Optional[str] = Field(None, description="Compression format")

