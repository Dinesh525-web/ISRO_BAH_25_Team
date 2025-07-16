"""
MOSDAC domain ontology definition.
"""
from typing import Dict, List, Set
from dataclasses import dataclass


@dataclass
class EntityType:
    """Entity type definition."""
    name: str
    description: str
    properties: List[str]
    parent_types: List[str]


@dataclass
class RelationType:
    """Relation type definition."""
    name: str
    description: str
    source_types: List[str]
    target_types: List[str]
    properties: List[str]


class MOSDACOntology:
    """MOSDAC domain ontology."""
    
    def __init__(self):
        self.entity_types = self._define_entity_types()
        self.relation_types = self._define_relation_types()
        self.hierarchy = self._build_hierarchy()
    
    def _define_entity_types(self) -> Dict[str, EntityType]:
        """Define entity types for MOSDAC domain."""
        return {
            "satellite": EntityType(
                name="satellite",
                description="Artificial satellite for Earth observation",
                properties=["launch_date", "orbit_type", "mission_life", "status"],
                parent_types=["physical_object"]
            ),
            "instrument": EntityType(
                name="instrument",
                description="Scientific instrument on satellite",
                properties=["type", "resolution", "spectral_bands", "swath_width"],
                parent_types=["equipment"]
            ),
            "parameter": EntityType(
                name="parameter",
                description="Measurable parameter from satellite data",
                properties=["unit", "range", "accuracy", "frequency"],
                parent_types=["measurement"]
            ),
            "location": EntityType(
                name="location",
                description="Geographic location or region",
                properties=["latitude", "longitude", "type", "country"],
                parent_types=["spatial_entity"]
            ),
            "organization": EntityType(
                name="organization",
                description="Organization or institution",
                properties=["type", "country", "established", "website"],
                parent_types=["agent"]
            ),
            "application": EntityType(
                name="application",
                description="Application domain or use case",
                properties=["domain", "users", "benefits", "requirements"],
                parent_types=["abstract_entity"]
            ),
            "data_product": EntityType(
                name="data_product",
                description="Processed satellite data product",
                properties=["level", "format", "coverage", "update_frequency"],
                parent_types=["information_object"]
            ),
            "mission": EntityType(
                name="mission",
                description="Satellite mission",
                properties=["objective", "duration", "cost", "agencies"],
                parent_types=["project"]
            ),
            "phenomenon": EntityType(
                name="phenomenon",
                description="Natural or atmospheric phenomenon",
                properties=["type", "scale", "intensity", "duration"],
                parent_types=["natural_entity"]
            ),
            "algorithm": EntityType(
                name="algorithm",
                description="Data processing algorithm",
                properties=["input", "output", "methodology", "version"],
                parent_types=["process"]
            )
        }
    
    def _define_relation_types(self) -> Dict[str, RelationType]:
        """Define relation types for MOSDAC domain."""
        return {
            "has_instrument": RelationType(
                name="has_instrument",
                description="Satellite has instrument",
                source_types=["satellite"],
                target_types=["instrument"],
                properties=["installation_date", "status"]
            ),
            "measures": RelationType(
                name="measures",
                description="Instrument measures parameter",
                source_types=["instrument"],
                target_types=["parameter"],
                properties=["accuracy", "frequency", "method"]
            ),
            "observes": RelationType(
                name="observes",
                description="Satellite observes location",
                source_types=["satellite"],
                target_types=["location"],
                properties=["coverage", "revisit_time", "resolution"]
            ),
            "operated_by": RelationType(
                name="operated_by",
                description="Satellite operated by organization",
                source_types=["satellite"],
                target_types=["organization"],
                properties=["role", "responsibility", "start_date"]
            ),
            "used_for": RelationType(
                name="used_for",
                description="Data used for application",
                source_types=["data_product", "parameter"],
                target_types=["application"],
                properties=["purpose", "importance", "frequency"]
            ),
            "generates": RelationType(
                name="generates",
                description="Instrument generates data product",
                source_types=["instrument"],
                target_types=["data_product"],
                properties=["processing_level", "format", "frequency"]
            ),
            "part_of": RelationType(
                name="part_of",
                description="Component is part of larger system",
                source_types=["instrument", "satellite"],
                target_types=["mission", "satellite"],
                properties=["role", "contribution"]
            ),
            "processes": RelationType(
                name="processes",
                description="Algorithm processes data",
                source_types=["algorithm"],
                target_types=["data_product", "parameter"],
                properties=["input_type", "output_type", "method"]
            ),
            "detects": RelationType(
                name="detects",
                description="Instrument detects phenomenon",
                source_types=["instrument"],
                target_types=["phenomenon"],
                properties=["sensitivity", "accuracy", "method"]
            ),
            "located_in": RelationType(
                name="located_in",
                description="Location is within another location",
                source_types=["location"],
                target_types=["location"],
                properties=["relationship_type", "containment"]
            )
        }
    
    def _build_hierarchy(self) -> Dict[str, List[str]]:
        """Build entity type hierarchy."""
        hierarchy = {}
        
        for entity_type in self.entity_types.values():
            for parent_type in entity_type.parent_types:
                if parent_type not in hierarchy:
                    hierarchy[parent_type] = []
                hierarchy[parent_type].append(entity_type.name)
        
        return hierarchy
    
    def get_entity_type(self, name: str) -> EntityType:
        """Get entity type by name."""
        return self.entity_types.get(name)
    
    def get_relation_type(self, name: str) -> RelationType:
        """Get relation type by name."""
        return self.relation_types.get(name)
    
    def get_valid_relations(self, source_type: str, target_type: str) -> List[str]:
        """Get valid relations between two entity types."""
        valid_relations = []
        
        for relation_name, relation_type in self.relation_types.items():
            if (source_type in relation_type.source_types and 
                target_type in relation_type.target_types):
                valid_relations.append(relation_name)
        
        return valid_relations
    
    def validate_entity(self, entity_type: str, properties: Dict[str, any]) -> bool:
        """Validate entity against ontology."""
        if entity_type not in self.entity_types:
            return False
        
        entity_def = self.entity_types[entity_type]
        
        # Check if all required properties are present
        # (In a full implementation, we'd define required vs optional properties)
        return True
    
    def validate_relation(
        self, 
        relation_type: str, 
        source_type: str, 
        target_type: str
    ) -> bool:
        """Validate relation against ontology."""
        if relation_type not in self.relation_types:
            return False
        
        relation_def = self.relation_types[relation_type]
        
        return (source_type in relation_def.source_types and 
                target_type in relation_def.target_types)
