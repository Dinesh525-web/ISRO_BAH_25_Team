"""
Relation extractor for identifying relationships between entities in MOSDAC documents.
"""
import re
from typing import List, Dict, Any, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict

from ..ontology.mosdac_ontology import MOSDACOntology
from ..ontology.schemas import KGRelation, RelationTypeEnum, RelationProperties
from app.data_ingestion.processors.text_processor import TextProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RelationExtractor:
    """Extract relationships between entities from MOSDAC documents."""
    
    def __init__(self):
        self.ontology = MOSDACOntology()
        self.text_processor = TextProcessor()
        
        # Relation patterns for different types
        self.relation_patterns = {
            RelationTypeEnum.HAS_INSTRUMENT: [
                r'({satellite})\s+(?:has|carries|equipped\s+with|features)\s+(?:an?|the)?\s*({instrument})',
                r'({satellite})\s+(?:payload|instruments?):\s*({instrument})',
                r'({instrument})\s+(?:on|aboard|in)\s+({satellite})',
                r'({satellite})\s*[-–]\s*({instrument})',
            ],
            RelationTypeEnum.MEASURES: [
                r'({instrument})\s+(?:measures|detects|observes|monitors)\s+({parameter})',
                r'({parameter})\s+(?:measured|detected|observed|monitored)\s+(?:by|using|with)\s+({instrument})',
                r'({instrument})\s+(?:for|to)\s+(?:measure|detect|observe|monitor)\s+({parameter})',
                r'({parameter})\s+(?:data|measurements|observations)\s+(?:from|using)\s+({instrument})',
            ],
            RelationTypeEnum.OBSERVES: [
                r'({satellite})\s+(?:observes|monitors|covers|studies)\s+({location})',
                r'({location})\s+(?:observed|monitored|covered|studied)\s+(?:by|using)\s+({satellite})',
                r'({satellite})\s+(?:over|above)\s+({location})',
                r'({location})\s+(?:coverage|observation|monitoring)\s+(?:by|from)\s+({satellite})',
            ],
            RelationTypeEnum.OPERATED_BY: [
                r'({satellite})\s+(?:operated|managed|controlled)\s+(?:by|from)\s+({organization})',
                r'({organization})\s+(?:operates|manages|controls)\s+({satellite})',
                r'({satellite})\s+(?:mission|operation)\s+(?:by|from)\s+({organization})',
                r'({organization})\s*[-–]\s*({satellite})\s+(?:mission|operation)',
            ],
            RelationTypeEnum.USED_FOR: [
                r'({parameter}|{data_product})\s+(?:used|utilized|applied)\s+(?:for|in)\s+({application})',
                r'({application})\s+(?:using|utilizes|applies)\s+({parameter}|{data_product})',
                r'({parameter}|{data_product})\s+(?:for|in)\s+({application})\s+(?:applications?|studies?|research)',
                r'({application})\s+(?:requires|needs)\s+({parameter}|{data_product})',
            ],
            RelationTypeEnum.GENERATES: [
                r'({instrument})\s+(?:generates|produces|provides)\s+({data_product})',
                r'({data_product})\s+(?:generated|produced|provided)\s+(?:by|from)\s+({instrument})',
                r'({instrument})\s+(?:output|data|product):\s*({data_product})',
                r'({data_product})\s+(?:from|using)\s+({instrument})\s+(?:data|measurements)',
            ],
            RelationTypeEnum.PART_OF: [
                r'({instrument})\s+(?:part\s+of|component\s+of|in)\s+({satellite})',
                r'({satellite})\s+(?:includes|contains|has)\s+({instrument})',
                r'({instrument})\s+(?:subsystem|component)\s+(?:of|in)\s+({satellite})',
                r'({satellite})\s+(?:mission|system)\s+(?:includes|contains)\s+({instrument})',
            ],
            RelationTypeEnum.DETECTS: [
                r'({instrument})\s+(?:detects|identifies|finds)\s+({phenomenon})',
                r'({phenomenon})\s+(?:detected|identified|found)\s+(?:by|using)\s+({instrument})',
                r'({instrument})\s+(?:for|to)\s+(?:detect|identify|find)\s+({phenomenon})',
                r'({phenomenon})\s+(?:detection|identification)\s+(?:using|with)\s+({instrument})',
            ],
            RelationTypeEnum.LOCATED_IN: [
                r'({location})\s+(?:in|within|inside)\s+({location})',
                r'({location})\s+(?:part\s+of|region\s+of)\s+({location})',
                r'({location})\s*[,]\s*({location})',
                r'({location})\s+(?:state|province|region)\s+(?:of|in)\s+({location})',
            ],
            RelationTypeEnum.DEVELOPED_BY: [
                r'({satellite}|{instrument}|{algorithm})\s+(?:developed|built|designed)\s+(?:by|at)\s+({organization})',
                r'({organization})\s+(?:developed|built|designed)\s+({satellite}|{instrument}|{algorithm})',
                r'({satellite}|{instrument}|{algorithm})\s+(?:development|design)\s+(?:by|at)\s+({organization})',
                r'({organization})\s*[-–]\s*({satellite}|{instrument}|{algorithm})\s+(?:development|design)',
            ],
            RelationTypeEnum.PROCESSES: [
                r'({algorithm})\s+(?:processes|handles|analyzes)\s+({parameter}|{data_product})',
                r'({parameter}|{data_product})\s+(?:processed|handled|analyzed)\s+(?:by|using)\s+({algorithm})',
                r'({algorithm})\s+(?:for|to)\s+(?:process|handle|analyze)\s+({parameter}|{data_product})',
                r'({parameter}|{data_product})\s+(?:processing|analysis)\s+(?:using|with)\s+({algorithm})',
            ],
        }
        
        # Context patterns to improve relation extraction
        self.context_patterns = {
            'temporal': [
                r'(?:since|from|during|in)\s+(\d{4}|\w+\s+\d{4})',
                r'(?:launched|operational|active)\s+(?:since|from|in)\s+(\d{4}|\w+\s+\d{4})',
                r'(?:between|from)\s+(\d{4})\s+(?:to|and)\s+(\d{4})',
            ],
            'spatial': [
                r'(?:over|above|covering)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:in|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                r'(?:region|area|zone):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            ],
            'quantitative': [
                r'(\d+(?:\.\d+)?)\s*(km|m|degrees?|%|Hz|GHz|MHz)',
                r'(?:resolution|accuracy|frequency):\s*(\d+(?:\.\d+)?)\s*(km|m|degrees?|%|Hz|GHz|MHz)',
                r'(?:approximately|about|around)\s+(\d+(?:\.\d+)?)\s*(km|m|degrees?|%|Hz|GHz|MHz)',
            ],
        }
        
        # Negative patterns to exclude false positives
        self.negative_patterns = [
            r'(?:not|no|without|except|excluding)\s+',
            r'(?:unlike|different\s+from|compared\s+to)\s+',
            r'(?:if|when|while|although|though)\s+',
            r'(?:might|could|would|should|may)\s+',
        ]
    
    async def extract_relations(
        self,
        document: Dict[str, Any],
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract relations from a document given its entities.
        
        Args:
            document: Processed document
            entities: List of entities found in the document
            
        Returns:
            List of extracted relations
        """
        try:
            relations = []
            
            # Get document content
            content = document.get('main_content', '')
            title = document.get('title', '')
            
            # Combine title and content for relation extraction
            text = f"{title} {content}"
            
            # Create entity lookup by type
            entities_by_type = defaultdict(list)
            for entity in entities:
                entities_by_type[entity['type']].append(entity)
            
            # Extract relations using patterns
            pattern_relations = self._extract_relations_by_patterns(
                text, entities_by_type, document.get('id', '')
            )
            relations.extend(pattern_relations)
            
            # Extract relations from structured data
            structured_relations = self._extract_relations_from_structured_data(
                document, entities_by_type
            )
            relations.extend(structured_relations)
            
            # Extract co-occurrence relations
            cooccurrence_relations = self._extract_cooccurrence_relations(
                text, entities_by_type, document.get('id', '')
            )
            relations.extend(cooccurrence_relations)
            
            # Extract relations from satellite info
            satellite_relations = self._extract_relations_from_satellite_info(
                document.get('satellite_info', {}), entities_by_type, document.get('id', '')
            )
            relations.extend(satellite_relations)
            
            # Filter and validate relations
            relations = self._filter_and_validate_relations(relations)
            
            # Add confidence scores
            relations = self._calculate_relation_confidence(relations, text)
            
            logger.debug(f"Extracted {len(relations)} relations from document")
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def _extract_relations_by_patterns(
        self,
        text: str,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract relations using predefined patterns."""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                # Find all possible entity type combinations for this pattern
                entity_combinations = self._get_entity_combinations_for_pattern(
                    pattern, entities_by_type
                )
                
                for source_entities, target_entities in entity_combinations:
                    # Create specific pattern for these entity types
                    specific_pattern = self._create_specific_pattern(
                        pattern, source_entities, target_entities
                    )
                    
                    # Find matches in text
                    matches = re.finditer(specific_pattern, text, re.IGNORECASE)
                    
                    for match in matches:
                        # Check for negative patterns
                        if self._has_negative_context(text, match.start(), match.end()):
                            continue
                        
                        # Extract entity names from match
                        source_name = match.group(1).strip()
                        target_name = match.group(2).strip()
                        
                        # Find matching entities
                        source_entity = self._find_entity_by_name(source_name, source_entities)
                        target_entity = self._find_entity_by_name(target_name, target_entities)
                        
                        if source_entity and target_entity:
                            # Extract context around the match
                            context = self._extract_context(text, match.start(), match.end())
                            
                            # Create relation
                            relation = {
                                'source_entity': source_entity['name'],
                                'source_type': source_entity['type'],
                                'target_entity': target_entity['name'],
                                'target_type': target_entity['type'],
                                'relation_type': relation_type,
                                'confidence': 0.8,  # Base confidence for pattern matching
                                'context': context,
                                'source_documents': [document_id],
                                'extraction_method': 'pattern_matching',
                                'pattern': pattern,
                                'match_position': match.start(),
                                'properties': {}
                            }
                            
                            # Extract additional properties based on context
                            properties = self._extract_relation_properties(
                                relation_type, context, match.group(0)
                            )
                            relation['properties'].update(properties)
                            
                            relations.append(relation)
        
        return relations
    
    def _get_entity_combinations_for_pattern(
        self,
        pattern: str,
        entities_by_type: Dict[str, List[Dict[str, Any]]]
    ) -> List[Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]]:
        """Get valid entity combinations for a pattern."""
        combinations = []
        
        # Extract entity type placeholders from pattern
        placeholders = re.findall(r'\{(\w+)\}', pattern)
        
        if len(placeholders) >= 2:
            source_type = placeholders[0]
            target_type = placeholders[1]
            
            # Handle multiple source types (e.g., {parameter}|{data_product})
            source_types = source_type.split('|')
            target_types = target_type.split('|')
            
            for src_type in source_types:
                for tgt_type in target_types:
                    if src_type in entities_by_type and tgt_type in entities_by_type:
                        combinations.append((
                            entities_by_type[src_type],
                            entities_by_type[tgt_type]
                        ))
        
        return combinations
    
    def _create_specific_pattern(
        self,
        pattern: str,
        source_entities: List[Dict[str, Any]],
        target_entities: List[Dict[str, Any]]
    ) -> str:
        """Create a specific regex pattern for given entities."""
        # Get entity names and aliases
        source_names = []
        target_names = []
        
        for entity in source_entities:
            source_names.append(re.escape(entity['name']))
            source_names.extend([re.escape(alias) for alias in entity.get('aliases', [])])
        
        for entity in target_entities:
            target_names.append(re.escape(entity['name']))
            target_names.extend([re.escape(alias) for alias in entity.get('aliases', [])])
        
        # Create regex groups
        source_group = '(' + '|'.join(source_names) + ')'
        target_group = '(' + '|'.join(target_names) + ')'
        
        # Replace placeholders in pattern
        specific_pattern = pattern
        placeholders = re.findall(r'\{(\w+)\}', pattern)
        
        if len(placeholders) >= 2:
            specific_pattern = specific_pattern.replace(f'{{{placeholders[0]}}}', source_group)
            specific_pattern = specific_pattern.replace(f'{{{placeholders[1]}}}', target_group)
        
        return specific_pattern
    
    def _has_negative_context(self, text: str, start: int, end: int) -> bool:
        """Check if the match has negative context."""
        # Check 50 characters before the match
        context_start = max(0, start - 50)
        context = text[context_start:start]
        
        for negative_pattern in self.negative_patterns:
            if re.search(negative_pattern, context, re.IGNORECASE):
                return True
        
        return False
    
    def _find_entity_by_name(
        self,
        name: str,
        entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find entity by name or alias."""
        name_lower = name.lower()
        
        for entity in entities:
            if entity['name'].lower() == name_lower:
                return entity
            
            # Check aliases
            for alias in entity.get('aliases', []):
                if alias.lower() == name_lower:
                    return entity
        
        return None
    
    def _extract_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        """Extract context around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        
        context = text[context_start:context_end]
        
        # Clean up the context
        context = re.sub(r'\s+', ' ', context).strip()
        
        return context
    
    def _extract_relation_properties(
        self,
        relation_type: RelationTypeEnum,
        context: str,
        match_text: str
    ) -> Dict[str, Any]:
        """Extract additional properties for a relation."""
        properties = {}
        
        # Extract temporal information
        temporal_matches = []
        for pattern in self.context_patterns['temporal']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            temporal_matches.extend(matches)
        
        if temporal_matches:
            properties['temporal_context'] = temporal_matches
        
        # Extract spatial information
        spatial_matches = []
        for pattern in self.context_patterns['spatial']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            spatial_matches.extend(matches)
        
        if spatial_matches:
            properties['spatial_context'] = spatial_matches
        
        # Extract quantitative information
        quantitative_matches = []
        for pattern in self.context_patterns['quantitative']:
            matches = re.findall(pattern, context, re.IGNORECASE)
            quantitative_matches.extend(matches)
        
        if quantitative_matches:
            properties['quantitative_context'] = quantitative_matches
        
        # Relation-specific properties
        if relation_type == RelationTypeEnum.MEASURES:
            # Extract measurement details
            accuracy_match = re.search(r'accuracy[:\s]+([^,\.\s]+)', context, re.IGNORECASE)
            if accuracy_match:
                properties['measurement_accuracy'] = accuracy_match.group(1)
            
            frequency_match = re.search(r'frequency[:\s]+([^,\.\s]+)', context, re.IGNORECASE)
            if frequency_match:
                properties['measurement_frequency'] = frequency_match.group(1)
        
        elif relation_type == RelationTypeEnum.OBSERVES:
            # Extract observation details
            coverage_match = re.search(r'coverage[:\s]+([^,\.\s]+)', context, re.IGNORECASE)
            if coverage_match:
                properties['coverage_area'] = coverage_match.group(1)
            
            resolution_match = re.search(r'resolution[:\s]+([^,\.\s]+)', context, re.IGNORECASE)
            if resolution_match:
                properties['spatial_resolution'] = resolution_match.group(1)
        
        elif relation_type == RelationTypeEnum.OPERATED_BY:
            # Extract operation details
            role_match = re.search(r'(?:role|responsibility)[:\s]+([^,\.\s]+)', context, re.IGNORECASE)
            if role_match:
                properties['role'] = role_match.group(1)
        
        return properties
    
    def _extract_relations_from_structured_data(
        self,
        document: Dict[str, Any],
        entities_by_type: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from structured data in the document."""
        relations = []
        
        # Extract from tables
        tables = document.get('tables', [])
        for table in tables:
            table_relations = self._extract_relations_from_table(table, entities_by_type)
            relations.extend(table_relations)
        
        # Extract from forms
        forms = document.get('forms', [])
        for form in forms:
            form_relations = self._extract_relations_from_form(form, entities_by_type)
            relations.extend(form_relations)
        
        return relations
    
    def _extract_relations_from_table(
        self,
        table: Dict[str, Any],
        entities_by_type: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from a table."""
        relations = []
        
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        # Look for entity columns
        entity_columns = {}
        for i, header in enumerate(headers):
            header_lower = header.lower()
            
            # Check if header matches entity types
            for entity_type, entities in entities_by_type.items():
                if entity_type in header_lower or any(
                    entity['name'].lower() in header_lower for entity in entities
                ):
                    entity_columns[i] = entity_type
        
        # Extract relations from rows
        for row in rows:
            if len(row) >= 2:
                # Find entity pairs in the row
                for i, cell1 in enumerate(row):
                    for j, cell2 in enumerate(row):
                        if i != j and i in entity_columns and j in entity_columns:
                            # Find matching entities
                            entity1 = self._find_entity_by_name(cell1, entities_by_type[entity_columns[i]])
                            entity2 = self._find_entity_by_name(cell2, entities_by_type[entity_columns[j]])
                            
                            if entity1 and entity2:
                                # Determine relation type based on entity types
                                relation_type = self._infer_relation_type(
                                    entity1['type'], entity2['type']
                                )
                                
                                if relation_type:
                                    relation = {
                                        'source_entity': entity1['name'],
                                        'source_type': entity1['type'],
                                        'target_entity': entity2['name'],
                                        'target_type': entity2['type'],
                                        'relation_type': relation_type,
                                        'confidence': 0.7,  # Medium confidence for table extraction
                                        'context': f"Table: {table.get('caption', 'Unknown')}",
                                        'extraction_method': 'table_extraction',
                                        'properties': {
                                            'table_headers': headers,
                                            'row_data': row
                                        }
                                    }
                                    relations.append(relation)
        
        return relations
    
    def _extract_relations_from_form(
        self,
        form: Dict[str, Any],
        entities_by_type: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Extract relations from a form."""
        relations = []
        
        # This is a simplified implementation
        # In practice, you would analyze form structure and field relationships
        
        return relations
    
    def _extract_cooccurrence_relations(
        self,
        text: str,
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract relations based on entity co-occurrence."""
        relations = []
        
        # Find all entity mentions in text
        entity_mentions = []
        for entity_type, entities in entities_by_type.items():
            for entity in entities:
                # Find all occurrences of this entity
                for match in re.finditer(re.escape(entity['name']), text, re.IGNORECASE):
                    entity_mentions.append({
                        'entity': entity,
                        'start': match.start(),
                        'end': match.end(),
                        'type': entity_type
                    })
        
        # Sort by position
        entity_mentions.sort(key=lambda x: x['start'])
        
        # Find co-occurrences within a window
        window_size = 200  # characters
        
        for i, mention1 in enumerate(entity_mentions):
            for j, mention2 in enumerate(entity_mentions[i+1:], i+1):
                # Check if entities are within window
                if mention2['start'] - mention1['end'] <= window_size:
                    # Don't create relations between same entities
                    if mention1['entity']['name'] == mention2['entity']['name']:
                        continue
                    
                    # Infer relation type
                    relation_type = self._infer_relation_type(
                        mention1['type'], mention2['type']
                    )
                    
                    if relation_type:
                        # Extract context
                        context = self._extract_context(
                            text, mention1['start'], mention2['end']
                        )
                        
                        relation = {
                            'source_entity': mention1['entity']['name'],
                            'source_type': mention1['type'],
                            'target_entity': mention2['entity']['name'],
                            'target_type': mention2['type'],
                            'relation_type': relation_type,
                            'confidence': 0.5,  # Lower confidence for co-occurrence
                            'context': context,
                            'source_documents': [document_id],
                            'extraction_method': 'cooccurrence',
                            'properties': {
                                'distance': mention2['start'] - mention1['end'],
                                'window_size': window_size
                            }
                        }
                        relations.append(relation)
        
        return relations
    
    def _extract_relations_from_satellite_info(
        self,
        satellite_info: Dict[str, Any],
        entities_by_type: Dict[str, List[Dict[str, Any]]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """Extract relations from satellite information."""
        relations = []
        
        satellites = satellite_info.get('satellites_mentioned', [])
        instruments = satellite_info.get('instruments_mentioned', [])
        parameters = satellite_info.get('parameters_mentioned', [])
        
        # Create HAS_INSTRUMENT relations
        for satellite_name in satellites:
            satellite_entity = self._find_entity_by_name(satellite_name, entities_by_type.get('satellite', []))
            
            if satellite_entity:
                for instrument_name in instruments:
                    instrument_entity = self._find_entity_by_name(instrument_name, entities_by_type.get('instrument', []))
                    
                    if instrument_entity:
                        relation = {
                            'source_entity': satellite_entity['name'],
                            'source_type': 'satellite',
                            'target_entity': instrument_entity['name'],
                            'target_type': 'instrument',
                            'relation_type': RelationTypeEnum.HAS_INSTRUMENT,
                            'confidence': 0.8,
                            'context': 'Satellite information section',
                            'source_documents': [document_id],
                            'extraction_method': 'satellite_info',
                            'properties': {}
                        }
                        relations.append(relation)
        
        # Create MEASURES relations
        for instrument_name in instruments:
            instrument_entity = self._find_entity_by_name(instrument_name, entities_by_type.get('instrument', []))
            
            if instrument_entity:
                for parameter_name in parameters:
                    parameter_entity = self._find_entity_by_name(parameter_name, entities_by_type.get('parameter', []))
                    
                    if parameter_entity:
                        relation = {
                            'source_entity': instrument_entity['name'],
                            'source_type': 'instrument',
                            'target_entity': parameter_entity['name'],
                            'target_type': 'parameter',
                            'relation_type': RelationTypeEnum.MEASURES,
                            'confidence': 0.8,
                            'context': 'Satellite information section',
                            'source_documents': [document_id],
                            'extraction_method': 'satellite_info',
                            'properties': {}
                        }
                        relations.append(relation)
        
        return relations
    
    def _infer_relation_type(self, source_type: str, target_type: str) -> Optional[RelationTypeEnum]:
        """Infer relation type based on entity types."""
        # Use ontology to get valid relations
        valid_relations = self.ontology.get_valid_relations(source_type, target_type)
        
        if valid_relations:
            # Return the most likely relation type
            # This is a simplified heuristic - in practice, you might use ML models
            return valid_relations[0]
        
        return None
    
    def _filter_and_validate_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and validate extracted relations."""
        filtered_relations = []
        seen_relations = set()
        
        for relation in relations:
            # Create a unique key for deduplication
            key = (
                relation['source_entity'],
                relation['target_entity'],
                relation['relation_type']
            )
            
            if key not in seen_relations:
                # Validate relation against ontology
                if self.ontology.validate_relation(
                    relation['relation_type'],
                    relation['source_type'],
                    relation['target_type']
                ):
                    seen_relations.add(key)
                    filtered_relations.append(relation)
        
        return filtered_relations
    
    def _calculate_relation_confidence(
        self,
        relations: List[Dict[str, Any]],
        text: str
    ) -> List[Dict[str, Any]]:
        """Calculate confidence scores for relations."""
        for relation in relations:
            base_confidence = relation.get('confidence', 0.5)
            
            # Adjust confidence based on extraction method
            method = relation.get('extraction_method', 'unknown')
            if method == 'pattern_matching':
                base_confidence *= 1.2
            elif method == 'table_extraction':
                base_confidence *= 1.1
            elif method == 'satellite_info':
                base_confidence *= 1.15
            elif method == 'cooccurrence':
                base_confidence *= 0.8
            
            # Adjust confidence based on context quality
            context = relation.get('context', '')
            if len(context) > 50:
                base_confidence *= 1.1
            
            # Adjust confidence based on entity frequency
            source_count = text.lower().count(relation['source_entity'].lower())
            target_count = text.lower().count(relation['target_entity'].lower())
            
            if source_count > 1 and target_count > 1:
                base_confidence *= 1.05
            
            # Ensure confidence is within bounds
            relation['confidence'] = min(base_confidence, 1.0)
        
        return relations
