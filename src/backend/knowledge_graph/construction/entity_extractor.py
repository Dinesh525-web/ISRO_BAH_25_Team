"""
Entity extractor for MOSDAC domain knowledge.
"""
import re
from typing import List, Dict, Any, Set
from datetime import datetime

from ..ontology.mosdac_ontology import MOSDACOntology
from app.data_ingestion.processors.text_processor import TextProcessor
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EntityExtractor:
    """Extract entities from MOSDAC documents."""
    
    def __init__(self):
        self.ontology = MOSDACOntology()
        self.text_processor = TextProcessor()
        
        # Entity patterns for MOSDAC domain
        self.entity_patterns = {
            'satellite': [
                r'(?:INSAT|insat)[-\s]?(\d+[A-Z]?)',
                r'(?:KALPANA|kalpana)[-\s]?(\d+)?',
                r'(?:MEGHA|megha)[-\s]?TROPIQUES?',
                r'(?:SCATSAT|scatsat)[-\s]?(\d+)?',
                r'(?:OCEANSAT|oceansat)[-\s]?(\d+)?',
                r'(?:CARTOSAT|cartosat)[-\s]?(\d+[A-Z]?)?',
                r'(?:RESOURCESAT|resourcesat)[-\s]?(\d+[A-Z]?)?',
                r'(?:RISAT|risat)[-\s]?(\d+[A-Z]?)?',
                r'(?:ASTROSAT|astrosat)',
                r'(?:CHANDRAYAAN|chandrayaan)[-\s]?(\d+)?',
                r'(?:MANGALYAAN|mangalyaan)',
                r'(?:ADITYA|aditya)[-\s]?L?(\d+)?'
            ],
            'instrument': [
                r'(?:VHRR|vhrr)',
                r'(?:CCD|ccd)',
                r'(?:LISS|liss)[-\s]?(\d+)?',
                r'(?:AWI|awi)',
                r'(?:MSMR|msmr)',
                r'(?:MADRAS|madras)',
                r'(?:SARAL|saral)',
                r'(?:ALTIKA|altika)',
                r'(?:SCATTEROMETER|scatterometer)',
                r'(?:RADIOMETER|radiometer)',
                r'(?:IMAGER|imager)',
                r'(?:SOUNDER|sounder)',
                r'(?:PAYLOAD|payload)',
                r'(?:SENSOR|sensor)',
                r'(?:DETECTOR|detector)',
                r'(?:SPECTRORADIOMETER|spectroradiometer)'
            ],
            'parameter': [
                r'(?:SST|sst|sea\s+surface\s+temperature)',
                r'(?:CHLOROPHYLL|chlorophyll)[-\s]?a?',
                r'(?:AEROSOL|aerosol)',
                r'(?:PRECIPITATION|precipitation)',
                r'(?:HUMIDITY|humidity)',
                r'(?:TEMPERATURE|temperature)',
                r'(?:PRESSURE|pressure)',
                r'(?:WIND|wind)\s+(?:speed|direction|vector)',
                r'(?:OZONE|ozone)',
                r'(?:RADIATION|radiation)',
                r'(?:REFLECTANCE|reflectance)',
                r'(?:BRIGHTNESS|brightness)\s+temperature',
                r'(?:NDVI|ndvi|normalized\s+difference\s+vegetation\s+index)',
                r'(?:ANOMALY|anomaly)',
                r'(?:SOIL\s+MOISTURE|soil\s+moisture)'
            ],
            'location': [
                r'(?:INDIA|india)',
                r'(?:INDIAN\s+OCEAN|indian\s+ocean)',
                r'(?:BAY\s+OF\s+BENGAL|bay\s+of\s+bengal)',
                r'(?:ARABIAN\s+SEA|arabian\s+sea)',
                r'(?:HIMALAYAS|himalayas)',
                r'(?:WESTERN\s+GHATS|western\s+ghats)',
                r'(?:GANGETIC\s+PLAINS|gangetic\s+plains)',
                r'(?:DECCAN\s+PLATEAU|deccan\s+plateau)',
                r'(?:MUMBAI|mumbai)',
                r'(?:DELHI|delhi)',
                r'(?:CHENNAI|chennai)',
                r'(?:KOLKATA|kolkata)',
                r'(?:BENGALURU|bengaluru)',
                r'(?:HYDERABAD|hyderabad)'
            ],
            'application': [
                r'(?:METEOROLOGY|meteorology)',
                r'(?:OCEANOGRAPHY|oceanography)',
                r'(?:AGRICULTURE|agriculture)',
                r'(?:FORESTRY|forestry)',
                r'(?:DISASTER\s+MANAGEMENT|disaster\s+management)',
                r'(?:CYCLONE|cyclone)\s+(?:tracking|monitoring|detection)',
                r'(?:DROUGHT|drought)\s+(?:monitoring|assessment)',
                r'(?:FLOOD|flood)\s+(?:monitoring|prediction)',
                r'(?:TSUNAMI|tsunami)\s+(?:warning|detection)',
                r'(?:CLIMATE|climate)\s+(?:monitoring|change)',
                r'(?:WEATHER|weather)\s+(?:forecasting|prediction)',
                r'(?:MONSOON|monsoon)\s+(?:monitoring|prediction)',
                r'(?:CROP|crop)\s+(?:monitoring|yield\s+estimation)',
                r'(?:WATER\s+RESOURCES|water\s+resources)'
            ],
            'organization': [
                r'(?:ISRO|isro)',
                r'(?:MOSDAC|mosdac)',
                r'(?:NRSC|nrsc)',
                r'(?:SAC|sac)',
                r'(?:VSSC|vssc)',
                r'(?:ISTRAC|istrac)',
                r'(?:LPSC|lpsc)',
                r'(?:IIRS|iirs)',
                r'(?:INCOIS|incois)',
                r'(?:IMD|imd)',
                r'(?:NCMRWF|ncmrwf)',
                r'(?:NIOT|niot)',
                r'(?:CMMACS|cmmacs)'
            ]
        }
    
    async def extract_entities(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from a document.
        
        Args:
            document: Processed document
            
        Returns:
            List of extracted entities
        """
        try:
            entities = []
            
            # Get document content
            content = document.get('main_content', '')
            title = document.get('title', '')
            
            # Combine title and content for entity extraction
            text = f"{title} {content}"
            
            # Extract entities using patterns
            pattern_entities = self._extract_entities_by_patterns(text)
            entities.extend(pattern_entities)
            
            # Extract entities from technical terms
            technical_terms = document.get('technical_terms', {})
            technical_entities = self._extract_entities_from_technical_terms(technical_terms)
            entities.extend(technical_entities)
            
            # Extract entities from satellite info
            satellite_info = document.get('satellite_info', {})
            satellite_entities = self._extract_entities_from_satellite_info(satellite_info)
            entities.extend(satellite_entities)
            
            # Extract named entities
            named_entities = document.get('named_entities', {})
            named_entity_objects = self._extract_entities_from_named_entities(named_entities)
            entities.extend(named_entity_objects)
            
            # Deduplicate entities
            entities = self._deduplicate_entities(entities)
            
            # Add document metadata
            for entity in entities:
                entity['source_documents'] = [document.get('id', document.get('url', ''))]
                entity['extraction_method'] = 'pattern_matching'
                entity['confidence'] = self._calculate_entity_confidence(entity, text)
            
            logger.debug(f"Extracted {len(entities)} entities from document")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def _extract_entities_by_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    entity_name = match.group(0).strip()
                    
                    # Skip very short or common words
                    if len(entity_name) < 3 or entity_name.lower() in ['the', 'and', 'for', 'with']:
                        continue
                    
                    entity = {
                        'name': entity_name,
                        'type': entity_type,
                        'description': self._generate_entity_description(entity_name, entity_type),
                        'aliases': self._generate_entity_aliases(entity_name, entity_type),
                        'properties': {
                            'pattern_match': pattern,
                            'match_position': match.start(),
                            'match_length': len(entity_name)
                        }
                    }
                    
                    entities.append(entity)
        
        return entities
    
    def _extract_entities_from_technical_terms(self, technical_terms: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract entities from technical terms."""
        entities = []
        
        for category, terms in technical_terms.items():
            for term in terms:
                entity = {
                    'name': term,
                    'type': category.rstrip('s'),  # Remove plural 's'
                    'description': self._generate_entity_description(term, category),
                    'aliases': [term.lower(), term.upper()],
                    'properties': {
                        'source_category': category,
                        'extraction_method': 'technical_terms'
                    }
                }
                entities.append(entity)
        
        return entities
    
    def _extract_entities_from_satellite_info(self, satellite_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract entities from satellite information."""
        entities = []
        
        # Extract satellites
        satellites = satellite_info.get('satellites_mentioned', [])
        for satellite in satellites:
            entity = {
                'name': satellite,
                'type': 'satellite',
                'description': f"Satellite mentioned in document: {satellite}",
                'aliases': [satellite.lower(), satellite.upper()],
                'properties': {
                    'extraction_method': 'satellite_info'
                }
            }
            entities.append(entity)
        
        # Extract instruments
        instruments = satellite_info.get('instruments_mentioned', [])
        for instrument in instruments:
            entity = {
                'name': instrument,
                'type': 'instrument',
                'description': f"Instrument mentioned in document: {instrument}",
                'aliases': [instrument.lower(), instrument.upper()],
                'properties': {
                    'extraction_method': 'satellite_info'
                }
            }
            entities.append(entity)
        
        # Extract parameters
        parameters = satellite_info.get('parameters_mentioned', [])
        for parameter in parameters:
            entity = {
                'name': parameter,
                'type': 'parameter',
                'description': f"Parameter mentioned in document: {parameter}",
                'aliases': [parameter.lower(), parameter.upper()],
                'properties': {
                    'extraction_method': 'satellite_info'
                }
            }
            entities.append(entity)
        
        return entities
    
    def _extract_entities_from_named_entities(self, named_entities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Extract entities from named entity recognition results."""
        entities = []
        
        # Map NER labels to our entity types
        ner_type_mapping = {
            'PERSON': 'person',
            'ORG': 'organization',
            'GPE': 'location',
            'LOCATION': 'location',
            'PRODUCT': 'product',
            'EVENT': 'event',
            'WORK_OF_ART': 'work_of_art',
            'LAW': 'law',
            'LANGUAGE': 'language',
            'DATE': 'date',
            'TIME': 'time',
            'PERCENT': 'percent',
            'MONEY': 'money',
            'QUANTITY': 'quantity',
            'ORDINAL': 'ordinal',
            'CARDINAL': 'cardinal'
        }
        
        for ner_type, entity_names in named_entities.items():
            entity_type = ner_type_mapping.get(ner_type, 'entity')
            
            for entity_name in entity_names:
                entity = {
                    'name': entity_name,
                    'type': entity_type,
                    'description': self._generate_entity_description(entity_name, entity_type),
                    'aliases': [entity_name.lower(), entity_name.upper()],
                    'properties': {
                        'ner_type': ner_type,
                        'extraction_method': 'named_entity_recognition'
                    }
                }
                entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate entities."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            # Create a unique key for the entity
            key = f"{entity['name'].lower()}:{entity['type']}"
            
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def _generate_entity_description(self, name: str, entity_type: str) -> str:
        """Generate description for an entity."""
        descriptions = {
            'satellite': f"Satellite system: {name}",
            'instrument': f"Scientific instrument: {name}",
            'parameter': f"Measurement parameter: {name}",
            'location': f"Geographic location: {name}",
            'application': f"Application domain: {name}",
            'organization': f"Organization: {name}",
            'person': f"Person: {name}",
            'product': f"Product: {name}",
            'event': f"Event: {name}",
        }
        
        return descriptions.get(entity_type, f"Entity of type {entity_type}: {name}")
    
    def _generate_entity_aliases(self, name: str, entity_type: str) -> List[str]:
        """Generate aliases for an entity."""
        aliases = [name.lower(), name.upper()]
        
        # Add common variations
        if entity_type == 'satellite':
            # Remove hyphens and spaces
            clean_name = re.sub(r'[-\s]', '', name)
            aliases.append(clean_name)
            aliases.append(clean_name.upper())
        
        return list(set(aliases))
    
    def _calculate_entity_confidence(self, entity: Dict[str, Any], text: str) -> float:
        """Calculate confidence score for an entity."""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for entities that appear multiple times
        entity_count = text.lower().count(entity['name'].lower())
        if entity_count > 1:
            confidence += min(0.1 * entity_count, 0.2)
        
        # Increase confidence for entities with specific patterns
        if entity['properties'].get('pattern_match'):
            confidence += 0.1
        
        # Increase confidence for technical terms
        if entity['properties'].get('extraction_method') == 'technical_terms':
            confidence += 0.1
        
        return min(confidence, 1.0)
