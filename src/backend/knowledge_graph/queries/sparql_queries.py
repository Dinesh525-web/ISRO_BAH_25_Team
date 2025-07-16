"""
SPARQL query implementations for RDF knowledge graph operations.
This module provides SPARQL query capabilities for when the knowledge graph
is stored in an RDF triplestore (like Apache Jena Fuseki, Blazegraph, etc.)
"""
import asyncio
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import time
import json
import aiohttp
from urllib.parse import urlencode

from ..ontology.schemas import (
    KGQuery, KGQueryResult, KGEntity, KGRelation, KGPath,
    EntityTypeEnum, RelationTypeEnum, KGStatistics
)
from ..ontology.mosdac_ontology import MOSDACOntology
from app.services.redis_client import RedisClient
from app.core.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SPARQLQueryEngine:
    """SPARQL query engine for RDF triplestore operations."""
    
    def __init__(self, sparql_endpoint: str, redis_client: Optional[RedisClient] = None):
        self.sparql_endpoint = sparql_endpoint
        self.redis_client = redis_client
        self.ontology = MOSDACOntology()
        self.cache_ttl = 300  # 5 minutes
        
        # SPARQL prefixes for MOSDAC ontology
        self.prefixes = {
            "mosdac": "http://mosdac.gov.in/ontology/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dc": "http://purl.org/dc/elements/1.1/",
            "dcterms": "http://purl.org/dc/terms/",
            "foaf": "http://xmlns.com/foaf/0.1/",
            "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
            "time": "http://www.w3.org/2006/time#"
        }
        
        # Query templates
        self.templates = {
            'find_entities': self._build_find_entities_template,
            'find_relations': self._build_find_relations_template,
            'find_paths': self._build_find_paths_template,
            'find_similar': self._build_find_similar_template,
            'subgraph_extraction': self._build_subgraph_template,
            'aggregation': self._build_aggregation_template,
            'domain_specific': self._build_domain_template,
            'reasoning': self._build_reasoning_template,
        }
    
    async def execute_query(self, query_type: str, parameters: Dict[str, Any]) -> KGQueryResult:
        """
        Execute a SPARQL query based on type and parameters.
        
        Args:
            query_type: Type of query to execute
            parameters: Query parameters
            
        Returns:
            Query result object
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query_type, parameters)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                logger.debug(f"Returning cached result for query type: {query_type}")
                return cached_result
            
            # Build and execute query
            sparql_query = self._build_query(query_type, parameters)
            
            # Execute query
            raw_results = await self._execute_sparql_query(sparql_query)
            
            # Process results
            result = await self._process_results(raw_results, query_type, start_time)
            
            # Cache result
            await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            return KGQueryResult(
                entities=[],
                relations=[],
                paths=[],
                total_count=0,
                query_time=time.time() - start_time,
                query_info={"error": str(e), "query_type": query_type}
            )
    
    def _build_query(self, query_type: str, parameters: Dict[str, Any]) -> str:
        """Build SPARQL query based on type and parameters."""
        if query_type not in self.templates:
            raise ValueError(f"Unsupported query type: {query_type}")
        
        template_builder = self.templates[query_type]
        return template_builder(parameters)
    
    def _get_prefixes(self) -> str:
        """Get SPARQL prefixes."""
        prefix_lines = []
        for prefix, uri in self.prefixes.items():
            prefix_lines.append(f"PREFIX {prefix}: <{uri}>")
        return "\n".join(prefix_lines)
    
    def _build_find_entities_template(self, params: Dict[str, Any]) -> str:
        """Build entity finding SPARQL query."""
        prefixes = self._get_prefixes()
        
        # Base query
        query_parts = [
            prefixes,
            "SELECT DISTINCT ?entity ?name ?type ?confidence ?description WHERE {",
            "  ?entity rdf:type mosdac:Entity .",
            "  ?entity mosdac:name ?name .",
            "  ?entity mosdac:type ?type .",
            "  OPTIONAL { ?entity mosdac:confidence ?confidence }",
            "  OPTIONAL { ?entity rdfs:comment ?description }"
        ]
        
        # Add filters
        filters = []
        
        # Entity type filter
        if "entity_type" in params:
            filters.append(f'  FILTER(?type = "{params["entity_type"]}")')
        
        # Name filter
        if "name" in params:
            if params.get("exact_match", False):
                filters.append(f'  FILTER(?name = "{params["name"]}")')
            else:
                filters.append(f'  FILTER(CONTAINS(LCASE(?name), LCASE("{params["name"]}")))') 
        
        # Confidence filter
        if "min_confidence" in params:
            filters.append(f'  FILTER(?confidence >= {params["min_confidence"]})')
        
        # Add filters to query
        query_parts.extend(filters)
        
        # Close WHERE clause
        query_parts.append("}")
        
        # Add ordering
        order_by = params.get("order_by", "name")
        order_desc = params.get("order_desc", False)
        query_parts.append(f"ORDER BY {'DESC' if order_desc else 'ASC'}(?{order_by})")
        
        # Add pagination
        offset = params.get("offset", 0)
        limit = params.get("limit", 10)
        if offset > 0:
            query_parts.append(f"OFFSET {offset}")
        query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    def _build_find_relations_template(self, params: Dict[str, Any]) -> str:
        """Build relation finding SPARQL query."""
        prefixes = self._get_prefixes()
        
        # Base query
        query_parts = [
            prefixes,
            "SELECT DISTINCT ?source ?target ?relation ?sourceName ?targetName ?relationType ?confidence WHERE {",
            "  ?source ?relation ?target .",
            "  ?source mosdac:name ?sourceName .",
            "  ?target mosdac:name ?targetName .",
            "  ?relation rdf:type mosdac:Relation .",
            "  ?relation mosdac:type ?relationType .",
            "  OPTIONAL { ?relation mosdac:confidence ?confidence }"
        ]
        
        # Add filters
        filters = []
        
        # Source entity filter
        if "source_entity" in params:
            filters.append(f'  FILTER(?sourceName = "{params["source_entity"]}")')
        
        # Target entity filter
        if "target_entity" in params:
            filters.append(f'  FILTER(?targetName = "{params["target_entity"]}")')
        
        # Relation type filter
        if "relation_type" in params:
            filters.append(f'  FILTER(?relationType = "{params["relation_type"]}")')
        
        # Entity type filters
        if "source_type" in params:
            filters.append(f'  ?source mosdac:type ?sourceType .')
            filters.append(f'  FILTER(?sourceType = "{params["source_type"]}")')
        
        if "target_type" in params:
            filters.append(f'  ?target mosdac:type ?targetType .')
            filters.append(f'  FILTER(?targetType = "{params["target_type"]}")')
        
        # Confidence filter
        if "min_confidence" in params:
            filters.append(f'  FILTER(?confidence >= {params["min_confidence"]})')
        
        # Add filters to query
        query_parts.extend(filters)
        
        # Close WHERE clause
        query_parts.append("}")
        
        # Add ordering
        query_parts.append("ORDER BY ?sourceName ?targetName")
        
        # Add pagination
        offset = params.get("offset", 0)
        limit = params.get("limit", 10)
        if offset > 0:
            query_parts.append(f"OFFSET {offset}")
        query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    def _build_find_paths_template(self, params: Dict[str, Any]) -> str:
        """Build path finding SPARQL query."""
        prefixes = self._get_prefixes()
        
        # Validate required parameters
        if "source_entity" not in params or "target_entity" not in params:
            raise ValueError("Source and target entities are required for path queries")
        
        # Property path with length constraints
        max_length = params.get("max_length", 5)
        path_expression = "/".join(["?p"] * max_length)
        
        query_parts = [
            prefixes,
            "SELECT DISTINCT ?source ?target ?path WHERE {",
            f'  ?source mosdac:name "{params["source_entity"]}" .',
            f'  ?target mosdac:name "{params["target_entity"]}" .',
            f"  ?source ({path_expression})* ?target .",
            "  BIND(CONCAT(str(?source), ' -> ', str(?target)) AS ?path)"
        ]
        
        # Add filters
        filters = []
        
        # Relation type filter
        if "relation_types" in params:
            relation_filter = " | ".join([f"mosdac:{rt}" for rt in params["relation_types"]])
            filters.append(f"  FILTER(?p IN ({relation_filter}))")
        
        # Add filters to query
        query_parts.extend(filters)
        
        # Close WHERE clause
        query_parts.append("}")
        
        # Add pagination
        limit = params.get("limit", 10)
        query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    def _build_find_similar_template(self, params: Dict[str, Any]) -> str:
        """Build similarity finding SPARQL query."""
        prefixes = self._get_prefixes()
        
        if "entity_name" not in params:
            raise ValueError("Entity name is required for similarity queries")
        
        # Similarity based on shared properties/relations
        query_parts = [
            prefixes,
            "SELECT DISTINCT ?similar ?similarName ?similarity WHERE {",
            f'  ?target mosdac:name "{params["entity_name"]}" .',
            "  ?target mosdac:type ?targetType .",
            "  ?similar mosdac:type ?targetType .",
            "  ?similar mosdac:name ?similarName .",
            f'  FILTER(?similarName != "{params["entity_name"]}")',
            "  {",
            "    SELECT ?similar (COUNT(DISTINCT ?sharedProp) AS ?sharedCount) WHERE {",
            "      ?target ?sharedProp ?value .",
            "      ?similar ?sharedProp ?value .",
            "    }",
            "    GROUP BY ?similar",
            "  }",
            "  BIND(?sharedCount / 10.0 AS ?similarity)"  # Normalize similarity
        ]
        
        # Minimum similarity threshold
        if "min_similarity" in params:
            query_parts.append(f"  FILTER(?similarity >= {params['min_similarity']})")
        
        # Close WHERE clause
        query_parts.append("}")
        
        # Add ordering
        query_parts.append("ORDER BY DESC(?similarity)")
        
        # Add pagination
        limit = params.get("limit", 10)
        query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    def _build_subgraph_template(self, params: Dict[str, Any]) -> str:
        """Build subgraph extraction SPARQL query."""
        prefixes = self._get_prefixes()
        
        if "center_entity" in params:
            # Ego subgraph
            radius = params.get("radius", 2)
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?node ?relation ?neighbor WHERE {",
                f'  ?center mosdac:name "{params["center_entity"]}" .',
                f"  ?center (mosdac:relatedTo){{{1},{radius}}} ?node .",
                "  OPTIONAL {",
                "    ?node ?relation ?neighbor .",
                "    ?neighbor (mosdac:relatedTo)* ?center .",
                "  }",
                "}"
            ]
        
        elif "entity_types" in params:
            # Subgraph by entity types
            entity_types = params["entity_types"]
            types_filter = " | ".join([f'"{t}"' for t in entity_types])
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?node ?relation ?neighbor WHERE {",
                "  ?node mosdac:type ?nodeType .",
                f"  FILTER(?nodeType IN ({types_filter}))",
                "  OPTIONAL {",
                "    ?node ?relation ?neighbor .",
                "    ?neighbor mosdac:type ?neighborType .",
                f"    FILTER(?neighborType IN ({types_filter}))",
                "  }",
                "}"
            ]
        
        else:
            # Full graph
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?node ?relation ?neighbor WHERE {",
                "  ?node rdf:type mosdac:Entity .",
                "  OPTIONAL {",
                "    ?node ?relation ?neighbor .",
                "    ?neighbor rdf:type mosdac:Entity .",
                "  }",
                "}"
            ]
        
        return "\n".join(query_parts)
    
    def _build_aggregation_template(self, params: Dict[str, Any]) -> str:
        """Build aggregation SPARQL query."""
        prefixes = self._get_prefixes()
        
        agg_type = params.get("aggregation_type", "count")
        group_by = params.get("group_by", "type")
        
        if agg_type == "count":
            if group_by == "type":
                query_parts = [
                    prefixes,
                    "SELECT ?entityType (COUNT(?entity) AS ?count) WHERE {",
                    "  ?entity rdf:type mosdac:Entity .",
                    "  ?entity mosdac:type ?entityType .",
                    "}",
                    "GROUP BY ?entityType",
                    "ORDER BY DESC(?count)"
                ]
            elif group_by == "relation_type":
                query_parts = [
                    prefixes,
                    "SELECT ?relationType (COUNT(?relation) AS ?count) WHERE {",
                    "  ?relation rdf:type mosdac:Relation .",
                    "  ?relation mosdac:type ?relationType .",
                    "}",
                    "GROUP BY ?relationType",
                    "ORDER BY DESC(?count)"
                ]
        
        elif agg_type == "avg_confidence":
            query_parts = [
                prefixes,
                "SELECT ?entityType (AVG(?confidence) AS ?avgConfidence) WHERE {",
                "  ?entity rdf:type mosdac:Entity .",
                "  ?entity mosdac:type ?entityType .",
                "  ?entity mosdac:confidence ?confidence .",
                "}",
                "GROUP BY ?entityType",
                "ORDER BY DESC(?avgConfidence)"
            ]
        
        # Add pagination
        limit = params.get("limit", 20)
        query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    def _build_domain_template(self, params: Dict[str, Any]) -> str:
        """Build domain-specific SPARQL queries for MOSDAC."""
        prefixes = self._get_prefixes()
        
        domain_type = params.get("domain_type", "satellite_instruments")
        
        if domain_type == "satellite_instruments":
            if "satellite_name" not in params:
                raise ValueError("Satellite name is required")
            
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?satellite ?instrument ?instrumentName WHERE {",
                f'  ?satellite mosdac:name "{params["satellite_name"]}" .',
                "  ?satellite mosdac:type 'satellite' .",
                "  ?satellite mosdac:hasInstrument ?instrument .",
                "  ?instrument mosdac:type 'instrument' .",
                "  ?instrument mosdac:name ?instrumentName .",
                "}",
                "ORDER BY ?instrumentName"
            ]
        
        elif domain_type == "instrument_parameters":
            if "instrument_name" not in params:
                raise ValueError("Instrument name is required")
            
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?instrument ?parameter ?parameterName WHERE {",
                f'  ?instrument mosdac:name "{params["instrument_name"]}" .',
                "  ?instrument mosdac:type 'instrument' .",
                "  ?instrument mosdac:measures ?parameter .",
                "  ?parameter mosdac:type 'parameter' .",
                "  ?parameter mosdac:name ?parameterName .",
                "}",
                "ORDER BY ?parameterName"
            ]
        
        elif domain_type == "satellite_coverage":
            if "satellite_name" not in params:
                raise ValueError("Satellite name is required")
            
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?satellite ?location ?locationName ?coverage WHERE {",
                f'  ?satellite mosdac:name "{params["satellite_name"]}" .',
                "  ?satellite mosdac:type 'satellite' .",
                "  ?satellite mosdac:observes ?location .",
                "  ?location mosdac:type 'location' .",
                "  ?location mosdac:name ?locationName .",
                "  OPTIONAL { ?satellite mosdac:coverage ?coverage }",
                "}",
                "ORDER BY ?locationName"
            ]
        
        return "\n".join(query_parts)
    
    def _build_reasoning_template(self, params: Dict[str, Any]) -> str:
        """Build reasoning SPARQL query with inference."""
        prefixes = self._get_prefixes()
        
        reasoning_type = params.get("reasoning_type", "transitive")
        
        if reasoning_type == "transitive":
            # Transitive reasoning (e.g., if A contains B and B contains C, then A contains C)
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?source ?target ?inferredRelation WHERE {",
                "  ?source mosdac:contains ?intermediate .",
                "  ?intermediate mosdac:contains ?target .",
                "  BIND(mosdac:contains AS ?inferredRelation)",
                "  # Ensure we don't already have this direct relation",
                "  FILTER NOT EXISTS { ?source mosdac:contains ?target }",
                "}"
            ]
        
        elif reasoning_type == "subsumption":
            # Subsumption reasoning based on entity hierarchy
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?entity ?type ?superType WHERE {",
                "  ?entity mosdac:type ?type .",
                "  ?type rdfs:subClassOf ?superType .",
                "  FILTER(?type != ?superType)",
                "}"
            ]
        
        elif reasoning_type == "inverse":
            # Inverse property reasoning
            query_parts = [
                prefixes,
                "SELECT DISTINCT ?source ?target ?inverseRelation WHERE {",
                "  ?target mosdac:hasInstrument ?source .",
                "  BIND(mosdac:instrumentOf AS ?inverseRelation)",
                "}"
            ]
        
        return "\n".join(query_parts)
    
    async def _execute_sparql_query(self, sparql_query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against the endpoint."""
        try:
            # Prepare query parameters
            params = {
                'query': sparql_query,
                'format': 'application/sparql-results+json'
            }
            
            # Make HTTP request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.sparql_endpoint,
                    data=urlencode(params),
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('results', {}).get('bindings', [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"SPARQL query failed: {response.status} - {error_text}")
        
        except Exception as e:
            logger.error(f"Error executing SPARQL query: {e}")
            raise
    
    async def _process_results(
        self,
        raw_results: List[Dict[str, Any]],
        query_type: str,
        start_time: float
    ) -> KGQueryResult:
        """Process raw SPARQL results."""
        entities = []
        relations = []
        paths = []
        
        # Process SPARQL results
        for binding in raw_results:
            # Extract entities
            if 'entity' in binding:
                entity_data = {
                    'id': binding['entity']['value'],
                    'name': binding.get('name', {}).get('value', ''),
                    'type': binding.get('type', {}).get('value', ''),
                    'confidence': float(binding.get('confidence', {}).get('value', 1.0))
                }
                
                entity = self._convert_sparql_to_entity(entity_data)
                if entity not in entities:
                    entities.append(entity)
            
            # Extract relations
            if 'source' in binding and 'target' in binding:
                relation_data = {
                    'source': binding['source']['value'],
                    'target': binding['target']['value'],
                    'relation_type': binding.get('relationType', {}).get('value', ''),
                    'confidence': float(binding.get('confidence', {}).get('value', 1.0))
                }
                
                relation = self._convert_sparql_to_relation(relation_data)
                if relation not in relations:
                    relations.append(relation)
        
        return KGQueryResult(
            entities=entities,
            relations=relations,
            paths=paths,
            total_count=len(entities) + len(relations) + len(paths),
            query_time=time.time() - start_time,
            query_info={"query_type": query_type, "endpoint": self.sparql_endpoint}
        )
    
    def _convert_sparql_to_entity(self, entity_data: Dict[str, Any]) -> KGEntity:
        """Convert SPARQL result to KGEntity."""
        from ..ontology.schemas import EntityProperties
        
        return KGEntity(
            id=entity_data.get('id', ''),
            name=entity_data.get('name', ''),
            type=entity_data.get('type', 'entity'),
            properties=EntityProperties(),
            confidence=entity_data.get('confidence', 1.0)
        )
    
    def _convert_sparql_to_relation(self, relation_data: Dict[str, Any]) -> KGRelation:
        """Convert SPARQL result to KGRelation."""
        from ..ontology.schemas import RelationProperties
        
        return KGRelation(
            source_entity_id=relation_data.get('source', ''),
            target_entity_id=relation_data.get('target', ''),
            relation_type=relation_data.get('relation_type', 'related'),
            properties=RelationProperties(),
            confidence=relation_data.get('confidence', 1.0)
        )
    
    def _generate_cache_key(self, query_type: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for query."""
        import hashlib
        
        query_str = json.dumps({
            "query_type": query_type,
            "parameters": parameters,
            "endpoint": self.sparql_endpoint
        }, sort_keys=True)
        
        return f"sparql_query:{hashlib.md5(query_str.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[KGQueryResult]:
        """Get cached result."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                return KGQueryResult(**cached_data)
        except Exception as e:
            logger.warning(f"Error getting cached result: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: KGQueryResult):
        """Cache query result."""
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.set(cache_key, result.dict(), ttl=self.cache_ttl)
        except Exception as e:
            logger.warning(f"Error caching result: {e}")
    
    async def update_knowledge_graph(self, triples: List[Tuple[str, str, str]]):
        """Update knowledge graph with new triples."""
        try:
            # Build SPARQL UPDATE query
            insert_triples = []
            for subject, predicate, obj in triples:
                insert_triples.append(f"<{subject}> <{predicate}> <{obj}> .")
            
            update_query = f"""
            {self._get_prefixes()}
            INSERT DATA {{
                {' '.join(insert_triples)}
            }}
            """
            
            # Execute update
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.sparql_endpoint,
                    data={'update': update_query},
                    headers={'Content-Type': 'application/x-www-form-urlencoded'}
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"SPARQL update failed: {response.status} - {error_text}")
            
            logger.info(f"Successfully updated knowledge graph with {len(triples)} triples")
            
        except Exception as e:
            logger.error(f"Error updating knowledge graph: {e}")
            raise


# Query execution utilities
class SPARQLQueryBuilder:
    """Utility class for building SPARQL queries."""
    
    @staticmethod
    def build_select_query(
        select_vars: List[str],
        where_patterns: List[str],
        prefixes: Dict[str, str],
        filters: Optional[List[str]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> str:
        """Build a SELECT SPARQL query."""
        query_parts = []
        
        # Add prefixes
        for prefix, uri in prefixes.items():
            query_parts.append(f"PREFIX {prefix}: <{uri}>")
        
        # Add SELECT clause
        query_parts.append(f"SELECT {' '.join(select_vars)}")
        
        # Add WHERE clause
        query_parts.append("WHERE {")
        query_parts.extend([f"  {pattern}" for pattern in where_patterns])
        
        # Add filters
        if filters:
            query_parts.extend([f"  FILTER({filter_expr})" for filter_expr in filters])
        
        query_parts.append("}")
        
        # Add ORDER BY
        if order_by:
            query_parts.append(f"ORDER BY {order_by}")
        
        # Add pagination
        if offset:
            query_parts.append(f"OFFSET {offset}")
        if limit:
            query_parts.append(f"LIMIT {limit}")
        
        return "\n".join(query_parts)
    
    @staticmethod
    def build_construct_query(
        construct_template: List[str],
        where_patterns: List[str],
        prefixes: Dict[str, str],
        filters: Optional[List[str]] = None
    ) -> str:
        """Build a CONSTRUCT SPARQL query."""
        query_parts = []
        
        # Add prefixes
        for prefix, uri in prefixes.items():
            query_parts.append(f"PREFIX {prefix}: <{uri}>")
        
        # Add CONSTRUCT clause
        query_parts.append("CONSTRUCT {")
        query_parts.extend([f"  {template}" for template in construct_template])
        query_parts.append("}")
        
        # Add WHERE clause
        query_parts.append("WHERE {")
        query_parts.extend([f"  {pattern}" for pattern in where_patterns])
        
        # Add filters
        if filters:
            query_parts.extend([f"  FILTER({filter_expr})" for filter_expr in filters])
        
        query_parts.append("}")
        
        return "\n".join(query_parts)
    
    @staticmethod
    def build_ask_query(
        where_patterns: List[str],
        prefixes: Dict[str, str],
        filters: Optional[List[str]] = None
    ) -> str:
        """Build an ASK SPARQL query."""
        query_parts = []
        
        # Add prefixes
        for prefix, uri in prefixes.items():
            query_parts.append(f"PREFIX {prefix}: <{uri}>")
        
        # Add ASK clause
        query_parts.append("ASK {")
        query_parts.extend([f"  {pattern}" for pattern in where_patterns])
        
        # Add filters
        if filters:
            query_parts.extend([f"  FILTER({filter_expr})" for filter_expr in filters])
        
        query_parts.append("}")
        
        return "\n".join(query_parts)
