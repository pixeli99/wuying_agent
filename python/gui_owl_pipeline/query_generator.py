import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import random
from enum import Enum
import networkx as nx
from .config import Platform

logger = logging.getLogger(__name__)

@dataclass
class Page:
    id: str
    name: str
    description: str
    elements: List[Dict[str, Any]]
    metadata: Dict[str, Any] = None

@dataclass
class PageTransition:
    from_page: str
    to_page: str
    action: str
    condition: Optional[str] = None

@dataclass
class Query:
    id: str
    platform: Platform
    instruction: str
    natural_instruction: str
    path: List[str]
    slots: Dict[str, Any]
    metadata: Dict[str, Any] = None
    difficulty: str = "medium"

class QueryType(Enum):
    NAVIGATION = "navigation"
    FORM_FILLING = "form_filling"
    SEARCH = "search"
    COMPLEX_TASK = "complex_task"
    ATOMIC_OPERATION = "atomic_operation"

class QueryGenerator:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.dag_graphs: Dict[str, nx.DiGraph] = {}
        self.page_metadata: Dict[str, Page] = {}
        self.query_templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[QueryType, List[str]]:
        return {
            QueryType.NAVIGATION: [
                "Navigate to {target_page} in {app_name}",
                "Go to the {target_page} section",
                "Open {app_name} and find {target_page}"
            ],
            QueryType.FORM_FILLING: [
                "Fill out the {form_name} with {data_description}",
                "Complete the registration form with {user_details}",
                "Enter {field_value} in the {field_name} field"
            ],
            QueryType.SEARCH: [
                "Search for {query} in {app_name}",
                "Find {item_name} using the search function",
                "Look up {search_term} and select the first result"
            ],
            QueryType.COMPLEX_TASK: [
                "Book a {service_type} for {date} at {time}",
                "Purchase {product} with {payment_method}",
                "Set up {feature} with {configuration}"
            ],
            QueryType.ATOMIC_OPERATION: [
                "Double-click on {element}",
                "Drag {source} to {destination}",
                "Type {text} in the {field}"
            ]
        }
        
    def build_dag_graph(self, pages: List[Page], transitions: List[PageTransition]) -> nx.DiGraph:
        G = nx.DiGraph()
        
        for page in pages:
            G.add_node(page.id, data=page)
            self.page_metadata[page.id] = page
            
        for transition in transitions:
            G.add_edge(
                transition.from_page,
                transition.to_page,
                action=transition.action,
                condition=transition.condition
            )
            
        return G
        
    def generate_mobile_queries(self, app_name: str, dag: nx.DiGraph, count: int = 10) -> List[Query]:
        queries = []
        
        for i in range(count):
            path = self._sample_path(dag)
            
            if not path:
                continue
                
            metadata = self._extract_path_metadata(path, dag)
            
            slots = self._extract_slots(metadata)
            
            raw_instruction = self._generate_raw_instruction(path, metadata, slots)
            
            natural_instruction = self._naturalize_instruction(raw_instruction, app_name)
            
            query = Query(
                id=f"mobile_query_{i}",
                platform=Platform.MOBILE,
                instruction=raw_instruction,
                natural_instruction=natural_instruction,
                path=[node for node in path],
                slots=slots,
                metadata={
                    "app_name": app_name,
                    "path_length": len(path),
                    "query_type": self._classify_query(metadata).value
                },
                difficulty=self._assess_difficulty(path, metadata)
            )
            
            queries.append(query)
            logger.info(f"Generated mobile query {i}: {natural_instruction[:50]}...")
            
        return queries
        
    def generate_pc_queries(self, software_name: str, skills: List[Dict], count: int = 10) -> List[Query]:
        queries = []
        
        for i in range(count):
            skill_chain = self._sample_skill_chain(skills)
            
            if not skill_chain:
                continue
                
            requires_object = self._check_object_requirement(skill_chain)
            
            raw_instruction = self._generate_skill_instruction(skill_chain, requires_object)
            
            natural_instruction = self._naturalize_pc_instruction(raw_instruction, software_name)
            
            query = Query(
                id=f"pc_query_{i}",
                platform=Platform.PC,
                instruction=raw_instruction,
                natural_instruction=natural_instruction,
                path=[skill["name"] for skill in skill_chain],
                slots={"requires_object": requires_object},
                metadata={
                    "software_name": software_name,
                    "skill_count": len(skill_chain),
                    "query_type": QueryType.ATOMIC_OPERATION.value
                },
                difficulty=self._assess_pc_difficulty(skill_chain)
            )
            
            queries.append(query)
            logger.info(f"Generated PC query {i}: {natural_instruction[:50]}...")
            
        return queries
        
    def _sample_path(self, dag: nx.DiGraph) -> List[str]:
        if not dag.nodes():
            return []
            
        start_nodes = [n for n in dag.nodes() if dag.in_degree(n) == 0]
        if not start_nodes:
            start_nodes = list(dag.nodes())
            
        start = random.choice(start_nodes)
        
        path = [start]
        current = start
        max_length = 10
        
        while len(path) < max_length:
            neighbors = list(dag.successors(current))
            if not neighbors:
                break
                
            next_node = random.choice(neighbors)
            path.append(next_node)
            current = next_node
            
        return path
        
    def _extract_path_metadata(self, path: List[str], dag: nx.DiGraph) -> Dict[str, Any]:
        metadata = {
            "pages": [],
            "transitions": []
        }
        
        for node in path:
            if node in self.page_metadata:
                page = self.page_metadata[node]
                metadata["pages"].append({
                    "id": page.id,
                    "name": page.name,
                    "description": page.description
                })
                
        for i in range(len(path) - 1):
            if dag.has_edge(path[i], path[i+1]):
                edge_data = dag.get_edge_data(path[i], path[i+1])
                metadata["transitions"].append(edge_data)
                
        return metadata
        
    def _extract_slots(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        slots = {}
        
        for page in metadata.get("pages", []):
            if "form" in page.get("description", "").lower():
                slots["form_fields"] = self._generate_form_fields()
            if "search" in page.get("description", "").lower():
                slots["search_query"] = self._generate_search_query()
                
        return slots
        
    def _generate_form_fields(self) -> Dict[str, str]:
        return {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "555-0123",
            "address": "123 Main St"
        }
        
    def _generate_search_query(self) -> str:
        queries = [
            "restaurants near me",
            "best coffee shops",
            "electronics store",
            "movie tickets"
        ]
        return random.choice(queries)
        
    def _generate_raw_instruction(self, path: List[str], metadata: Dict, slots: Dict) -> str:
        pages = metadata.get("pages", [])
        if not pages:
            return "Navigate through the application"
            
        target = pages[-1]["name"] if pages else "destination"
        
        instruction_parts = []
        
        for i, page in enumerate(pages):
            if i == 0:
                instruction_parts.append(f"Open {page['name']}")
            else:
                instruction_parts.append(f"Navigate to {page['name']}")
                
            if "form_fields" in slots and "form" in page.get("description", "").lower():
                for field, value in slots["form_fields"].items():
                    instruction_parts.append(f"Enter '{value}' in {field}")
                    
            if "search_query" in slots and "search" in page.get("description", "").lower():
                instruction_parts.append(f"Search for '{slots['search_query']}'")
                
        return " -> ".join(instruction_parts)
        
    def _naturalize_instruction(self, raw_instruction: str, app_name: str) -> str:
        if self.llm_client:
            prompt = f"""
            Convert this technical instruction into natural language:
            Technical: {raw_instruction}
            App: {app_name}
            
            Make it sound like a natural user request. Keep it concise.
            """
            
            return raw_instruction
            
        parts = raw_instruction.split(" -> ")
        if len(parts) > 2:
            natural = f"In {app_name}, {parts[0].lower()} and then {parts[-1].lower()}"
        else:
            natural = f"Please {raw_instruction.lower()} in {app_name}"
            
        return natural
        
    def _sample_skill_chain(self, skills: List[Dict]) -> List[Dict]:
        if not skills:
            return []
            
        chain_length = random.randint(1, min(5, len(skills)))
        
        return random.sample(skills, chain_length)
        
    def _check_object_requirement(self, skill_chain: List[Dict]) -> bool:
        for skill in skill_chain:
            if skill.get("type") in ["drag", "select", "modify"]:
                return True
        return False
        
    def _generate_skill_instruction(self, skill_chain: List[Dict], requires_object: bool) -> str:
        instructions = []
        
        if requires_object:
            instructions.append("Select target object")
            
        for skill in skill_chain:
            skill_name = skill.get("name", "action")
            skill_type = skill.get("type", "click")
            
            if skill_type == "click":
                instructions.append(f"Click on {skill_name}")
            elif skill_type == "type":
                instructions.append(f"Type text in {skill_name}")
            elif skill_type == "drag":
                instructions.append(f"Drag {skill_name} to destination")
            else:
                instructions.append(f"Perform {skill_name}")
                
        return " -> ".join(instructions)
        
    def _naturalize_pc_instruction(self, raw_instruction: str, software_name: str) -> str:
        parts = raw_instruction.split(" -> ")
        
        if len(parts) == 1:
            return f"In {software_name}, {parts[0].lower()}"
        elif len(parts) == 2:
            return f"Using {software_name}, {parts[0].lower()} and then {parts[1].lower()}"
        else:
            first = parts[0].lower()
            last = parts[-1].lower()
            return f"In {software_name}, {first}, perform several steps, and finally {last}"
            
    def _classify_query(self, metadata: Dict) -> QueryType:
        pages = metadata.get("pages", [])
        
        if not pages:
            return QueryType.NAVIGATION
            
        descriptions = " ".join([p.get("description", "") for p in pages]).lower()
        
        if "form" in descriptions or "registration" in descriptions:
            return QueryType.FORM_FILLING
        elif "search" in descriptions:
            return QueryType.SEARCH
        elif len(pages) > 3:
            return QueryType.COMPLEX_TASK
        else:
            return QueryType.NAVIGATION
            
    def _assess_difficulty(self, path: List[str], metadata: Dict) -> str:
        path_length = len(path)
        has_forms = any("form" in str(p).lower() for p in metadata.get("pages", []))
        has_conditions = any(t.get("condition") for t in metadata.get("transitions", []))
        
        score = path_length
        if has_forms:
            score += 2
        if has_conditions:
            score += 3
            
        if score <= 3:
            return "easy"
        elif score <= 7:
            return "medium"
        else:
            return "hard"
            
    def _assess_pc_difficulty(self, skill_chain: List[Dict]) -> str:
        chain_length = len(skill_chain)
        has_complex = any(s.get("type") in ["drag", "script"] for s in skill_chain)
        
        if chain_length <= 2 and not has_complex:
            return "easy"
        elif chain_length <= 4:
            return "medium"
        else:
            return "hard"
            
    def validate_query(self, query: Query) -> bool:
        if not query.instruction or not query.natural_instruction:
            logger.warning(f"Query {query.id} missing instructions")
            return False
            
        if not query.path:
            logger.warning(f"Query {query.id} has empty path")
            return False
            
        if query.platform == Platform.MOBILE and len(query.path) > 20:
            logger.warning(f"Query {query.id} path too long: {len(query.path)}")
            return False
            
        return True
        
    def export_queries(self, queries: List[Query], output_file: str):
        data = []
        for query in queries:
            data.append({
                "id": query.id,
                "platform": query.platform.value,
                "instruction": query.instruction,
                "natural_instruction": query.natural_instruction,
                "path": query.path,
                "slots": query.slots,
                "metadata": query.metadata,
                "difficulty": query.difficulty
            })
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported {len(queries)} queries to {output_file}")