"""
Fixed LangChain GraphRAG Implementation
Addresses both import deprecation warnings and Pydantic field validation errors
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

# Fixed imports - using the new package structure
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Updated imports for the reorganized packages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # This is the correct import path now
from langchain_community.vectorstores import Chroma     # This is the correct import path now
from langchain_google_genai import ChatGoogleGenerativeAI

# LangGraph imports for workflow orchestration
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Pydantic for tool schemas - with proper field definitions
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = ""
GOOGLE_API_KEY = "USE_YOUR_OWN"

class GraphRAGState(TypedDict):
    """
    State object for LangGraph workflow
    This defines what information flows between different steps in our recommendation process
    """
    query: str  # Original user query
    item_name: str  # Extracted item name
    graph_recommendations: List[Dict]  # Results from graph analysis
    semantic_recommendations: List[Dict]  # Results from vector similarity
    recommendation_strategy: str  # Which approach to use
    final_response: str  # Generated recommendation text
    messages: List[str]  # For debugging and logging

class GraphRecommendationInput(BaseModel):
    """Schema for graph recommendation tool input"""
    item_name: str = Field(description="Name of the menu item to find recommendations for")
    top_k: int = Field(default=5, description="Number of recommendations to return")

class SemanticRecommendationInput(BaseModel):
    """Schema for semantic recommendation tool input"""
    item_name: str = Field(description="Name of the menu item to find similar items for")
    top_k: int = Field(default=5, description="Number of similar items to return")

class GraphRecommendationTool(BaseTool):
    """
    Fixed version of GraphRecommendationTool that properly handles Pydantic validation
    
    The key insight: we need to define the neo4j_driver as a class attribute
    and properly initialize it within the Pydantic model structure
    """
    name = "graph_recommendations"
    description = """
    Find novel food pairings using advanced graph co-occurrence analysis.
    This tool analyzes customer ordering patterns to find items that appear together 
    with 15-40% frequency, creating surprising but proven combinations.
    Includes cross-category and cross-cuisine bonuses for truly novel discoveries.
    """
    args_schema = GraphRecommendationInput
    
    # This is the crucial fix: we define neo4j_driver as a proper field
    # But we need to be careful about how Pydantic handles this
    class Config:
        arbitrary_types_allowed = True  # This allows non-Pydantic types like Neo4j driver
    
    def __init__(self, neo4j_driver, **kwargs):
        # Initialize the parent class first
        super().__init__(**kwargs)
        # Now we can safely assign the driver
        # Using object.__setattr__ bypasses Pydantic's field validation
        object.__setattr__(self, 'neo4j_driver', neo4j_driver)
    
    def _run(
        self, 
        item_name: str, 
        top_k: int = 5,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        Execute your existing sophisticated graph recommendation logic
        Now with proper access to the neo4j driver
        """
        try:
            with self.neo4j_driver.session() as session:
                # Your sophisticated co-occurrence algorithm with novelty scoring
                result = session.run("""
                    MATCH (item:Item {name: $item_name})
                    OPTIONAL MATCH (item)-[co:CO_OCCURS]-(other:Item)
                    WHERE co.rate >= 0.15 AND co.rate <= 0.40
                    WITH item, other, co
                    ORDER BY co.rate DESC
                    LIMIT $limit
                    RETURN 
                        other.name as recommended_item,
                        other.category as category,
                        co.rate as cooccurrence_rate,
                        co.count as times_ordered_together,
                        CASE 
                            WHEN item.category <> other.category THEN 2.0
                            ELSE 1.0
                        END as cross_category_bonus,
                        CASE
                            WHEN item.cuisine_type <> other.cuisine_type THEN 1.5
                            ELSE 1.0
                        END as cross_cuisine_bonus
                """, item_name=item_name, limit=top_k * 2)
                
                recommendations = []
                for record in result:
                    if record['recommended_item']:
                        # Your novel novelty scoring algorithm - preserved exactly
                        novelty_score = (
                            record['cooccurrence_rate'] * 
                            record['cross_category_bonus'] * 
                            record['cross_cuisine_bonus']
                        )
                        
                        recommendations.append({
                            'item': record['recommended_item'],
                            'category': record['category'],
                            'cooccurrence_rate': record['cooccurrence_rate'],
                            'novelty_score': novelty_score,
                            'times_ordered_together': record['times_ordered_together']
                        })
                
                # Sort by novelty score and return top K - your proven approach
                recommendations.sort(key=lambda x: x['novelty_score'], reverse=True)
                top_recommendations = recommendations[:top_k]
                
                if not top_recommendations:
                    return f"No novel graph-based recommendations found for {item_name}"
                
                # Format results for LangChain consumption while preserving detail
                result_text = f"Graph-based novel pairings for {item_name}:\n"
                for i, rec in enumerate(top_recommendations, 1):
                    result_text += f"{i}. {rec['item']} ({rec['category']}) - "
                    result_text += f"Novelty: {rec['novelty_score']:.2f}, "
                    result_text += f"Co-occurrence: {int(rec['cooccurrence_rate']*100)}%, "
                    result_text += f"Ordered together: {rec['times_ordered_together']} times\n"
                
                return result_text
                
        except Exception as e:
            logger.error(f"Graph recommendation error: {e}")
            return f"Unable to generate graph recommendations for {item_name}: {str(e)}"
    
    async def _arun(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        """Async version - delegates to sync for now"""
        return self._run(item_name, top_k, run_manager)

class SemanticRecommendationTool(BaseTool):
    """
    Fixed version of SemanticRecommendationTool with proper Pydantic handling
    """
    name = "semantic_recommendations"
    description = """
    Find menu items that are semantically similar based on flavor profiles, 
    ingredients, cooking methods, and culinary characteristics. Uses advanced 
    vector embeddings to identify items that share similar essence even if 
    they don't frequently appear together in orders.
    """
    args_schema = SemanticRecommendationInput
    
    class Config:
        arbitrary_types_allowed = True  # Allow vector store and retriever objects
    
    def __init__(self, vector_store, retriever, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation for these complex objects
        object.__setattr__(self, 'vector_store', vector_store)
        object.__setattr__(self, 'retriever', retriever)
    
    def _run(
        self, 
        item_name: str, 
        top_k: int = 5,
        run_manager: Optional[Any] = None
    ) -> str:
        """
        Find semantically similar items using your vector embeddings approach
        """
        try:
            # Create a rich query that captures the semantic search intent
            query = f"Find dishes similar to {item_name} in flavor profile, texture, ingredients, and cooking style"
            
            # Use LangChain's retriever to find similar documents
            similar_docs = self.vector_store.similarity_search(query,top_k=10)
            
            recommendations = []
            seen_items=set()
            
            for doc in similar_docs:
                # Skip self-matches and ensure we have valid metadata
                item_name_from_doc = doc.metadata.get('item_name')
                if (doc.metadata.get('item_name') and 
                    doc.metadata.get('item_name') != item_name and 
                    item_name_from_doc not in seen_items):# new check for duplicates
                    recommendations.append({
                        'item': doc.metadata.get('item_name'),
                        'category': doc.metadata.get('category', 'Unknown'),
                        'cuisine_type': doc.metadata.get('cuisine_type', 'Unknown'),
                        'similarity_reason': 'semantic_similarity'
                    })
                    seen_items.add(item_name_from_doc)
                    if len(recommendations)>= top_k:
                     break
            
            # Limit to requested number
            recommendations = recommendations[:top_k]
            
            if not recommendations:
                return f"No semantically similar items found for {item_name}"
            
            # Format results with rich context
            result_text = f"Semantically similar items to {item_name}:\n"
            for i, rec in enumerate(recommendations, 1):
                result_text += f"{i}. {rec['item']} ({rec['category']}) - "
                result_text += f"Similar flavor/texture profile ({rec['cuisine_type']} cuisine)\n"
            
            return result_text
            
        except Exception as e:
            logger.error(f"Semantic recommendation error: {e}")
            return f"Unable to generate semantic recommendations for {item_name}: {str(e)}"
    
    async def _arun(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        return self._run(item_name, top_k, run_manager)

class LangChainGraphRAGProcessor:
    """
    Complete LangChain-based GraphRAG processor with fixed tool initialization
    """
    
    def __init__(self):
        logger.info("Initializing LangChain GraphRAG Processor...")
        
        # Initialize Neo4j connection (unchanged from your current approach)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Initialize LangChain components with CORRECT imports
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize LLM with your current Gemini setup
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
        )
        
        # Initialize vector store and tools as None - will be set up later
        self.vector_store = None
        self.retriever = None
        self.graph_tool = None
        self.semantic_tool = None
        self.workflow = None
        
        logger.info("✓ Core components initialized")
    
    def load_neo4j_data_as_documents(self) -> List[Document]:
        """
        Convert your Neo4j graph data into LangChain Documents
        """
        logger.info("Loading Neo4j data and converting to LangChain Documents...")
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (i:Item)
                RETURN i.name as name, 
                       i.category as category,
                       i.ingredients as ingredients,
                       i.flavor_profile as flavor,
                       i.texture as texture,
                       i.spice_level as spice_level,
                       i.occasion as occasion,
                       i.complementary_items as complementary,
                       i.cuisine_type as cuisine_type
            """)
            
            documents = []
            for record in result:
                page_content = f"""
                {record['name']} is a {record['category']} dish from {record['cuisine_type']} cuisine.
                Ingredients: {record['ingredients']}
                Flavor Profile: {record['flavor']}
                Texture: {record['texture']}
                Spice Level: {record['spice_level']}/10
                Best for: {record['occasion']}
                Goes well with: {', '.join(record['complementary'] or [])}
                """.strip()
                
                metadata = {
                    "item_name": record['name'],
                    "category": record['category'],
                    "spice_level": record['spice_level'],
                    "cuisine_type": record['cuisine_type'],
                    "source": "neo4j_graph",
                    "has_ingredients": bool(record['ingredients']),
                    "has_flavor_profile": bool(record['flavor'])
                }
                
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"✓ Created {len(documents)} LangChain Documents from Neo4j data")
            return documents
    
    def create_vector_index(self, documents: List[Document]):
        """
        Create vector embeddings and storage using ChromaDB
        """
        logger.info("Creating vector embeddings and ChromaDB index...")
        
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chromadb_storage",
            collection_name="menu_items"
        )
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,
                "score_threshold": 0.7
            }
        )
        
        logger.info("✓ Vector index and retriever created successfully")
        return self.vector_store
    
    def initialize_tools(self):
        """
        Initialize tools with proper Pydantic handling - this is the key fix
        """
        logger.info("Initializing recommendation tools...")
        
        if not self.vector_store:
            raise ValueError("Vector store must be created before initializing tools")
        
        # Create tools using the fixed constructors
        self.graph_tool = GraphRecommendationTool(self.driver)
        self.semantic_tool = SemanticRecommendationTool(self.vector_store, self.retriever)
        
        logger.info("✓ Graph and semantic recommendation tools initialized")
    
    # The rest of your methods remain the same...
    def extract_item_name(self, query: str) -> Optional[str]:
        """Enhanced item name extraction from user queries"""
        query_lower = query.lower().strip()
        
        patterns = [
            r"what (?:goes|pairs) with (.+?)(?:\?|$)",
            r"what should i (?:eat|order|have) with (.+?)(?:\?|$)",
            r"recommend (?:something|items) (?:for|to go with) (.+?)(?:\?|$)",
            r"what are (?:some )?(?:good|novel|surprising) pairings? for (.+?)(?:\?|$)",
            r"suggest (?:something|items) (?:for|with) (.+?)(?:\?|$)",
            r"find (?:recommendations|pairings) for (.+?)(?:\?|$)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                item_name = match.group(1).strip()
                item_name = item_name.replace("?", "").strip()
                return " ".join(word.capitalize() for word in item_name.split())
        
        return None
    
    def setup_system(self, data_path: str = None):
        """Complete system setup - call this once to initialize everything"""
        logger.info("Setting up complete LangChain GraphRAG system...")
        
        documents = self.load_neo4j_data_as_documents()
        self.create_vector_index(documents)
        self.initialize_tools()
        
        logger.info("✓ LangChain GraphRAG system setup complete!")
        return self
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()
        logger.info("✓ Resources cleaned up")

# Test the fixed implementation
def test_fixed_implementation():
    """Test that the fixed tools work correctly"""
    print("Testing Fixed GraphRAG Implementation")
    print("=" * 50)
    
    try:
        processor = LangChainGraphRAGProcessor()
        processor.setup_system()
        
        print("✓ System initialization successful!")
        print("✓ Tools created without Pydantic errors")
        print("✓ Ready for food recommendation queries")
        
        return processor
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        return None

if __name__ == "__main__":
    processor = test_fixed_implementation()
    if processor:
        processor.close()
