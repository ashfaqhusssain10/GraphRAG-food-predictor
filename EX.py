"""
LangChain GraphRAG Implementation - Version Compatible Edition
A complete transition from LlamaIndex to LangChain/LangGraph for sophisticated food recommendation system

This implementation uses imports compatible with:
- langchain>=0.2.16,<0.3.0
- langgraph>=0.2.28,<0.3.0
- langchain-community>=0.2.16,<0.3.0
- langchain-core>=0.2.39,<0.3.0
"""

import os
import logging
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from EX1 import SemanticRecommendationTool
# Updated imports for new package structure
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Corrected import path
from langchain_community.vectorstores import Chroma     # Corrected import path
from langchain_google_genai import ChatGoogleGenerativeAI


# LangGraph imports for workflow orchestration - stable in 0.2.x series
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Pydantic for tool schemas - external dependency, stable
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Ashfaq8790"
GOOGLE_API_KEY = "AIzaSyBMk1QqgrS6ngivCNWeZf8TlARR58_BwQ4"

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
    name = "graph_recommendations"
    description = """
    Find novel food pairings using advanced graph co-occurrence analysis.
    This tool analyzes customer ordering patterns to find items that appear together 
    with 15-40% frequency, creating surprising but proven combinations.
    Includes cross-category and cross-cuisine bonuses for truly novel discoveries.
    """
    args_schema = GraphRecommendationInput

    # Fix to handle non-Pydantic types like Neo4j driver
    class Config:
        arbitrary_types_allowed = True  # This allows non-Pydantic types like Neo4j driver
    
    def __init__(self, neo4j_driver, **kwargs):
        super().__init__(**kwargs)
        # Use object.__setattr__ to bypass Pydantic validation for the driver assignment
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
                result = session.run(""" MATCH (item:Item {name: $item_name})
                    OPTIONAL MATCH (item)-[co:CO_OCCURS]-(other:Item)
                    WHERE co.rate >= 0.15 AND co.rate <= 0.60
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
                        END as cross_cuisine_bonus""", item_name=item_name, limit=top_k * 2)
                recommendations = []
                for record in result:
                    if record['recommended_item']:
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
                
                recommendations.sort(key=lambda x: x['novelty_score'], reverse=True)
                top_recommendations = recommendations[:top_k]
                
                if not top_recommendations:
                    return f"No novel graph-based recommendations found for {item_name}"
                
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


class LangChainGraphRAGProcessor:
    """
    Complete LangChain-based GraphRAG processor that rebuilds your system
    with explicit control and intelligent orchestration
    """
    
    def __init__(self):
        logger.info("Initializing LangChain GraphRAG Processor...")
        
        # Initialize Neo4j connection (unchanged from your current approach)
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
        # Initialize LangChain components with explicit configuration
        # Updated import path ensures compatibility with specified version ranges
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},  # Explicitly specify device
            encode_kwargs={'normalize_embeddings': True}  # Ensure normalized embeddings
        )
        
        # Text splitter for document processing - updated import ensures compatibility
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize LLM with your current Gemini setup
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,  # Slightly creative for food recommendations
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
        This preserves your rich item descriptions while making them available to LangChain
        """
        logger.info("Loading Neo4j data and converting to LangChain Documents...")
        
        with self.driver.session() as session:
            # Get all items with their rich attributes (same query as your current system)
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
                # Create the same rich text representation you had before
                page_content = f"""
                {record['name']} is a {record['category']} dish from {record['cuisine_type']} cuisine.
                Ingredients: {record['ingredients']}
                Flavor Profile: {record['flavor']}
                Texture: {record['texture']}
                Spice Level: {record['spice_level']}/10
                Best for: {record['occasion']}
                Goes well with: {', '.join(record['complementary'] or [])}
                """.strip()
                
                # LangChain metadata - more flexible than LlamaIndex
                metadata = {
                    "item_name": record['name'],
                    "category": record['category'],
                    "spice_level": record['spice_level'],
                    "cuisine_type": record['cuisine_type'],
                    "source": "neo4j_graph",
                    # Additional metadata for potential filtering
                    "has_ingredients": bool(record['ingredients']),
                    "has_flavor_profile": bool(record['flavor'])
                }
                
                # Create LangChain Document - using updated import
                doc = Document(page_content=page_content, metadata=metadata)
                documents.append(doc)
            
            logger.info(f"✓ Created {len(documents)} LangChain Documents from Neo4j data")
            return documents
    
    def create_vector_index(self, documents: List[Document]):
        """
        Create vector embeddings and storage using ChromaDB
        Updated to use compatible import paths
        """
        logger.info("Creating vector embeddings and ChromaDB index...")
        
        # Split documents if needed (your rich descriptions might benefit from this)
        split_docs = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # Create ChromaDB vector store with persistence - updated import ensures compatibility
        self.vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chromadb_storage",  # Persistent storage
            collection_name="menu_items"
        )
        
        # Create retriever with optimized settings for your use case
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 10,  # Get more candidates for better filtering
                #"score_threshold": 0.7  # Ensure minimum relevance
            }
        )
        
        logger.info("✓ Vector index and retriever created successfully")
        return self.vector_store
    
    def initialize_tools(self):
        """
        Initialize your sophisticated recommendation tools
        """
        logger.info("Initializing recommendation tools...")
        
        if not self.vector_store:
            raise ValueError("Vector store must be created before initializing tools")
        
        # Create tools that wrap your sophisticated algorithms
        self.graph_tool = GraphRecommendationTool(self.driver)
        self.semantic_tool = SemanticRecommendationTool(self.vector_store, self.retriever)
        
        logger.info("✓ Graph and semantic recommendation tools initialized")
    
    def extract_item_name(self, query: str) -> Optional[str]:
        """
        Enhanced item name extraction from user queries
        This is more sophisticated than your current simple pattern matching
        """
        query_lower = query.lower().strip()
        
        # Define comprehensive query patterns
        patterns = [
            # Direct "what goes with X" patterns
            r"what (?:goes|pairs) with (.+?)(?:\?|$)",
            r"what should i (?:eat|order|have) with (.+?)(?:\?|$)",
            
            # "Recommend X for Y" patterns - this catches "Recommend something surprising for Masala Dosa"
            r"recommend (?:something|items) (?:surprising|good|novel) for (.+?)(?:\?|$)",
            r"recommend (?:something|items) (?:for|to go with) (.+?)(?:\?|$)",
            
            # "Suggest X for Y" patterns - this catches "Suggest items that pair well with Dal Tadka"
            r"suggest (?:something|items) (?:that )?(?:pair well with|for) (.+?)(?:\?|$)",
            r"suggest (?:something|items) (?:for|with) (.+?)(?:\?|$)",
            
            # Novel/pairing patterns
            r"what are (?:some )?(?:good|novel|surprising) pairings? for (.+?)(?:\?|$)",
            r"find (?:recommendations|pairings) for (.+?)(?:\?|$)",
            
            # More direct patterns
            r"pairings? for (.+?)(?:\?|$)",
            r"(?:what|which) (?:foods?|dishes?|items?) go with (.+?)(?:\?|$)",
            
            # Catch-all pattern for any query ending with a food item
            r"(?:.*\s+)?(.+?)(?:\?|$)"  # This should be last as it's very broad
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                item_name = match.group(1).strip()
                # Clean up the extracted name
                item_name = item_name.replace("?", "").strip()
                # Convert to title case for consistency with your data
                return " ".join(word.capitalize() for word in item_name.split())
        
        return None
    
    def create_workflow(self):
        """
        Create LangGraph workflow for intelligent orchestration
        This is where LangGraph shines - intelligent decision making about which tools to use when
        """
        logger.info("Creating LangGraph workflow...")
        
        def extract_item(state: GraphRAGState) -> GraphRAGState:
            """Extract item name from user query"""
            item_name = self.extract_item_name(state["query"])
            if not item_name:
                state["final_response"] = "I couldn't identify a specific menu item in your query. Please try: 'What goes with Chicken Biryani?'"
                return state
            
            state["item_name"] = item_name
            state["messages"] = [f"Extracted item: {item_name}"]
            return state
        
        def decide_strategy(state: GraphRAGState) -> str:
            """
            Intelligent decision making about which recommendation approach to use
            This demonstrates LangGraph's power - adaptive behavior based on context
            """
            if not state.get("item_name"):
                return "end"
            
            # For now, we'll use both approaches and let the synthesis step decide
            # In a more advanced version, you could add logic like:
            # - Use only graph for popular items with rich co-occurrence data
            # - Use semantic for rare items with little co-occurrence data
            # - Use both for comprehensive recommendations
            
            return "get_recommendations"
        
        def get_recommendations(state: GraphRAGState) -> GraphRAGState:
            """Get recommendations from both graph and semantic tools"""
            item_name = state["item_name"]
            
            # Get graph-based recommendations (your sophisticated algorithm)
            try:
                graph_result = self.graph_tool._run(item_name, top_k=5)
                state["graph_recommendations"] = self._parse_graph_results(graph_result)
                state["messages"].append(f"Graph recommendations: {len(state['graph_recommendations'])} found")
            except Exception as e:
                logger.error(f"Graph tool error: {e}")
                state["graph_recommendations"] = []
                state["messages"].append(f"Graph recommendations failed: {e}")
            
            # Get semantic recommendations
            try:
                semantic_result = self.semantic_tool._run(item_name, top_k=5)
                state["semantic_recommendations"] = self._parse_semantic_results(semantic_result)
                state["messages"].append(f"Semantic recommendations: {len(state['semantic_recommendations'])} found")
            except Exception as e:
                logger.error(f"Semantic tool error: {e}")
                state["semantic_recommendations"] = []
                state["messages"].append(f"Semantic recommendations failed: {e}")
            
            return state
        
        def synthesize_response(state: GraphRAGState) -> GraphRAGState:
            """
            Use your sophisticated prompt engineering to generate final recommendations
            This preserves your current high-quality natural language generation
            """
            item_name = state["item_name"]
            graph_recs = state.get("graph_recommendations", [])
            semantic_recs = state.get("semantic_recommendations", [])
            
            # Create context strings (similar to your current approach)
            graph_context = self._format_graph_context(graph_recs)
            semantic_context = self._format_semantic_context(semantic_recs)
            
            # Use your proven prompt template approach
            prompt = PromptTemplate(
                input_variables=["item_name", "graph_context", "semantic_context"],
                template="""You are a culinary expert specializing in food recommendations based on DATA ANALYSIS.

Current item: {item_name}

ANALYSIS RESULTS FROM OUR SYSTEMS:
NOVEL PAIRINGS (from customer ordering patterns):
{graph_context}
SIMILAR ITEMS (based on flavor/texture analysis):
{semantic_context}

CRITICAL INSTRUCTION: You MUST base your recommendations on the items listed above. Do NOT invent new items.

Task: Choose 2-3 items from the data above and explain why they pair well with {item_name}.

Start each recommendation with: "Based on our analysis, [ITEM NAME] pairs surprisingly well with {item_name}..."
"""

            )
            
            # Generate response using LangChain chain - updated for newer version compatibility
            chain = prompt | self.llm | StrOutputParser()
            
            try:
                response = chain.invoke({
                    "item_name": item_name,
                    "graph_context": graph_context,
                    "semantic_context": semantic_context
                })
                state["final_response"] = response
                state["messages"].append("Successfully generated recommendation response")
            except Exception as e:
                logger.error(f"Response generation error: {e}")
                state["final_response"] = f"I found some interesting pairings for {item_name}, but had trouble generating the detailed explanation. Here's what I discovered: {graph_context}"
                state["messages"].append(f"Response generation failed: {e}")
            
            return state
        
        # Build the workflow graph
        workflow = StateGraph(GraphRAGState)
        
        # Add nodes (processing steps)
        workflow.add_node("extract_item", extract_item)
        workflow.add_node("get_recommendations", get_recommendations)
        workflow.add_node("synthesize_response", synthesize_response)
        
        # Add edges (flow control)
        workflow.set_entry_point("extract_item")
        workflow.add_conditional_edges(
            "extract_item",
            decide_strategy,
            {
                "get_recommendations": "get_recommendations",
                "end": END
            }
        )
        workflow.add_edge("get_recommendations", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        # Compile the workflow
        self.workflow = workflow.compile()
        logger.info("✓ LangGraph workflow created successfully")
        
        return self.workflow
    
    def _parse_graph_results(self, result_text: str) -> List[Dict]:
        """Parse graph tool results back into structured format"""
        # This is a simplified parser - you could make it more sophisticated
        recommendations = []
        if "Graph-based novel pairings" in result_text:
            lines = result_text.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip() and '. ' in line:
                    # Extract item name and details
                    parts = line.split('. ', 1)[1].split(' - ')
                    if len(parts) >= 2:
                        item_category = parts[0]
                        item_name = item_category.split(' (')[0]
                        recommendations.append({
                            'item': item_name,
                            'details': parts[1] if len(parts) > 1 else ''
                        })
        return recommendations
    
    def _parse_semantic_results(self, result_text: str) -> List[Dict]:
        """Parse semantic tool results back into structured format"""
        recommendations = []
        if "Semantically similar items" in result_text:
            lines = result_text.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip() and '. ' in line:
                    parts = line.split('. ', 1)[1].split(' - ')
                    if len(parts) >= 1:
                        item_category = parts[0]
                        item_name = item_category.split(' (')[0]
                        recommendations.append({
                            'item': item_name,
                            'reason': 'semantic_similarity'
                        })
        return recommendations
    
    def _format_graph_context(self, recommendations: List[Dict]) -> str:
        """Format graph recommendations for prompt context"""
        if not recommendations:
            return "No graph-based recommendations found."
        
        context = ""
        for i, rec in enumerate(recommendations, 1):
            context += f"{i}. {rec['item']} - {rec.get('details', 'Novel pairing based on customer data')}\n"
        return context
    
    def _format_semantic_context(self, recommendations: List[Dict]) -> str:
        """Format semantic recommendations for prompt context"""
        if not recommendations:
            return "No semantic recommendations available."
        
        context = ""
        for i, rec in enumerate(recommendations, 1):
            context += f"{i}. {rec['item']} - Similar flavor/texture profile\n"
        return context
    
    def query(self, user_query: str) -> str:
        """
        Main query interface - now powered by LangGraph workflow
        """
        if not self.workflow:
            return "System not properly initialized. Please run setup_system() first."
        
        # Execute the intelligent workflow
        try:
            initial_state = GraphRAGState(
                query=user_query,
                item_name="",
                graph_recommendations=[],
                semantic_recommendations=[],
                recommendation_strategy="",
                final_response="",
                messages=[]
            )
            
            # Run the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Return the final response
            return final_state["final_response"]
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            return f"I encountered an error processing your query: {str(e)}. Please try rephrasing your question."
    
    def setup_system(self, data_path: str = None):
        """
        Complete system setup - call this once to initialize everything
        """
        logger.info("Setting up complete LangChain GraphRAG system...")
        
        # Load documents from Neo4j
        documents = self.load_neo4j_data_as_documents()
        
        # Create vector index
        self.create_vector_index(documents)
        
        # Initialize tools
        self.initialize_tools()
        
        # Create workflow
        self.create_workflow()
        
        logger.info("✓ LangChain GraphRAG system setup complete!")
        return self
    
    def close(self):
        """Clean up resources"""
        if self.driver:
            self.driver.close()
        logger.info("✓ Resources cleaned up")

# Usage example and testing
def main():
    """
    Main function demonstrating how to use the new LangChain GraphRAG system
    """
    # Initialize the system
    processor = LangChainGraphRAGProcessor()
    
    # Set up the complete system (this replaces your current setup_graphrag_system function)
    processor.setup_system()
    
    # Test queries (same as your current test cases)
    test_queries = [
        "What goes with Chicken Biryani?",
        "Recommend something surprising for Masala Dosa",
        "What are some novel pairings for Phulka?",
        "Suggest items that pair well with Dal Tadka"
    ]
    
    print("\n" + "="*80)
    print("LANGCHAIN GRAPHRAG SYSTEM - RECOMMENDATION RESULTS")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        response = processor.query(query)
        print(response)
        print()
    
    # Clean up
    processor.close()

if __name__ == "__main__":
    main()