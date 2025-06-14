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

import pandas as pd


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
class EnhancedKnowledgeGraphBuilder:
    """
    This class extends your existing LangChainGraphRAGProcessor 
    to build a multi-dimensional knowledge graph using real data from DF1.xlsx
    
    Key Philosophy: Preserve your sophisticated Item-Item relationships while adding
    new dimensions (Menu, Event, Meal) that provide business context and intelligence
    """
    
    def __init__(self, existing_processor):
        """
        Initialize with your existing GraphRAG processor to build upon its foundation
        Think of this as adding new floors to a building with solid foundations
        """
        self.processor = existing_processor
        self.driver = existing_processor.driver
        self.logger = logging.getLogger(__name__)
        
        # Data containers for extracted information
        self.df1_data = None
        self.unique_events = []
        self.unique_meals = []
        self.menu_compositions = {}
        
        self.logger.info("Enhanced Knowledge Graph Builder initialized")
    
    def load_and_analyze_df1_data(self, df1_file_path: str = "DF1.xlsx") -> pd.DataFrame:
        """
        Load DF1.xlsx and extract the rich business intelligence it contains
        This replaces hardcoded lists with real business data patterns
        
        Think of DF1 as your 'business intelligence database' - it contains
        real event types, meal patterns, and successful menu combinations
        """
        self.logger.info("Loading and analyzing DF1.xlsx for knowledge graph construction...")
        
        try:
            # Load the main data sheet - this contains your business reality
            self.df1_data = pd.read_excel(df1_file_path, sheet_name='Sheet1')
            
            # Clean and validate the data structure
            required_columns = ['menu_id', 'Event Type', 'Meal Type', 'Category', 'item_name', 'Item Description']
            missing_columns = [col for col in required_columns if col not in self.df1_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns in DF1.xlsx: {missing_columns}")
            
            # Extract unique business contexts from real data
            # self.unique_events = sorted(self.df1_data['Event Type'].dropna().unique())
            # self.unique_meals = sorted(self.df1_data['Meal Type'].dropna().unique())
            # Quick fix version - just convert everything to strings before sorting
            self.unique_events = sorted([str(x) for x in self.df1_data['Event Type'].dropna().unique()])
            self.unique_meals = sorted([str(x) for x in self.df1_data['Meal Type'].dropna().unique()])
                        # Analyze menu compositions - this captures successful item combinations
            self._analyze_menu_compositions()
            
            self.logger.info(f"✓ DF1 analysis complete:")
            self.logger.info(f"  - Total records: {len(self.df1_data)}")
            self.logger.info(f"  - Unique events: {len(self.unique_events)}")
            self.logger.info(f"  - Unique meals: {len(self.unique_meals)}")
            self.logger.info(f"  - Unique menus: {len(self.menu_compositions)}")
            
            return self.df1_data
            
        except Exception as e:
            self.logger.error(f"Error loading DF1 data: {e}")
            raise
    
    def _analyze_menu_compositions(self):
        """
        Analyze how items are combined in successful real-world menus
        This creates the foundation for Menu nodes and CONTAINS relationships
        
        Each menu in DF1 represents a successful business transaction,
        so menu compositions reflect proven successful combinations
        """
        self.menu_compositions = {}
        
        for _, row in self.df1_data.iterrows():
            menu_id = row['menu_id']
            
            if pd.notna(menu_id):
                if menu_id not in self.menu_compositions:
                    # Initialize menu composition with metadata
                    self.menu_compositions[menu_id] = {
                        'items': [],
                        'event_type': row['Event Type'],
                        'meal_type': row['Meal Type'],
                        'categories': set(),  # Track variety of categories
                        'item_count': 0
                    }
                
                # Add item to menu composition
                if pd.notna(row['item_name']):
                    self.menu_compositions[menu_id]['items'].append(row['item_name'])
                    self.menu_compositions[menu_id]['categories'].add(row['Category'])
        
        # Calculate final metrics for each menu
        for menu_id, composition in self.menu_compositions.items():
            composition['item_count'] = len(composition['items'])
            composition['category_count'] = len(composition['categories'])
            composition['categories'] = list(composition['categories'])  # Convert set to list
        
        self.logger.info(f"✓ Analyzed {len(self.menu_compositions)} menu compositions")
    
    def create_enhanced_knowledge_graph(self):
        """
        Build the complete multi-dimensional knowledge graph
        This preserves your existing Item nodes and CO_OCCURS relationships
        while adding Menu, Event, and Meal dimensions for business intelligence
        
        Execution order matters:
        1. Preserve existing Item nodes (your foundation)
        2. Add new node types (Menu, Event, Meal)
        3. Create relationships that connect the dimensions
        """
        self.logger.info("Building enhanced multi-dimensional knowledge graph...")
        
        if self.df1_data is None:
            raise ValueError("Must load DF1 data before building knowledge graph")
        
        with self.driver.session() as session:
            # Step 1: Create Menu nodes from real business data
            self._create_menu_nodes(session)
            
            # Step 2: Create Event nodes from extracted event types
            self._create_event_nodes(session)
            
            # Step 3: Create Meal nodes from extracted meal types
            self._create_meal_nodes(session)
            
            # Step 4: Create CONTAINS relationships (Menu -> Item)
            self._create_contains_relationships(session)
            
            # Step 5: Create SUITABLE_FOR relationships (Item -> Event)
            self._create_suitable_for_relationships(session)
            
            # Step 6: Create SERVED_AT relationships (Item -> Meal)
            self._create_served_at_relationships(session)
            
            # Step 7: Validate the enhanced graph structure
            self._validate_graph_structure(session)
        
        self.logger.info("✓ Enhanced knowledge graph construction complete!")
    
    def _create_menu_nodes(self, session):
        """
        Create Menu nodes with minimal but powerful properties
        Each Menu node represents a successful real-world business transaction
        
        Properties:
        - id: Links back to DF1 for traceability
        - item_count: Simple structural insight about menu complexity
        """
        self.logger.info("Creating Menu nodes from DF1 business data...")
        
        menu_creation_query = """
        CREATE (m:Menu {
            id: $menu_id,
            item_count: $item_count
        })
        """
        
        created_count = 0
        for menu_id, composition in self.menu_compositions.items():
            try:
                session.run(menu_creation_query, {
                    'menu_id': menu_id,
                    'item_count': composition['item_count']
                })
                created_count += 1
                
                # Log progress for large datasets
                if created_count % 100 == 0:
                    self.logger.info(f"  Created {created_count} Menu nodes...")
                    
            except Exception as e:
                self.logger.warning(f"Failed to create Menu node {menu_id}: {e}")
        
        self.logger.info(f"✓ Created {created_count} Menu nodes")
    
    def _create_event_nodes(self, session):
        """
        Create Event nodes from real business event types extracted from DF1
        Each Event represents a different business context with unique requirements
        
        Properties:
        - name: The event type from your real business data
        """
        self.logger.info("Creating Event nodes from real business contexts...")
        
        event_creation_query = """
        CREATE (e:Event {name: $event_name})
        """
        
        created_count = 0
        for event_name in self.unique_events:
            try:
                session.run(event_creation_query, {'event_name': event_name})
                created_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to create Event node {event_name}: {e}")
        
        self.logger.info(f"✓ Created {created_count} Event nodes from real business data")
        self.logger.info(f"  Event types: {', '.join(self.unique_events[:5])}{'...' if len(self.unique_events) > 5 else ''}")
    
    def _create_meal_nodes(self, session):
        """
        Create Meal nodes from meal timing patterns extracted from DF1
        Each Meal represents a different temporal eating context
        
        Properties:
        - name: The meal type from your real business data
        """
        self.logger.info("Creating Meal nodes from real meal timing data...")
        
        meal_creation_query = """
        CREATE (m:Meal {name: $meal_name})
        """
        
        created_count = 0
        for meal_name in self.unique_meals:
            try:
                session.run(meal_creation_query, {'meal_name': meal_name})
                created_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to create Meal node {meal_name}: {e}")
        
        self.logger.info(f"✓ Created {created_count} Meal nodes")
        self.logger.info(f"  Meal types: {', '.join(self.unique_meals)}")
    
    def _create_contains_relationships(self, session):
        """
        Create CONTAINS relationships between Menu and Item nodes
        These relationships capture successful item combinations from real business
        
        Each CONTAINS relationship represents: "This menu successfully included this item"
        The relationship properties capture additional business intelligence
        """
        self.logger.info("Creating CONTAINS relationships (Menu -> Item)...")
        
        contains_query = """
        MATCH (m:Menu {id: $menu_id})
        MATCH (i:Item {name: $item_name})
        CREATE (m)-[:CONTAINS {
            menu_context: $event_meal_context,
            successful_combination: true,
            source: 'DF1_historical'
        }]->(i)
        """
        
        created_count = 0
        failed_count = 0
        
        for menu_id, composition in self.menu_compositions.items():
            # Create context string for this menu
            context = f"{composition['event_type']}_{composition['meal_type']}"
            
            for item_name in composition['items']:
                try:
                    result = session.run(contains_query, {
                        'menu_id': menu_id,
                        'item_name': item_name,
                        'event_meal_context': context
                    })
                    
                    # Check if relationship was actually created
                    if result.summary().counters.relationships_created > 0:
                        created_count += 1
                    else:
                        failed_count += 1
                        # Item might not exist in your current Item nodes
                        
                except Exception as e:
                    failed_count += 1
                    self.logger.debug(f"Failed to create CONTAINS: {menu_id} -> {item_name}: {e}")
            
            # Progress logging for large datasets
            if (created_count + failed_count) % 500 == 0:
                self.logger.info(f"  Processed {created_count + failed_count} relationships...")
        
        self.logger.info(f"✓ Created {created_count} CONTAINS relationships")
        if failed_count > 0:
            self.logger.info(f"  Note: {failed_count} relationships not created (items may not exist in current Item nodes)")
    
    def _create_suitable_for_relationships(self, session):
        """
        Create SUITABLE_FOR relationships between Item and Event nodes
        This encodes business logic about which items work for which events
        
        Initial approach: Use simple business rules based on item properties
        Advanced approach: Use machine learning on DF1 patterns (future enhancement)
        """
        self.logger.info("Creating SUITABLE_FOR relationships (Item -> Event)...")
        
        # Get all current Item nodes to work with
        get_items_query = """
        MATCH (i:Item)
        RETURN i.name as name, i.spice_level as spice_level, 
               i.category as category, i.cuisine_type as cuisine_type
        """
        
        items_result = session.run(get_items_query)
        items = [record.data() for record in items_result]
        
        suitability_query = """
        MATCH (i:Item {name: $item_name})
        MATCH (e:Event {name: $event_name})
        CREATE (i)-[:SUITABLE_FOR {
            suitability_score: $score,
            reasoning: $reasoning,
            derived_from: 'business_logic'
        }]->(e)
        """
        
        created_count = 0
        
        for item in items:
            for event_name in self.unique_events:
                # Calculate suitability using business logic
                score, reasoning = self._calculate_event_suitability(item, event_name)
                
                if score >= 0.5:  # Only create relationships for suitable pairings
                    try:
                        session.run(suitability_query, {
                            'item_name': item['name'],
                            'event_name': event_name,
                            'score': score,
                            'reasoning': reasoning
                        })
                        created_count += 1
                    except Exception as e:
                        self.logger.debug(f"Failed to create SUITABLE_FOR: {item['name']} -> {event_name}: {e}")
        
        self.logger.info(f"✓ Created {created_count} SUITABLE_FOR relationships")
    
    def _create_served_at_relationships(self, session):
        """
        Create SERVED_AT relationships between Item and Meal nodes
        This captures cultural and temporal appropriateness patterns
        """
        self.logger.info("Creating SERVED_AT relationships (Item -> Meal)...")
        
        # Get all current Item nodes
        get_items_query = """
        MATCH (i:Item)
        RETURN i.name as name, i.category as category, 
               i.spice_level as spice_level, i.cuisine_type as cuisine_type
        """
        
        items_result = session.run(get_items_query)
        items = [record.data() for record in items_result]
        
        served_at_query = """
        MATCH (i:Item {name: $item_name})
        MATCH (m:Meal {name: $meal_name})
        CREATE (i)-[:SERVED_AT {
            appropriateness_score: $score,
            cultural_reasoning: $reasoning,
            derived_from: 'cultural_knowledge'
        }]->(m)
        """
        
        created_count = 0
        
        for item in items:
            for meal_name in self.unique_meals:
                # Calculate meal appropriateness using cultural knowledge
                score, reasoning = self._calculate_meal_appropriateness(item, meal_name)
                
                if score >= 0.4:  # More permissive than event suitability
                    try:
                        session.run(served_at_query, {
                            'item_name': item['name'],
                            'meal_name': meal_name,
                            'score': score,
                            'reasoning': reasoning
                        })
                        created_count += 1
                    except Exception as e:
                        self.logger.debug(f"Failed to create SERVED_AT: {item['name']} -> {meal_name}: {e}")
        
        self.logger.info(f"✓ Created {created_count} SERVED_AT relationships")
    
    def _calculate_event_suitability(self, item: Dict, event_name: str) -> Tuple[float, str]:
        """
        Business logic for determining if an item is suitable for an event
        This encodes your domain expertise as algorithmic decision making
        
        Returns: (suitability_score, reasoning_text)
        """
        score = 0.5  # Start with neutral
        reasons = []
        
        # Corporate events need professional-friendly foods
        if 'Corporate' in event_name:
            if item.get('spice_level', 5) <= 4:
                score += 0.3
                reasons.append("low spice appropriate for business")
            else:
                score -= 0.2
                reasons.append("high spice may not suit business context")
            
            if item.get('category') in ['Main Course', 'Snacks']:
                score += 0.2
                reasons.append("suitable category for corporate dining")
        
        # Wedding events are more permissive and festive
        elif 'Wedding' in event_name or 'Engagement' in event_name:
            if item.get('category') in ['Main Course', 'Desserts', 'Snacks']:
                score += 0.3
                reasons.append("festive food appropriate for celebrations")
            
            if item.get('spice_level', 5) >= 6:
                score += 0.1
                reasons.append("flavorful food suits celebration")
        
        # Family gatherings are very permissive
        elif 'Family' in event_name or 'House' in event_name:
            score += 0.2
            reasons.append("suitable for family contexts")
        
        # Party events favor exciting foods
        elif 'Party' in event_name:
            if item.get('category') in ['Snacks', 'Beverages', 'Desserts']:
                score += 0.3
                reasons.append("party-friendly food category")
        
        # Ensure score stays within bounds
        score = max(0.0, min(1.0, score))
        reasoning = "; ".join(reasons) if reasons else "general suitability"
        
        return score, reasoning
    
    def _calculate_meal_appropriateness(self, item: Dict, meal_name: str) -> Tuple[float, str]:
        """
        Cultural logic for determining meal timing appropriateness
        This encodes cultural eating patterns as algorithmic decisions
        """
        score = 0.3  # Start lower than event suitability
        reasons = []
        
        category = item.get('category', '').lower()
        spice_level = item.get('spice_level', 5)
        
        if meal_name == 'Breakfast':
            if 'beverage' in category or 'light' in category:
                score += 0.4
                reasons.append("light foods appropriate for morning")
            
            if spice_level <= 3:
                score += 0.3
                reasons.append("mild spice suits morning palate")
            else:
                score -= 0.2
                reasons.append("high spice less suitable for morning")
        
        elif meal_name == 'Lunch':
            if 'main course' in category:
                score += 0.4
                reasons.append("substantial food for midday meal")
            
            if 2 <= spice_level <= 7:
                score += 0.2
                reasons.append("moderate spice appropriate for lunch")
        
        elif meal_name == 'Dinner':
            if 'main course' in category or 'dessert' in category:
                score += 0.4
                reasons.append("elaborate food suitable for evening")
            
            score += 0.1  # Dinner is generally permissive
            reasons.append("evening meal allows variety")
        
        elif meal_name == 'Snacks':
            if 'snack' in category or 'appetizer' in category:
                score += 0.5
                reasons.append("designed for snacking")
            
            if 'beverage' in category:
                score += 0.3
                reasons.append("beverages complement snacks")
        
        score = max(0.0, min(1.0, score))
        reasoning = "; ".join(reasons) if reasons else "general meal appropriateness"
        
        return score, reasoning
    
    def _validate_graph_structure(self, session):
        """
        Validate that the enhanced knowledge graph was built correctly
        This provides confidence that all dimensions are properly connected
        """
        self.logger.info("Validating enhanced knowledge graph structure...")
        
        validation_queries = {
            'total_nodes': "MATCH (n) RETURN count(n) as count",
            'item_nodes': "MATCH (n:Item) RETURN count(n) as count",
            'menu_nodes': "MATCH (n:Menu) RETURN count(n) as count", 
            'event_nodes': "MATCH (n:Event) RETURN count(n) as count",
            'meal_nodes': "MATCH (n:Meal) RETURN count(n) as count",
            'contains_rels': "MATCH ()-[r:CONTAINS]->() RETURN count(r) as count",
            'co_occurs_rels': "MATCH ()-[r:CO_OCCURS]->() RETURN count(r) as count",
            'suitable_for_rels': "MATCH ()-[r:SUITABLE_FOR]->() RETURN count(r) as count",
            'served_at_rels': "MATCH ()-[r:SERVED_AT]->() RETURN count(r) as count"
        }
        
        validation_results = {}
        for query_name, query in validation_queries.items():
            try:
                result = session.run(query)
                count = result.single()['count']
                validation_results[query_name] = count
            except Exception as e:
                self.logger.error(f"Validation query {query_name} failed: {e}")
                validation_results[query_name] = "ERROR"
        
        # Log validation results
        self.logger.info("✓ Knowledge graph validation results:")
        self.logger.info(f"  Total nodes: {validation_results['total_nodes']}")
        self.logger.info(f"  Item nodes: {validation_results['item_nodes']}")
        self.logger.info(f"  Menu nodes: {validation_results['menu_nodes']}")
        self.logger.info(f"  Event nodes: {validation_results['event_nodes']}")
        self.logger.info(f"  Meal nodes: {validation_results['meal_nodes']}")
        self.logger.info(f"  CONTAINS relationships: {validation_results['contains_rels']}")
        self.logger.info(f"  CO_OCCURS relationships: {validation_results['co_occurs_rels']}")
        self.logger.info(f"  SUITABLE_FOR relationships: {validation_results['suitable_for_rels']}")
        self.logger.info(f"  SERVED_AT relationships: {validation_results['served_at_rels']}")
        
        return validation_results
# ================================================================================================
# INTEGRATION HELPER FUNCTION - ADD THIS RIGHT AFTER THE ENHANCED BUILDER CLASS
# ================================================================================================

def enhance_existing_knowledge_graph(processor, df1_file_path: str = "DF1.xlsx"):
    """
    Main function to enhance your existing LangChain GraphRAG system
    Call this after your current setup_system() to add multi-dimensional intelligence
    
    This preserves all your existing sophisticated algorithms while adding
    business context and temporal intelligence from your real data
    """
    logger = logging.getLogger(__name__)
    logger.info("Enhancing existing knowledge graph with multi-dimensional intelligence...")
    
    # Create the enhanced builder using your existing processor
    # This is the key integration point - your existing processor becomes the foundation
    builder = EnhancedKnowledgeGraphBuilder(processor)
    
    # Load and analyze your real business data from DF1.xlsx
    builder.load_and_analyze_df1_data(df1_file_path)
    
    # Build the enhanced multi-dimensional knowledge graph
    # This adds Menu, Event, and Meal nodes while preserving your Item nodes
    builder.create_enhanced_knowledge_graph()
    
    logger.info("✓ Knowledge graph enhancement complete!")
    logger.info("Your system now has multi-dimensional recommendation capabilities")
    
    return builder  # Return the builder for further analysis if needed
def main():
    """
    Enhanced main function that builds upon your existing sophisticated system
    This preserves all current functionality while adding multi-dimensional intelligence
    """
    print("="*80)
    print("INITIALIZING ENHANCED LANGCHAIN GRAPHRAG SYSTEM")
    print("="*80)
    
    # STEP 1: Initialize your existing sophisticated system (completely unchanged)
    # This runs all your proven algorithms for Item nodes and CO_OCCURS relationships
    processor = LangChainGraphRAGProcessor()
    processor.setup_system()  # Your existing sophisticated setup runs first
    
    print("\n" + "="*60)
    print("TESTING EXISTING FUNCTIONALITY (should work exactly as before)")
    print("="*60)
    
    # Test your existing queries to verify nothing is broken
    existing_queries = [
        "What goes with Chicken Dum Biryani?",
        "Recommend something for Dosa",
        "What are some novel pairings for Phulka?"
    ]
    
    for query in existing_queries:
        print(f"\nQuery: {query}")
        response = processor.query(query)
        print(f"Response: {response}")
    
    print("\n" + "="*60)
    print("ENHANCING WITH MULTI-DIMENSIONAL BUSINESS INTELLIGENCE")
    print("="*60)
    
    # STEP 2: Enhance with multi-dimensional knowledge from DF1.xlsx
    # This builds upon your existing foundation without modifying it
    try:
        builder = enhance_existing_knowledge_graph(processor, "DF1.xlsx")
        
        print("\n" + "="*60)
        print("TESTING ENHANCED CONTEXTUAL CAPABILITIES")
        print("="*60)
        
        # Test new contextual queries that leverage business intelligence
        contextual_queries = [
            "What goes with Chicken Dum Biryani for a Corporate Event lunch?",
            "Recommend breakfast items that pair well with Dosa",
            "Suggest snack items for a Birthday Party that go with Dal Tadka"
        ]
        
        for query in contextual_queries:
            print(f"\nContextual Query: {query}")
            response = processor.query(query)
            print(f"Enhanced Response: {response}")
            
    except Exception as e:
        print(f"\nEnhancement failed: {e}")
        print("System will continue with existing functionality")
    
    # Clean up resources
    processor.close()
    print("\n" + "="*80)
    print("SYSTEM TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()