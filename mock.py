"""
Mock LangChain GraphRAG System - Works Without Neo4j
This allows you to test your LangChain implementation while setting up Neo4j
"""

import logging
from typing import List, Dict, Optional, Any
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY = "AIzaSyBMk1QqgrS6ngivCNWeZf8TlARR58_BwQ4"

class GraphRAGState(TypedDict):
    query: str
    item_name: str
    graph_recommendations: List[Dict]
    semantic_recommendations: List[Dict]
    recommendation_strategy: str
    final_response: str
    messages: List[str]

class MockGraphRecommendationTool(BaseTool):
    name = "graph_recommendations"
    description = "Find novel food pairings using mock graph data"
    
    def _run(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        """Mock sophisticated graph recommendations"""
        
        # Rich mock data that simulates your sophisticated algorithm
        mock_recommendations = {
            "Chicken Dum Biryani": [
                ("Raita", "Accompaniment", 0.85, 3.5, 127),
                ("Shorba", "Soup", 0.35, 2.8, 45),
                ("Pickle", "Condiment", 0.25, 2.1, 32),
                ("Papad", "Snacks", 0.30, 2.0, 38),
                ("Lassi", "Beverages", 0.20, 1.8, 25)
            ],
            "Dosa": [
                ("Sambar", "Curry", 0.90, 4.2, 156),
                ("Coconut Chutney", "Condiment", 0.85, 4.0, 143),
                ("Potato Curry", "Curry", 0.40, 2.5, 67),
                ("Filter Coffee", "Beverages", 0.35, 2.2, 48),
                ("Tomato Chutney", "Condiment", 0.30, 1.9, 41)
            ],
            "Dal Tadka": [
                ("Rice", "Main Course", 0.95, 4.8, 203),
                ("Roti", "Bread", 0.85, 4.1, 178),
                ("Pickle", "Condiment", 0.30, 2.3, 52),
                ("Onion Salad", "Vegetables", 0.25, 1.8, 34),
                ("Ghee", "Condiment", 0.40, 2.6, 89)
            ],
            "Phulka": [
                ("Dal", "Curry", 0.88, 4.0, 145),
                ("Sabzi", "Vegetables", 0.75, 3.2, 98),
                ("Pickle", "Condiment", 0.35, 2.4, 56),
                ("Yogurt", "Dairy", 0.45, 2.8, 73),
                ("Ghee", "Condiment", 0.50, 3.0, 82)
            ]
        }
        
        # Get recommendations for the item
        recs = mock_recommendations.get(item_name, [
            ("Suggested Item 1", "Main Course", 0.30, 2.0, 25),
            ("Suggested Item 2", "Snacks", 0.25, 1.8, 20),
            ("Suggested Item 3", "Beverages", 0.20, 1.5, 15)
        ])
        
        result_text = f"Graph-based novel pairings for {item_name}:\n"
        for i, (item, category, co_rate, novelty, times) in enumerate(recs[:top_k], 1):
            result_text += f"{i}. {item} ({category}) - "
            result_text += f"Novelty: {novelty:.2f}, "
            result_text += f"Co-occurrence: {int(co_rate*100)}%, "
            result_text += f"Ordered together: {times} times\n"
        
        return result_text
    
    async def _arun(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        return self._run(item_name, top_k, run_manager)

class MockSemanticRecommendationTool(BaseTool):
    name = "semantic_recommendations"
    description = "Find semantically similar items using mock embeddings"
    
    def _run(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        """Mock semantic similarity recommendations"""
        
        mock_semantic = {
            "Chicken Dum Biryani": [
                ("Mutton Biryani", "Main Course", 0.92),
                ("Chicken Pulao", "Main Course", 0.85),
                ("Veg Biryani", "Main Course", 0.78),
                ("Hyderabadi Biryani", "Main Course", 0.88),
                ("Fried Rice", "Main Course", 0.65)
            ],
            "Dosa": [
                ("Uttapam", "Main Course", 0.89),
                ("Idli", "Main Course", 0.85),
                ("Vada", "Snacks", 0.75),
                ("Paniyaram", "Snacks", 0.72),
                ("Appam", "Main Course", 0.68)
            ],
            "Dal Tadka": [
                ("Dal Fry", "Curry", 0.95),
                ("Moong Dal", "Curry", 0.88),
                ("Chana Dal", "Curry", 0.82),
                ("Sambar", "Curry", 0.75),
                ("Rajma", "Curry", 0.70)
            ]
        }
        
        recs = mock_semantic.get(item_name, [
            ("Similar Item 1", "Main Course", 0.70),
            ("Similar Item 2", "Curry", 0.65),
            ("Similar Item 3", "Snacks", 0.60)
        ])
        
        result_text = f"Semantically similar items to {item_name}:\n"
        for i, (item, category, similarity) in enumerate(recs[:top_k], 1):
            result_text += f"{i}. {item} ({category}) - Similarity: {similarity:.2f}\n"
        
        return result_text
    
    async def _arun(self, item_name: str, top_k: int = 5, run_manager: Optional[Any] = None) -> str:
        return self._run(item_name, top_k, run_manager)

class MockLangChainGraphRAGProcessor:
    """Mock version of your sophisticated system"""
    
    def __init__(self):
        logger.info("Initializing Mock LangChain GraphRAG Processor...")
        
        # Initialize LangChain components
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )
        
        self.vector_store = None
        self.graph_tool = None
        self.semantic_tool = None
        self.workflow = None
        
        logger.info("✓ Mock components initialized")
    
    def create_mock_documents(self) -> List[Document]:
        """Create mock documents that simulate your Neo4j data"""
        
        mock_items = [
            {
                "name": "Chicken Dum Biryani",
                "category": "Main Course",
                "ingredients": "Basmati rice, chicken, yogurt, spices",
                "flavor": "Rich, aromatic, spicy",
                "texture": "Fluffy rice, tender chicken",
                "spice_level": 7,
                "occasion": "Lunch, dinner, celebrations",
                "cuisine_type": "Hyderabadi"
            },
            {
                "name": "Dosa",
                "category": "Main Course",
                "ingredients": "Rice, lentils, fenugreek",
                "flavor": "Mild, tangy, crispy",
                "texture": "Crispy outside, soft inside",
                "spice_level": 2,
                "occasion": "Breakfast, lunch",
                "cuisine_type": "South Indian"
            },
            {
                "name": "Dal Tadka",
                "category": "Curry",
                "ingredients": "Lentils, onions, tomatoes, spices",
                "flavor": "Savory, mildly spiced",
                "texture": "Smooth, thick",
                "spice_level": 4,
                "occasion": "Lunch, dinner",
                "cuisine_type": "North Indian"
            },
            {
                "name": "Sambar",
                "category": "Curry",
                "ingredients": "Lentils, vegetables, tamarind, spices",
                "flavor": "Tangy, spicy",
                "texture": "Thick, chunky",
                "spice_level": 5,
                "occasion": "Lunch, dinner",
                "cuisine_type": "South Indian"
            },
            {
                "name": "Raita",
                "category": "Accompaniment",
                "ingredients": "Yogurt, cucumber, spices",
                "flavor": "Cool, refreshing",
                "texture": "Creamy, chunky",
                "spice_level": 1,
                "occasion": "With meals",
                "cuisine_type": "Indian"
            }
        ]
        
        documents = []
        for item in mock_items:
            page_content = f"""
            {item['name']} is a {item['category']} dish from {item['cuisine_type']} cuisine.
            Ingredients: {item['ingredients']}
            Flavor Profile: {item['flavor']}
            Texture: {item['texture']}
            Spice Level: {item['spice_level']}/10
            Best for: {item['occasion']}
            """.strip()
            
            metadata = {
                "item_name": item['name'],
                "category": item['category'],
                "spice_level": item['spice_level'],
                "cuisine_type": item['cuisine_type'],
                "source": "mock_data"
            }
            
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
        
        logger.info(f"✓ Created {len(documents)} mock documents")
        return documents
    
    def create_vector_index(self, documents: List[Document]):
        """Create vector index using mock documents"""
        logger.info("Creating vector embeddings with mock data...")
        
        split_docs = self.text_splitter.split_documents(documents)
        
        try:
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory="./mock_chromadb_storage",
                collection_name="mock_menu_items"
            )
            
            logger.info("✓ Mock vector store created")
            
        except Exception as e:
            logger.error(f"Vector store creation failed: {e}")
            raise
        
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        return self.vector_store
    
    def initialize_tools(self):
        """Initialize mock tools"""
        logger.info("Initializing mock recommendation tools...")
        
        self.graph_tool = MockGraphRecommendationTool()
        self.semantic_tool = MockSemanticRecommendationTool()
        
        logger.info("✓ Mock tools initialized")
    
    def extract_item_name(self, query: str) -> Optional[str]:
        """Extract item name from query"""
        query_lower = query.lower().strip()
        
        patterns = [
            r"what (?:goes|pairs) with (.+?)(?:\?|$)",
            r"recommend (?:something|items) (?:for|to go with) (.+?)(?:\?|$)",
            r"suggest (?:something|items) (?:for|with) (.+?)(?:\?|$)",
            r"pairings? for (.+?)(?:\?|$)",
            r"(?:.*\s+)?(.+?)(?:\?|$)"
        ]
        
        import re
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                item_name = match.group(1).strip().replace("?", "").strip()
                return " ".join(word.capitalize() for word in item_name.split())
        
        return None
    
    def create_workflow(self):
        """Create LangGraph workflow"""
        logger.info("Creating mock workflow...")
        
        def extract_item(state: GraphRAGState) -> GraphRAGState:
            item_name = self.extract_item_name(state["query"])
            if not item_name:
                state["final_response"] = "I couldn't identify a menu item. Try: 'What goes with Chicken Biryani?'"
                return state
            
            state["item_name"] = item_name
            state["messages"] = [f"Extracted item: {item_name}"]
            return state
        
        def decide_strategy(state: GraphRAGState) -> str:
            return "get_recommendations" if state.get("item_name") else "end"
        
        def get_recommendations(state: GraphRAGState) -> GraphRAGState:
            item_name = state["item_name"]
            
            # Get mock recommendations
            try:
                graph_result = self.graph_tool._run(item_name, top_k=5)
                state["graph_recommendations"] = self._parse_graph_results(graph_result)
                state["messages"].append(f"Graph: {len(state['graph_recommendations'])} found")
            except Exception as e:
                state["graph_recommendations"] = []
                state["messages"].append(f"Graph failed: {e}")
            
            try:
                semantic_result = self.semantic_tool._run(item_name, top_k=5)
                state["semantic_recommendations"] = self._parse_semantic_results(semantic_result)
                state["messages"].append(f"Semantic: {len(state['semantic_recommendations'])} found")
            except Exception as e:
                state["semantic_recommendations"] = []
                state["messages"].append(f"Semantic failed: {e}")
            
            return state
        
        def synthesize_response(state: GraphRAGState) -> GraphRAGState:
            item_name = state["item_name"]
            graph_recs = state.get("graph_recommendations", [])
            semantic_recs = state.get("semantic_recommendations", [])
            
            graph_context = self._format_graph_context(graph_recs)
            semantic_context = self._format_semantic_context(semantic_recs)
            
            prompt = PromptTemplate(
                input_variables=["item_name", "graph_context", "semantic_context"],
                template="""You are a culinary expert providing food recommendations.

Current item: {item_name}

NOVEL PAIRINGS (from ordering patterns):
{graph_context}

SIMILAR ITEMS (flavor/texture analysis):
{semantic_context}

Task: Choose 2-3 items from above and explain why they pair well with {item_name}.

Start each with: "Based on our analysis, [ITEM] pairs surprisingly well with {item_name}..."
"""
            )
            
            chain = prompt | self.llm | StrOutputParser()
            
            try:
                response = chain.invoke({
                    "item_name": item_name,
                    "graph_context": graph_context,
                    "semantic_context": semantic_context
                })
                state["final_response"] = response
            except Exception as e:
                state["final_response"] = f"Found interesting pairings for {item_name}: {graph_context}"
                logger.error(f"Response generation error: {e}")
            
            return state
        
        # Build workflow
        workflow = StateGraph(GraphRAGState)
        workflow.add_node("extract_item", extract_item)
        workflow.add_node("get_recommendations", get_recommendations)
        workflow.add_node("synthesize_response", synthesize_response)
        
        workflow.set_entry_point("extract_item")
        workflow.add_conditional_edges(
            "extract_item",
            decide_strategy,
            {"get_recommendations": "get_recommendations", "end": END}
        )
        workflow.add_edge("get_recommendations", "synthesize_response")
        workflow.add_edge("synthesize_response", END)
        
        self.workflow = workflow.compile()
        logger.info("✓ Mock workflow created")
        return self.workflow
    
    def _parse_graph_results(self, result_text: str) -> List[Dict]:
        recommendations = []
        if "Graph-based novel pairings" in result_text:
            lines = result_text.split('\n')[1:]
            for line in lines:
                if line.strip() and '. ' in line:
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
        recommendations = []
        if "Semantically similar items" in result_text:
            lines = result_text.split('\n')[1:]
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
        if not recommendations:
            return "No graph-based recommendations found."
        
        context = ""
        for i, rec in enumerate(recommendations, 1):
            context += f"{i}. {rec['item']} - {rec.get('details', 'Novel pairing')}\n"
        return context
    
    def _format_semantic_context(self, recommendations: List[Dict]) -> str:
        if not recommendations:
            return "No semantic recommendations available."
        
        context = ""
        for i, rec in enumerate(recommendations, 1):
            context += f"{i}. {rec['item']} - Similar flavor/texture profile\n"
        return context
    
    def query(self, user_query: str) -> str:
        """Main query interface"""
        if not self.workflow:
            return "System not initialized. Run setup_system() first."
        
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
            
            final_state = self.workflow.invoke(initial_state)
            return final_state["final_response"]
            
        except Exception as e:
            logger.error(f"Workflow error: {e}")
            return f"Error processing query: {str(e)}"
    
    def setup_system(self):
        """Complete system setup with mock data"""
        logger.info("Setting up mock LangChain GraphRAG system...")
        
        documents = self.create_mock_documents()
        self.create_vector_index(documents)
        self.initialize_tools()
        self.create_workflow()
        
        logger.info("✓ Mock system setup complete!")
        return self

def main():
    """Test the mock system"""
    print("="*60)
    print("TESTING MOCK LANGCHAIN GRAPHRAG SYSTEM")
    print("="*60)
    
    # Initialize mock system
    processor = MockLangChainGraphRAGProcessor()
    processor.setup_system()
    
    # Test queries
    test_queries = [
        "What goes with Chicken Dum Biryani?",
        "Recommend something for Dosa",
        "What are some pairings for Dal Tadka?",
        "Suggest items that go with Sambar"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = processor.query(query)
        print(f"Response: {response}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("MOCK TESTING COMPLETE")
    print("Now start your Neo4j database and run the real version!")
    print("="*60)

if __name__ == "__main__":
    main()