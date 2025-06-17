"""
Debug Script: Test What the Graph and Vector Tools Are Actually Returning
This will help us understand if the tools are working and what data they're providing
"""

from EX import LangChainGraphRAGProcessor

def debug_tools():
    """Test what the individual tools are returning"""
    print("="*80)
    print("DEBUGGING GRAPH AND VECTOR TOOLS")
    print("="*80)
    
    # Initialize the system
    processor = LangChainGraphRAGProcessor()
    processor.setup_system()
    
    # Test item
    test_item = "Chicken Biryani"
    print(f"\nTesting with: {test_item}")
    print("-" * 50)
    
    # Test Graph Tool directly
    print("\n🔹 TESTING GRAPH RECOMMENDATION TOOL:")
    print("-" * 40)
    try:
        graph_result = processor.graph_tool._run(test_item, top_k=5)
        print("Raw Graph Tool Output:")
        print(repr(graph_result))  # repr shows the exact string with \n characters
        print("\nFormatted Graph Tool Output:")
        print(graph_result)
        
        # Test parsing
        parsed_graph = processor._parse_graph_results(graph_result)
        print(f"\nParsed Graph Results ({len(parsed_graph)} items):")
        for i, rec in enumerate(parsed_graph, 1):
            print(f"  {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Graph Tool Error: {e}")
    
    # Test Semantic Tool directly  
    print("\n🔹 TESTING SEMANTIC RECOMMENDATION TOOL:")
    print("-" * 40)
    try:
        semantic_result = processor.semantic_tool._run(test_item, top_k=5)
        print("Raw Semantic Tool Output:")
        print(repr(semantic_result))
        print("\nFormatted Semantic Tool Output:")
        print(semantic_result)
        
        # Test parsing
        parsed_semantic = processor._parse_semantic_results(semantic_result)
        print(f"\nParsed Semantic Results ({len(parsed_semantic)} items):")
        for i, rec in enumerate(parsed_semantic, 1):
            print(f"  {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Semantic Tool Error: {e}")
    
    # Test what gets passed to LLM
    print("\n🔹 TESTING CONTEXT FORMATTING:")
    print("-" * 40)
    try:
        # Simulate what happens in the workflow
        graph_result = processor.graph_tool._run(test_item, top_k=5)
        semantic_result = processor.semantic_tool._run(test_item, top_k=5)
        
        parsed_graph = processor._parse_graph_results(graph_result)
        parsed_semantic = processor._parse_semantic_results(semantic_result)
        
        graph_context = processor._format_graph_context(parsed_graph)
        semantic_context = processor._format_semantic_context(parsed_semantic)
        
        print("Graph Context (what goes to LLM):")
        print(repr(graph_context))
        print("\nSemantic Context (what goes to LLM):")  
        print(repr(semantic_context))
        
        print("\n🔹 FINAL PROMPT PREVIEW:")
        print("-" * 40)
        prompt_preview = f"""
NOVEL PAIRINGS (from customer ordering patterns):
{graph_context}

SIMILAR ITEMS (based on flavor/texture analysis):
{semantic_context}
"""
        print(prompt_preview)
        
    except Exception as e:
        print(f"❌ Context Formatting Error: {e}")
    
    # Clean up
    processor.close()
    print("\n" + "="*80)
    print("DEBUG TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    debug_tools()