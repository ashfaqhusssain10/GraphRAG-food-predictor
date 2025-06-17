from EX2 import LangChainGraphRAGProcessor

# Initialize the system (uses existing Neo4j data)
processor = LangChainGraphRAGProcessor()
processor.setup_system()

# Now you can ask contextual questions!
# response = processor.query("What goes with Chicken Biryani for a Corporate lunch?")
# response = processor.query("Recommend breakfast items that pair with Dosa")
response = processor.query("Suggest items with Mutton Dum Biryani")
response = processor.query("What goes with Chicken Biryani?")
response = processor.query("What goes with Dosa?") 
response = processor.query("What goes with Rice?")
print(response)
processor.close()