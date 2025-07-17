import os
import json
import random
import numpy as np
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# Load environment variables
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key="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")
parser = StrOutputParser()

# Load predefined intents
with open("intents.json", "r", encoding="utf-8") as f:
    predefined_data = json.load(f)["intents"]

# Create a mapping of tags to patterns
tag_to_patterns = {intent["tag"]: intent["patterns"] for intent in predefined_data}
predefined_patterns = [pattern for patterns in tag_to_patterns.values() for pattern in patterns]

# Load sentence transformer model for semantic similarity
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
pattern_embeddings = embedding_model.encode(predefined_patterns)

# Optimized system prompt enforcing strict pattern matching
system_template = (
    "Match the user input exactly to one of the predefined patterns from the provided list. "
    "Analyze the full sentence, ensuring contextual relevance, and return only the exact matching pattern. "
    "Do NOT modify the patterns, generate new responses, or deviate from the provided patterns. "
    "Use the associated tags to improve accuracy in selecting the correct pattern. "
    "If no exact match is found, return the closest matching pattern from the predefined list."
)

prompt = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

def find_best_match(user_input):
    """
    Finds the closest matching pattern using semantic similarity.
    """
    user_embedding = embedding_model.encode([user_input])
    similarities = np.dot(pattern_embeddings, user_embedding.T).flatten()
    best_match_idx = np.argmax(similarities)
    return predefined_patterns[best_match_idx]

def match_question(user_input):
    """
    Matches user input against predefined patterns using tags for better accuracy.
    """
    # Check for an exact match first
    if user_input in predefined_patterns:
        return user_input
    
    # Use LLM only if no exact match
    chain = prompt | llm | parser
    matched_pattern = chain.invoke({"text": user_input}).strip()
    
    if matched_pattern in predefined_patterns:
        return matched_pattern
    
    # Use semantic similarity as a fallback
    return find_best_match(user_input)

# Example usage
if __name__ == "__main__":
    user_query = "how can i know whether i have depression or not?"
    response = match_question(user_query)
    print(response)
