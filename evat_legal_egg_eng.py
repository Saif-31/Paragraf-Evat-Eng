import os
import torch
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_mistralai import ChatMistralAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Set up required environment variables with error handling
required_env_vars = ["MISTRAL_API_KEY", "PINECONE_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Mistral system prompt
system_prompt = """Welcome to Paragraf Lex! I'm here to help you with any questions you have about VAT and e-invoicing in Serbia. How can I help you today?

Role Description:

I am a virtual assistant from Paragraf Lex, specializing in e-invoicing and Value Added Tax (VAT) legislation in the Republic of Serbia, using information from the Paragraf online legal library. My goal is to provide users with clear, detailed, and accurate information that surpasses previous examples of quality.

Response Guidelines:

Article Integration: I will use relevant parts of the provided articles (segments) related to the user's question. I will quote or reference specific sections of laws, articles, or clauses from these articles when necessary.

Response Structure:

Brief Introduction: I will confirm my understanding of the question.

Detailed Answer: I will provide comprehensive and easy-to-understand information, referencing the provided articles and regulations.

Legal References: I will cite specific laws, articles, and clauses when relevant.

Conclusion: I will offer additional assistance or clarification if needed.

Error Prevention:

I will verify the accuracy of the information before providing it.

I will avoid making assumptions; if information is missing, I will politely ask for clarification.

I will not provide inaccurate or outdated information.

Scope of Response:

Allowed Topics: E-invoicing, VAT, relevant Serbian laws and regulations.

Disallowed Topics: Questions unrelated to e-invoicing or VAT in Serbia. For such questions, I will politely explain this limitation.

Communication Style:

I will be professional, friendly, and approachable.

I will use simple language accessible to users without legal or accounting knowledge.

I will clearly explain technical terms.

Language Consistency: I will always respond in English.

Article Integration (Segments):

When a user asks a question, the system will provide relevant articles from the Paragraf online legal library as contextual data (segments) which I will use to formulate my response.

Notes:

I will combine information from the provided data (segments), my knowledge, and relevant laws to provide the most accurate answer.

I will always consider the latest amendments and updates to laws and regulations.

I will present the information as complete answers without mentioning the use of segments or internal sources.

Goal:

My goal is to provide users with the highest quality and most detailed information so they can understand and fulfill their legal obligations related to e-invoicing and VAT in the Republic of Serbia."""

# Initialize Mistral LLM (with fixed system message)
llm = ChatMistralAI(model="mistral-large-latest")

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")
index = pc.Index("electronicinvoice1")

# Hugging Face embeddings for text similarity
embedding_function = HuggingFaceEmbeddings(
    model_name="djovak/embedic-base",
    model_kwargs={'device': 'cpu'} if not torch.cuda.is_available() else {'device': 'cuda'}
)

# Pinecone Vectorstore
vectorstore = PineconeVectorStore(index=index, embedding=embedding_function, text_key='text', namespace="text_chunks")

# Retriever for semantic search
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

refinement_template = """Create a focused Serbian search query for the RAG retriever bot. Convert to Serbian language if not already. Include key terms, synonyms, and domain-specific vocabulary. Remove filler words. Output only the refined query in the following format: {{refined_query}},{{keyterms}},{{synonyms}}

Query: {original_question}

Refined Query:"""

refinement_prompt = PromptTemplate(input_variables=["original_question"], template=refinement_template)

# LLM Chain for refinement
refinement_chain = refinement_prompt | llm

# Combined Retrieval Prompt with Mistral
combined_prompt = ChatPromptTemplate.from_template(
    f"""{system_prompt}

    Please answer the following question using only the context provided:
    {{context}}

    Question: {{question}}
    Answer:"""
)

# RetrievalQA Chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": combined_prompt}
)

# Processing Query
def process_query(query: str):
    try:
        # Step 1: Refine Query with Mistral
        refined_query = refinement_chain.invoke({"original_question": query}).content

        # Step 2: Retrieve and Answer
        response = retrieval_chain.invoke({"query": refined_query})
        return response.get("result", "") if isinstance(response, dict) else str(response)
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI - Simplified Version
st.title("Legal Egg AI 🥚")

st.write("Welcome to Serbian E-Invoice and VAT Bot! I'm a comprehensive AI-powered Agent automating VAT compliance, E-invoicing workflows, and regulatory reporting for businesses operating in Serbia. Seamlessly navigate complex tax regulations, ensure precise digital invoicing, and minimize administrative overhead with real-time, intelligent guidance.")

# Sidebar with example questions and clear chat button
with st.sidebar:
    st.header("Common Queries")
    example_questions = [
        "1. When did e-invoicing become mandatory for B2B/B2G transactions in Serbia?",
        "2. What format is required for Serbian e-invoices?",
        "3. How long must e-invoices be stored in Serbia?",
        "4. What penalties apply for non-compliance with e-invoicing rules?",
        "5. Are cross-border transactions subject to Serbian e-invoicing?",
        "6. What are the deadlines for recording input VAT in SEF?",
        "7. Do e-invoices require a digital signature in Serbia?",
        "8. Are there exemptions to Serbia’s e-invoicing mandate?",
        "9. How to correct errors in e-invoices or VAT records?",
        "10. How do businesses register for the SEF platform?"
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    if st.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Manage chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User input
if prompt := st.chat_input("Ask your question..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.chat_message("assistant"):
        response = process_query(prompt)
        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})