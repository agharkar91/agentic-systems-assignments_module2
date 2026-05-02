from chromadb import chromadb
from openai import OpenAI
from typing import List

#1. creating open ai client -
openai_client = OpenAI()

#2.  setting up embedding and LLM models
EMBEDDING_MODEL="text-embedding-3-small"
LLM_MODEL = "gpt-5.4"

#3. creating sample policy documents
POLICY_DOCUMENTS =[
    {
        "id": "POL-LV-2024-001",
        "text": "Full-time employees are entitled to 25 days of paid annual leave per calendar year, accrued on a monthly basis. Sick leave requires a medical certificate for absences exceeding two consecutive business days to ensure proper health management. A maximum of 5 unused leave days can be carried forward to the next year, provided they are utilized by March 31st.",
        "metadata": {
            "category": "Leave Policy",
            "source": "Employee Handbook Section 4"
        }
    },
    {
        "id": "POL-WFH-2024-002",
        "text": "Employees are eligible to work from home for up to three days per week, subject to departmental requirements and performance standards. Remote work requests must be submitted through the internal portal and approved by the immediate supervisor at least 48 hours in advance. To maintain productivity, staff must ensure a stable internet connection and remain available via standard communication channels during core business hours.",
        "metadata": {
            "category": "Work From Home Policy",
            "source": "IT & Remote Operations Manual"
        }
    },
    {
        "id": "POL-APP-2024-003",
        "text": "The performance appraisal cycle occurs annually every January to evaluate achievements against set objectives from the previous fiscal year. Performance is measured on a 5-point rating scale, ranging from 'Unsatisfactory' to 'Exceptional.' Final ratings directly influence annual salary increments and eligibility for internal promotion opportunities within the firm.",
        "metadata": {
            "category": "Appraisal Policy",
            "source": "Performance Management Framework"
        }
    },
    {
        "id": "POL-COC-2024-004",
        "text": "All employees must maintain the highest standards of professional behavior, fostering a diverse and inclusive environment free from harassment. Confidential company data must be protected at all times, and any potential conflicts of interest must be disclosed to HR immediately. Failure to adhere to these ethical standards may result in disciplinary action up to and including termination of employment.",
        "metadata": {
            "category": "Code of Conduct",
            "source": "Corporate Ethics Policy"
        }
    }
]

# 4. create vector embeddings 
def create_embeddings(text: List[str]) -> List[List[float]] :
    responses = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    embeddings = [item.embedding for item in responses.data]

    return embeddings

# 5. set vector db
def setup_vectorDb():
    chroma_client= chromadb.PersistentClient("./chroma_policy_db")
    collection = chroma_client.get_or_create_collection(
        name="ecommerce_policy_collection",
        embedding_function=None
    )
    print("vector db setup successfully")
    return collection

#6. store documents in vector db
def index_hr_documents(collection):
    ids = [document["id"] for document in POLICY_DOCUMENTS]
    documents = [document["text"] for document in POLICY_DOCUMENTS]
    metadatas = [document["metadata"] for document in POLICY_DOCUMENTS]

    embeddings = create_embeddings(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )
    print("documents stored in vector db successfully")

#7. retrieve policy content
def retrieve_hr_content(query: str, collection, top_k: int = 3):

    #query embedding
    query_embedding = create_embeddings(query)[0]

    #similarity search
    results= collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    print("resultset received from vector db")
    return results["documents"][0]

# 8. build prompt to generate an answer
def build_grounded_prompt(query: str, retrieved_documents):
    context = ""
    number_of_retriever_documents = len(retrieved_documents)
    for index in range(0, number_of_retriever_documents):
        document = retrieved_documents[index]

        context += f"Policy document {index} | {document}\n" 
    
    prompt = f""" 
    
        You are a helpful HR policy support assistant for an e-commerce company.

        Answer the customer's question using ONLY the policy context provided below.

        Rules:
        1. Do not make up policy details.
        2. If the answer is not present in the context, say:
        "I do not have enough information in the provided policy documents."
        3. Keep the answer simple, clear, and customer-friendly.
        4. Mention important conditions or exceptions if they are present in the context.

        {context} 
        
        {query}
    
    """
    print("prompt built with context and query")

    return prompt

#9. generate final answer using the context
def generate_final_answer(query: str, retrieved_documents):
    prompt = build_grounded_prompt(query,retrieved_documents)
    response = openai_client.responses.create(
        model=LLM_MODEL,
        input=prompt
    )

    return response.output_text

# 10. Resolve the user query with RAG.
def answer_with_RAG(query:str, collection, top_k: int = 3):
    retrieved_documents = retrieve_hr_content(query,collection,top_k)
    print("doc content retrieved from vector db")
    answer = generate_final_answer(query,retrieved_documents)
    return answer

#11. resolve the user query without RAG
def generate_answer_without_retrieval(query:str):
    response = openai_client.responses.create(
        model=LLM_MODEL,
        input=query
    )
    return response.output_text

def main():
    collection = setup_vectorDb()
    index_hr_documents(collection)
    user_query1 = "How many days of annual leave am I entitled to per year?"
    answer1RAG= answer_with_RAG(user_query1,collection,3)
    user_query2 ="Do I need manager approval before working from home?"
    answer2RAG= answer_with_RAG(user_query2,collection,3)
    user_query3 ="When is the appraisal cycle conducted and how is the increment decided?"
    answer3RAG= answer_with_RAG(user_query3,collection,3)
    print(f"user_query1_RAG : {answer1RAG}")
    print(f"user_query2_RAG : {answer2RAG}")
    print(f"user_query3_RAG : {answer3RAG}")
    answer3withoutRAG= generate_answer_without_retrieval(user_query3)
    print(f"user_query3_withoutRAG : {answer3withoutRAG}")

main()
