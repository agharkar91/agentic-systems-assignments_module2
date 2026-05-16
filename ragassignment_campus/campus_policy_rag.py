from pathlib import Path
import re
from typing import Any, Dict, List

from chromadb import chromadb
from openai import OpenAI
from pypdf import PdfReader

#create open ai api cient
openai_client = OpenAI()

#set LLM model and embedding model
EMBEDDING_MODEL="text-embedding-3-small"
LLM_MODEL = "gpt-5.4"

#3. infer policy type
def infer_policy_type(file_Name:str)->str:
    if "refund" in file_Name:
        return "refund"
    if "hostel" in file_Name:
        return "hostel"
    return "library"


#4.setup vector db
def setup_vectordb():
    chroma_client = chromadb.PersistentClient("./campyspolicy.db")
    collection = chroma_client.get_or_create_collection("campus_policies")
    print("Vector DB ready. Collection: campus_policies")
    return collection

#5.create embeddings of vector chunks
def create_embeddings(text: List[str])->List[List[float]]:
    responses = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    embeddings = [item.embedding for item in responses.data]

    return embeddings

def clean_text(text:str)->str:
    if not text:
        return ""
    else:
        text = text.replace("\n"," ")
        text=re.sub(r"\s+", " ", text)
        return text.strip()

#6: serialize a document dictionary
#every doc in rag pipeline will have text and metdata
def create_doc_dictionary(text: str,metadata:Dict[str,Any])->Dict[str,Any]:
    return {
        "text":text,
        "metadata":metadata
    }

#7. load pdf files from the folder and convert them into List of strings
def load_policy_file(file_Path:Path)->List[Dict[str,Any]]:
    reader = PdfReader(file_Path)
    documents = []
    page_count=0
    for page_number,page  in enumerate(reader.pages,start=1):
        page_count=page_number
        page_text = page.extract_text()
        clear_text = clean_text(page_text)
        if clear_text:
            documents.append(create_doc_dictionary(
                text=clear_text,
                metadata={
                    "source": str(file_Path),
                    "source_type":"pdf",
                    "page_number":page_number,
                    "policy_type":infer_policy_type(file_Path.name)
                })
            )
        

    print(f"{page_count} pages from: {file_Path.name} ")
    return documents

#8. load all files from folder
def load_files_from_folder(folder_Path:str)->List[Dict[str, Any]]:
    base_path = Path(__file__).resolve().parent
    folder_Path=base_path / folder_Path
    print(f"folder path: {folder_Path}")
    folder = Path(folder_Path)
    
    if not folder.exists():
        raise FileNotFoundError(f"file no found : {folder_Path} ")
    
    all_docs=[]
    for filePath in folder.iterdir():
        if filePath.is_file() and filePath.suffix.lower()==".pdf":
            all_docs.extend(load_policy_file(filePath))
    return all_docs

#9. chunking - breaking down a document into smaller word based chunks for embedding into vector db
#  Example:
#     chunk_size_words = 120
#     overlap_words = 30
#     Chunk 1: words 1 to 120
#     Chunk 2: words 91 to 210
def chunk_text(text:List[str],chunk_size:int=120,overlap_words:int=30)->List[str]:
    words = text.split()
    if not words:
        return []
    chunks=[]
    start=0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end >= chunk_size:
            break
        start = end - overlap_words
    return chunks
    
#10. convert chunks to chunks_metadata
def create_chunks_withmetadata(documents:List[Dict[str, Any]],chunk_size:int=120, overlap_words:int=30):
    all_documents=[]
    for document in documents:
        text=document["text"]
        metadata=document["metadata"]
        text_chunk = chunk_text(text,chunk_size,overlap_words)
        for index,chunk in enumerate(text_chunk):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"]=index
            chunk_metadata["chunk_size_words"]=chunk_size
            chunk_metadata["chunk_overlap_words"]=overlap_words
            source = chunk_metadata["source"]
            chunk_id=f"{source}_{index}_{chunk}"
            all_documents.append({
                "id":chunk_id,
                "text":chunk,
                "metadata":chunk_metadata
            })

    return all_documents

#11. index chunks - create embeddings of chunks
def index_chunks(collection, chunks:List[Dict[str,Any]],batch_size:int=50)->None:
    if not chunks:
        print("no chunks to index")
        return
    for start in range(0,len(chunks),batch_size):
        batch = chunks[start:start+batch_size]
        ids= [chunk["id"] for chunk in batch]
        texts = [chunk["text"] for chunk in batch]
        metadatas = [chunk["metadata"] for chunk in batch]
        embeddings = create_embeddings(texts)
        collection.upsert(
            ids= ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        print(f" indexed batch { start // batch_size + 1 } : {len(batch)} chunks") # show batch no and length 

#12. complete ingestion pipeline
# load documents --> clean --> chunk --> batch --> embed --> store into vector db
def build_knowledge_base(collection, folder_Path:str,chunk_size:int=120,overlap_size:int=30):
    print("\nLoading documents...")
    all_pdf_docs = load_files_from_folder(folder_Path)
    print("\nFiles loaded from folder...")
    chunks = create_chunks_withmetadata(all_pdf_docs,chunk_size,overlap_size)
    print(f"Total chunks created: {len(chunks)}")
    index_chunks(collection,chunks)
    print(f"Successfully stored {len(chunks)} chunks in vector database.")
    print("knowledge base creation completed.")

#13. retreive relevant chunks based on query
def retrieve_relevant_chunks(collection,query:str, top_k:int=4, policy_type:str=None):
    query_embeddings = create_embeddings(query)[0]
    #similarity search
    result = collection.query(
        query_embeddings=[query_embeddings],
        n_results = top_k,
        include=["documents","metadatas","distances"]
    )

    retrieved_docs=[]
    documents = result["documents"]
    metadatas=result["metadatas"]
    distances=result["distances"]

    for document,metadata,distance in zip(documents,metadatas,distances):
        retrieved_docs.append({
            "text":document,
            "metadata":metadata,
            "distance":distance
        })
    return retrieved_docs

#14. build a prompt based on retreived chunks as context
def build_prompt(query:str,retrieved_chunks:List[Dict[str,Any]])->Any:
    context_parts=[]
    for index,chunk in enumerate(retrieved_chunks,start=1):
        metadata = chunk["metadata"]
        source = metadata[index].get("source","unknown")
        policy_type=metadata[index].get("policy_type","general")
        context_parts.append(
            f"""
            Policy Chunk {index}
            Source {source}
            Policy Type {policy_type}
            Content : {chunk["text"]}
            """
        )
    context = "\n".join(context_parts)

    prompt=f"""
        You are a helpful customer support assistant for an e-commerce company.

        Use ONLY the policy context below to answer the customer's question.

        Important rules:
        1. Do not make up policy details.
        2. If the answer is not present in the policy context, say:
        "I do not have enough information in the provided policy documents."
        3. Keep the answer simple and customer-friendly.
        4. Mention important conditions, timelines, and exceptions if they are present.
        5. Do not mention internal chunk numbers to the customer.
    
    Policy Context: {context}

    Customer Questin: {query}

    Final Answer: 

    """
    return prompt

#15. answer with RAG
def answer_with_RAG(query:str,collection, top_k:int=4, policy_type: str = None)->str:
    retrieved_chunks=retrieve_relevant_chunks(collection,query,top_k,policy_type)
    prompt = build_prompt(query,retrieved_chunks)
    response = openai_client.responses.create(
        model=LLM_MODEL,
        instructions="You are a precise and helpful e-commerce customer support assistant.",
        input=prompt
    )

    answer = response.output_text

    return answer

def main():
    collection = setup_vectordb()
    build_knowledge_base(collection,"policy_documents")
    user_query="Can I get a refund after dropping a course?"
    policy_type="refund"
    answer = answer_with_RAG(user_query,collection,4,policy_type)
    print(f"chatbot answer :{answer}")

main()
