#py -m pip install chromdb
#py -m pip install sentence_transformers
#python 3.14.2

from chromadb import chromadb
from sentence_transformers import SentenceTransformer

#data to be embedded
hr_policy_records=[
    {
        "id": "hrdoc1",
        "text": "All employees are entitled to 22 casual leaves per year.",
        "metadata": {"category": "leave", "version": 1 }
    },
    {
        "id": "hrdoc2",
        "text": "All employees are entitled to 12 sick leaves per year.",
        "metadata": {"category": "leave", "version": 1 }
    },
    {
        "id": "hrdoc3",
        "text": "All employees have mediclaim cover of Rs 25 lakh for self and 3 family members. Premium Rs 5000 per annum.",
        "metadata": {"category": "benefit", "version": 2 }
    },
    {
        "id": "hrdoc4",
        "text": "All employees are expected to wear business formals on Mon-Thurs and smart casuals on Fri.",
        "metadata": {"category": "conduct", "version": 1 }
    },
    {
        "id": "hrdoc5",
        "text": "Employees are expected to report to work between 8.30 am to 9.30 am and complete 9 hours shift",
        "metadata": {"category": "conduct", "version": 3 }
    }
]

#load embedding model
model=SentenceTransformer("all-MiniLM-L6-v2")

# converting documents into vector embeddings
hrdocuments = [record["text"] for record in hr_policy_records]
hrdoc_ids = [record["id"] for record in hr_policy_records]
hrdoc_metaDatas = [record["metadata"] for record in hr_policy_records]

hrdocuments_embeddings=model.encode(hrdocuments,convert_to_numpy=True).tolist()
#encodes into list of vector embeddings
#[[.....],[.....],[......]... ,[.....]]

#create a client for chroma DB
client = chromadb.PersistentClient(path="./chroma_DB")

#store embeddings into chromaDB
collection = client.get_or_create_collection(
    name="hrpolicy_vector_db",
    embedding_function=None
)

#store embeddings in collection 
#upsert - update existing and insert new

collection.upsert(
    ids=hrdoc_ids,
    documents=hrdocuments,
    metadatas=hrdoc_metaDatas,
    embeddings=hrdocuments_embeddings
)

print("number of records in collecton: ",collection.count())
print("single record from collection: ",collection.peek(1))

# search A - natual language query search
user_query="how many leave do we get if we are ill ?"

#converting query text into vector embeddings
hrquery_embeddings=model.encode([user_query],convert_to_numpy=True).tolist()

result = collection.query(
    query_embeddings=hrquery_embeddings,
    n_results=3,
    include=["metadatas","data","documents"]
)
print("search A results: ",result)

#search B - semantic search with a where condition
hr_userquery_B="what are the working hours ?"
hrquery_embeddingsB=model.encode([hr_userquery_B],convert_to_numpy=True).tolist()

resultB = collection.query(
    query_embeddings=hrquery_embeddingsB,
    n_results=2,
    where={"category":"conduct"},
    include=["metadatas","data","documents"]
)
print("search B results: ",resultB)

#update existing record and add new record
hr_record_upsertdoc=[
    {
        "id": "hrdoc3",
        "text": "All employees have mediclaim cover of Rs 30 lakh for self and 3 family members. Premium Rs 5500 per annum.",
        "metadata": {"category": "benefit", "version": 3 }
    },
    {
        "id": "hrdoc6",
        "text": "Employee will get subsidized executive lunch at Rs 50 per day and Rs 500 per month.",
        "metadata": {"category": "benefit", "version": 1 }
    }
]
# converting new documents into vector embeddings
hrdocs_new = [record["text"] for record in hr_record_upsertdoc]
hrdoc_ids_new = [record["id"] for record in hr_record_upsertdoc]
hrdoc_metaDatas_new = [record["metadata"] for record in hr_record_upsertdoc]

hr_record_upsertdoc_embeddings=model.encode(hrdocs_new,convert_to_numpy=True).tolist()

#store embeddings in collection 
#upsert - update existing and insert new

collection.upsert(
    ids=hrdoc_ids_new,
    documents=hrdocs_new,
    metadatas=hrdoc_metaDatas_new,
    embeddings=hr_record_upsertdoc_embeddings
)

user_queryB="tell me about mediclaim and canteen"
user_queryB_embeddings=model.encode([user_queryB],convert_to_numpy=True).tolist()


print("record count post upsert: ",collection.count())
print("updated and inserted records in category: benefit: ",collection.query(   
    query_embeddings=user_queryB_embeddings,
    n_results=2,
    where={"category":"benefit"}
))