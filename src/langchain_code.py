from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model = 'llama-3.1-70b-versatile', temperature = 0.5)

poem = llm.invoke("Write a short poem on love")

summary_prompt = PromptTemplate(
    input_variables = ['transcripts'],
    template = """You are a YouTube video explainer. You will recieve a transcript for a video and your task is to summarize it for others to understand. 
    You will output the complete summary of the transcript without missing any detail.
    Here is the transcript:\n {transcript}"""
)
summary_chain = summary_prompt | llm | StrOutputParser()

def get_transcript(video_url):
    video_id = video_url.split("v=")[1]  # Extract video ID from URL
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = " ".join([t['text'] for t in transcript])
    return transcript_text

def summarize_video(transcript):
    summary = summary_chain.invoke({'transcript':transcript})
    return summary

def split_transcript(transcript):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_text(transcript)

#Pinecone
pc = Pinecone(api_key = 'b670b069-a28b-4b19-85bf-bde5481cc688')
index_name = 'youtube-video-chatbot'
# if index_name not in pc.list_indexes():
#     pc.create_index(
#         name = index_name,
#         dimension=384,
#         metric='cosine',
#         spec = ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )

index = pc.Index(index_name) #connecting to index

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_in_pinecone(chunks):
    for i, chunk in enumerate(chunks):
        embeddings = embedder.encode(chunk)
        index.upsert([(str(i), embeddings.tolist())])
    return chunks

def search_pinecone(question):
    query_embedding = embedder.encode(question)
    vec_embed_list = query_embedding.tolist()
    result = index.query(vector = vec_embed_list, top_k=5, include_values=True)
    chunk_ids = [match['id'] for match in result['matches']]
    return chunk_ids

def get_relevant_chunks(chunk_ids, chunks):
    relevant_chunks = [chunks[int(i)] for i in chunk_ids]
    return relevant_chunks

def answer_question(chunks, question):
    # Search Pinecone for relevant chunks
    top_k_chunk_ids = search_pinecone(question)
    relevant_chunks = get_relevant_chunks(top_k_chunk_ids, chunks)

    # Construct context from the most relevant chunks
    context = " ".join(relevant_chunks)

    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an answering bot. You recieve a context and question. The context is taken from a youtube video transcript. 
        You have to answer the user question directly without any additional text other than answer to the question.
        However this doesnt mean that you always have to answer in short. Make use of the context and question to determine how long your answer should be.
        Answer in such a way that anyone can understand it without much difficulty.
        Just make sure to not include any irrelevant text in the answer.
        Here is the context: {context}, 
        and the question: {question}""",
    )

    qa_chain = qa_prompt | llm | StrOutputParser()
    final_ans = qa_chain.invoke({'context':context, 'question':question})

    return final_ans
