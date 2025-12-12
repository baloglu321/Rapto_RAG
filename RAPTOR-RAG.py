import chromadb
from typing import List, Dict
from croma_db_update import RaptorManager
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time

# --- AYARLAR ---
CLOUDFLARE_TUNNEL_URL = ".../"
OLLAMA_MAIN_MODEL = "gemma3:27b"
COLLECTION_NAME = "raptor_knowledge_base"
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000

# 1. EMBEDDING (İndeksleme ile AYNI olmak zorunda!)
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs={"device": "cpu"},  # Veya 'cuda'
    encode_kwargs={"normalize_embeddings": True},
)

# 2. LLM (Cevap veren model - Burada daha zeki bir model seçebilirsin)
# Llama 3.2 3B hızlıdır ama cevap kalitesi için imkanın varsa daha büyüğünü kullan.
LLM_MODEL = ChatOllama(
    base_url=CLOUDFLARE_TUNNEL_URL, model=OLLAMA_MAIN_MODEL, temperature=0.1
)


class RaptorRAG:
    def __init__(self):
        # Server'a bağlan
        self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

        # LangChain Vektör Store bağlantısı
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDING_MODEL,
        )

        # Retriever Ayarı
        # k=10 yapıyoruz çünkü hem özetleri hem detayları yakalamak istiyoruz.
        # LLM'e giden context zengin olsun.
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": 15, "fetch_k": 50}
        )

        # RAG Prompt'u
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question using only the sources provided in the "Context" section below.

                    Rules:
                    1. Use ONLY the information provided in the context.
                    2. Cite the source file name at the end of the sentence or paragraph in the format [filename].
                    Example: "...this happened in 1990 [train-v1.1.json]."
                    3. If the answer is not in the context, say "The provided sources do not contain this information."

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
        )

    def format_docs(self, docs):
        """
        Dokümanları birleştirirken başlarına dosya isimlerini ekler.
        Örn:
        [squad-tr.json] (Layer: 0)
        Metin içeriği...
        """
        formatted_context = []
        for doc in docs:
            # Metadata'dan dosya ismini çek
            source_file = doc.metadata.get("original_source", "Bilinmeyen Dosya")
            layer = doc.metadata.get("layer", 0)

            # Etiketi dosya ismi olacak şekilde ayarla
            # LLM bu köşeli parantez içindeki ismi kullanacak
            header = f"[{source_file}] (Layer: {layer})"

            # Metni temizle
            content = doc.page_content.replace("\n", " ")

            entry = f"{header}\n{content}"
            formatted_context.append(entry)

        return "\n\n---\n\n".join(formatted_context)

    def ask(self, question: str):
        print(f"\n❓ Soru: {question}")
        start_time = time.time()
        # 1. Retrieval (Getirme)
        docs = self.retriever.invoke(question)

        # Kaynakları göster (Debug için çok faydalı)
        print(f"🔍 Bulunan Doküman Sayısı: {len(docs)}")
        for i, doc in enumerate(docs[:3]):  # İlk 3 kaynağı ekrana bas
            layer = doc.metadata.get("layer")
            print(f"   {i+1}. [Katman {layer}] {doc.page_content[:100]}...")

        # 2. Generation (Cevap Üretme)
        rag_chain = (
            {
                "context": lambda x: self.format_docs(docs),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | LLM_MODEL
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        stop_time = time.time()
        answer_time = stop_time - start_time
        print(f"\n🤖 Cevap:\n{answer}")
        print(f"⏱️  Cevap Süresi: {answer_time:.2f} saniye")
        return answer


# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    bot = RaptorRAG()
    manager = RaptorManager()
    manager.sync_folder()
    bot.ask(
        "What is the location of the grotto that the University of Notre Dame's grotto is a replica of, where the Virgin Mary allegedly appeared in 1858?",
    )

    bot.ask(
        "What is the metric term less used than the Newton, and what is it sometimes referred to?",
    )

    bot.ask(
        "Normanların Fransa'daki kültürel ve idari dönüşüm sürecini; dil, din ve feodal yapı bağlamında özetleyerek anlat.",
    )

    bot.ask(
        "Give me a high-level overview of the topics covered in the 'train-v1.1.json' file based on the available documents.",
    )
    bot.ask(
        "Summarize the key outcomes of Super Bowl 50 and explain how the NFL altered its traditional naming conventions to celebrate this specific 'golden anniversary'.",
    )

    bot.ask(
        "Why were the traditional Roman numerals (L) not used for Super Bowl 50?",
    )
