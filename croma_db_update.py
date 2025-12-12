from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from tqdm import tqdm
from typing import List, Optional
from sklearn.mixture import GaussianMixture as GMM
import chromadb
import umap
import hashlib
import os
import json
import numpy as np
import pandas as pd
import glob

# --- Ayarlar ---
CHROMA_HOST = "localhost"  # Sadece ana bilgisayar adı
CHROMA_PORT = 8000  # Sadece port numarası
DB_FOLDER = "./database"  # JSON dosyalarının olduğu klasör
COLLECTION_NAME = "raptor_knowledge_base"
model_kwargs = {"device": "cuda"}
encode_kwargs = {
    "normalize_embeddings": True
}  # Cosine similarity için normalizasyon iyidir
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
MAX_BATCH_SIZE = 5000
CLOUDFLARE_TUNNEL_URL = ".../"
OLLAMA_FAST_MODEL = "llama3.1:8b"
LLM_MODEL = ChatOllama(
    model=OLLAMA_FAST_MODEL, base_url=CLOUDFLARE_TUNNEL_URL, temperature=0
)


class RaptorManager:
    def __init__(self):
        print(f"📡 ChromaDB Server'a bağlanılıyor (localhost:8000)...")
        self.client = chromadb.HttpClient(host="localhost", port=8000)

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=EMBEDDING_MODEL,
        )

        # Text Splitter: JSON içeriğini küçük parçalara bölmek için
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50  # MPNet için ideal boyutlar
        )

    def _calculate_file_hash(self, file_path: str) -> str:
        """Dosyanın MD5 hash'ini hesaplar. İçerik değişirse bu hash değişir."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def _load_json_and_split(self, file_path: str) -> List[Document]:
        """JSON dosyasını okur ve Document objelerine çevirir."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # JSON yapına göre burayı düzenleyebilirsin.
            # Varsayım: JSON bir liste veya string içeriyor.
            # Tüm içeriği string'e çevirip split ediyoruz.
            text_content = json.dumps(data, ensure_ascii=False)

            docs = self.text_splitter.create_documents([text_content])
            return docs
        except Exception as e:
            print(f"❌ Hata: {file_path} okunamadı. Sebep: {e}")
            return []

    def _delete_file_from_db(self, filename: str):
        """Verilen dosya adına sahip tüm kayıtları DB'den siler."""
        print(f"🗑️  Eski kayıtlar siliniyor: {filename}")
        try:
            # ChromaDB'den 'original_source' metadata'sı eşleşenleri sil
            # Not: LangChain Chroma wrapper'ında doğrudan delete by metadata bazen zordur,
            # bu yüzden native client kullanıyoruz.
            collection = self.client.get_collection(COLLECTION_NAME)
            collection.delete(where={"original_source": filename})
        except Exception as e:
            print(f"Silme işlemi uyarısı (ilk çalıştırma olabilir): {e}")

    def _check_if_exists_and_current(self, filename: str, current_hash: str) -> bool:
        """Dosya DB'de var mı ve Hash'i güncel mi kontrol eder."""
        try:
            collection = self.client.get_collection(COLLECTION_NAME)
            # Sadece 1 tane örnek kayıt çekip hash kontrolü yapıyoruz
            results = collection.get(
                where={"original_source": filename}, limit=1, include=["metadatas"]
            )

            if len(results["ids"]) > 0:
                stored_hash = results["metadatas"][0].get("file_hash", "")
                if stored_hash == current_hash:
                    return True  # Dosya var ve güncel
            return False  # Dosya yok veya güncel değil
        except Exception:
            return False  # Collection yoksa false döner

    def _add_texts_in_batches(self, texts, metadatas, batch_size=5000):
        """Verileri batch'ler halinde ChromaDB'ye ekler."""
        total_docs = len(texts)
        for i in range(0, total_docs, batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_metadatas = metadatas[i : i + batch_size]
            print(
                f"      ↳ Batch ekleniyor: {i} - {i + len(batch_texts)} / {total_docs}"
            )
            self.vectorstore.add_texts(texts=batch_texts, metadatas=batch_metadatas)

    # --- RAPTOR Core Fonksiyonları (Önceki koddan) ---
    def _cluster_and_summarize(
        self, documents: List[Document], filename: str, file_hash: str
    ):
        """Sadece yeni gelen dokümanlar için RAPTOR ağacı oluşturur."""
        print(f"🚀 RAPTOR Başlatılıyor: {filename} ({len(documents)} chunk)")

        # 1. Katman 0 (Orijinaller) Ekleme
        current_texts = [doc.page_content for doc in documents]
        current_metadatas = []
        for doc in documents:
            meta = doc.metadata.copy()
            meta.update(
                {
                    "layer": 0,
                    "type": "original",
                    "original_source": filename,
                    "file_hash": file_hash,
                }
            )
            current_metadatas.append(meta)

        self._add_texts_in_batches(current_texts, current_metadatas, batch_size=5000)

        # RAPTOR Döngüsü (Basitleştirilmiş max_layer=3)
        max_layers = 3
        for layer in range(1, max_layers + 1):
            embeddings = np.array(EMBEDDING_MODEL.embed_documents(current_texts))

            if len(embeddings) <= 5:
                break  # Yetersiz veri

            # UMAP & GMM
            n_neighbors = min(10, len(embeddings) - 1)
            umap_reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=2,
                metric="cosine",
                random_state=42,
            )
            reduced_emb = umap_reducer.fit_transform(embeddings)

            n_clusters = int(np.sqrt(len(embeddings)))
            gmm = GMM(n_components=n_clusters, random_state=42)
            gmm.fit(reduced_emb)
            labels = gmm.predict(reduced_emb)

            # Özetleme Döngüsü
            df = pd.DataFrame({"text": current_texts, "cluster": labels})
            new_texts = []
            new_metadatas = []

            print(f"   ⚙️  Katman {layer}: {n_clusters} adet küme özetleniyor...")

            unique_clusters = df["cluster"].unique()
            for cluster_id in tqdm(unique_clusters, desc=f"Katman {layer} İlerlemesi"):

                cluster_docs = df[df["cluster"] == cluster_id]["text"].tolist()
                combined_text = "\n".join(cluster_docs)

                # --- GÜVENLİK ÖNLEMİ ---
                # Eğer kümedeki metin çok çok uzunsa (Ollama 8192 token limitini aşarsa)
                # takılma yapabilir. İlk 25.000 karakteri alıp keselim.
                if len(combined_text) > 25000:
                    combined_text = combined_text[:25000]

                # LLM Özetleme
                prompt = ChatPromptTemplate.from_template(
                    "Metinleri Türkçe özetle. Sadece özeti yaz: {context}"
                )
                chain = prompt | LLM_MODEL | StrOutputParser()
                summary = chain.invoke({"context": combined_text})

                new_texts.append(summary)
                new_metadatas.append(
                    {
                        "layer": layer,
                        "type": "summary",
                        "original_source": filename,
                        "file_hash": file_hash,
                        "cluster_id": int(cluster_id),
                    }
                )

            if new_texts:
                # Özetlerde genelde sayı azdır ama garanti olsun diye burayı da değiştirelim
                self._add_texts_in_batches(new_texts, new_metadatas, batch_size=5000)
                current_texts = new_texts
            else:
                break

    def sync_folder(self):
        """Klasörü tarar ve gerekli güncellemeleri yapar."""
        json_files = glob.glob(os.path.join(DB_FOLDER, "*.json"))

        if not json_files:
            print("⚠️ Klasörde .json dosyası bulunamadı.")
            return

        print(f"📂 Bulunan dosyalar: {[os.path.basename(f) for f in json_files]}")

        for file_path in json_files:
            filename = os.path.basename(file_path)
            current_hash = self._calculate_file_hash(file_path)

            # KONTROL: Güncel mi?
            is_synced = self._check_if_exists_and_current(filename, current_hash)

            if is_synced:
                print(f"✅ [ATLANDI] {filename} zaten güncel.")
            else:
                print(f"🔄 [GÜNCELLENİYOR] {filename} değişmiş veya yeni.")

                # 1. Eski veriyi temizle (Eğer varsa)
                self._delete_file_from_db(filename)

                # 2. Dosyayı oku ve parçala
                docs = self._load_json_and_split(file_path)

                if docs:
                    # 3. RAPTOR işlemini başlat
                    self._cluster_and_summarize(docs, filename, current_hash)
                    print(f"🎉 {filename} başarıyla işlendi.")


def rebuild_db(collection_name):
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        # Koleksiyonu sil
        client.delete_collection(name=collection_name)
        print(f"✅ '{collection_name}' koleksiyonu tamamen silindi.")
    except Exception as e:
        print(f"Hata veya koleksiyon zaten yok: {e}")
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
        print(f"Lütfen '{DB_FOLDER}' klasörüne json dosyalarınızı koyun.")
    else:
        manager = RaptorManager()
        manager.sync_folder()


# --- ÇALIŞTIRMA ---
if __name__ == "__main__":
    if not os.path.exists(DB_FOLDER):
        os.makedirs(DB_FOLDER)
        print(f"Lütfen '{DB_FOLDER}' klasörüne json dosyalarınızı koyun.")
    else:
        rebuild_db(collection_name=COLLECTION_NAME)
