import os
import re
import pickle
import fitz
import faiss
import numpy as np
from llama_cpp import Llama
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5ForConditionalGeneration, AutoTokenizer
from neo4j import GraphDatabase
from langdetect import detect

# === Setup T5 Translation Model ===
def setup_translation():
    tokenizer = AutoTokenizer.from_pretrained(
        'mesolitica/translation-t5-tiny-standard-bahasa-cased',
        use_fast=False
    )
    model = T5ForConditionalGeneration.from_pretrained(
        'mesolitica/translation-t5-tiny-standard-bahasa-cased'
    )
    return model, tokenizer

model_t5, tokenizer_t5 = setup_translation()

# Translate text using T5
def translate_text(text, target_lang='ms'):
    input_ids = tokenizer_t5.encode(f'terjemah ke {target_lang}: {text}', return_tensors='pt')
    outputs = model_t5.generate(input_ids, max_length=512)
    outputs = [i for i in outputs[0] if i not in [0, 1, 2]]
    return tokenizer_t5.decode(outputs, spaces_between_special_tokens=False)

# === PDF & Chunking ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_faq_section(text):
    match = re.search(r"(Soalan Lazim\s*\(FAQ\).*)", text, re.DOTALL)
    if match:
        return text[:match.start()], match.group(1)
    return text, ""

def chunk_general_text(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)

def extract_faq_chunks(faq_text):
    qa_pairs = re.findall(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\nQuestion:|$)", faq_text, re.DOTALL)
    return [f"Question: {q.strip()}\nAnswer: {a.strip()}" for q, a in qa_pairs]

# === FAISS Index ===
def build_faiss_index(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, chunks, embedding_model, index, k=2):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype(np.float32)
    scores, indices = index.search(query_embedding, k)
    return scores, indices, [chunks[i] for i in indices[0]]

# === Prompt + LLM ===
def build_prompt(context_chunks, query):
    context = "\n".join(context_chunks)
    return f"""Jawab soalan berikut berdasarkan konteks yang diberi:

Konteks:
{context}

Soalan:
{query}

Jawapan:"""

def ask_llm(prompt, llm):
    try:
        response = llm(prompt, max_tokens=512, stop=["\n\n"], echo=False)
        return response["choices"][0]["text"].strip()
    except Exception as e:
        return f"Ralat dari LLM: {str(e)}"

def preprocess_query(query, synonyms):
    for word, replacement in synonyms.items():
        query = query.replace(word, replacement)
    return query

def log_qa_to_csv(query, answer, filepath="qa_log.csv"):
    new_row = {
        "timestamp": datetime.now().isoformat(),
        "question": query,
        "answer": answer,
        "status": "unchecked"
    }
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(filepath, index=False)

# === Neo4j ===
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def is_kg_question(query):
    kg_keywords = ["kaum", "Malaysia", "tergolong", "berasal", "berasal dari", "negeri",
                    "makanan", "makan", "menikmati", "dinikmati",
                    "pakaian", "memakai", "dipakai",
                    "alat muzik", "muzik", 
                    "bermain", "memainkan", "dimainkan", "dimain", "permainan", 
                    "sambutan", "menyambut", "perayaan", "disambut",
                    "tarian", "menari", 
                    "bahasa", "bercakap", "bertutur", "cakap", "ditutur",
                    "kepercayaan", "agama", "dipercayai", "percaya",
                    "suku", "suku kaum"]
    
    return any(word in query.lower() for word in kg_keywords)

def generate_cypher(query):
    query = query.lower()

    match = re.search(r"kaum\s+(\w+)", query)
    ethnic = match.group(1).capitalize() if match else None
    if not ethnic:
        return None, None

    if any(word in query for word in ["makanan", "makan", "menikmati", "dinikmati"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MENIKMATI]->(m:Makanan) RETURN m.nama",
            f"Makanan yang dinikmati oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["pakaian", "memakai", "dipakai"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MEMAKAI]->(p:Pakaian) RETURN p.nama",
            f"Pakaian yang dipakai oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["alat muzik", "muzik"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MEMAINKAN]->(a:AlatMuzik) RETURN a.nama",
            f"Alat muzik yang dimainkan oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["permainan", "bermain", "memainkan", "dimainkan", "dimain"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:BERMAIN]->(g:Permainan) RETURN g.nama",
            f"Permainan yang dimainkan oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["perayaan", "menyambut", "sambutan", "disambut"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MENYAMBUT]->(f:Perayaan) RETURN f.nama",
            f"Perayaan yang disambut oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["tarian", "menari"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MENARI]->(t:Tarian) RETURN t.nama",
            f"Tarian yang dipersembahkan oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["bahasa", "bercakap", "bertutur", "cakap", "ditutur"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:BERTUTUR]->(b:Bahasa) RETURN b.nama",
            f"Bahasa yang dituturkan oleh kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["kepercayaan", "agama", "dipercayai", "percaya"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:PERCAYA_KPD]->(r:Kepercayaan) RETURN r.nama",
            f"Kepercayaan kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["suku", "suku kaum"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:MEMPUNYAI_SUKU]->(s:SukuKaum) RETURN s.nama",
            f"Suku kaum bagi kaum {ethnic} ialah"
        )

    elif any(word in query for word in ["berasal", "berasal dari", "negeri"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:BERASAL_DARI]->(n:Negeri) RETURN n.nama",
            f"Kaum {ethnic} berasal dari negeri"
        )

    elif any(word in query for word in ["tergolong", "kumpulan", "malaysia"]):
        return (
            f"MATCH (k:Kaum {{nama: '{ethnic}'}})-[:TERGOLONG_DALAM]->(kg:KumpulanKaum) RETURN kg.nama",
            f"Kaum {ethnic} tergolong dalam kumpulan"
        )

    return None, None

def query_neo4j(cypher_query):
    with neo4j_driver.session() as session:
        result = session.run(cypher_query)
        return [record[0] for record in result]

# === MAIN ===
synonyms = {
    "kaum": "etnik",
    "etnik": "kaum",
    "menikmati": "makan",
    "makan": "menikmati"
}

if __name__ == "__main__":
    pdf_file = "Training Document.pdf"
    model_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_batch=32, use_mlock=True)
    
    print("[1] Loading embedding model...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    update_pdf_file = "user_updates.pdf"
    if os.path.exists(update_pdf_file):
        print("[EXTRA] Detected update file: user_updates.pdf")
        update_text = extract_text_from_pdf(update_pdf_file)
        updated_faq_chunks = extract_faq_chunks(update_text)
        if os.path.exists("chunks.pkl"):
            with open("chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
        else:
            chunks = []
        chunks += updated_faq_chunks
        index, _ = build_faiss_index(chunks, embedding_model)
        faiss.write_index(index, "faiss.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
        os.remove(update_pdf_file)

    if os.path.exists("faiss.index") and os.path.exists("chunks.pkl"):
        print("[2] Loading FAISS index and chunks...")
        index = faiss.read_index("faiss.index")
        with open("chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
    else:
        print("[2] Extracting text from PDF...")
        full_text = extract_text_from_pdf(pdf_file)
        general_text, faq_text = extract_faq_section(full_text)
        general_chunks = chunk_general_text(general_text)
        faq_chunks = extract_faq_chunks(faq_text)
        chunks = general_chunks + faq_chunks
        index, _ = build_faiss_index(chunks, embedding_model)
        faiss.write_index(index, "faiss.index")
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)

    while True:
        print("\n❓ Ask a question (in Malay or English) [type 'keluar' or 'quit' to exit]:")
        query = input("> ").strip()
        if query.lower() in ["keluar", "quit"]:
            print("👋 Program tamat. Terima kasih!")
            break

        processed_query = preprocess_query(query, synonyms)
        # try:
        #     keywords = ["yang", "apakah", "di", "itu", "terdapat", "dalam",
        #                 "tergolong", "berasal", "berasal dari", "negeri",
        #                 "makanan", "makan", "menikmati", "dinikmati",
        #                 "pakaian", "memakai", "dipakai",
        #                 "alat muzik", "muzik", 
        #                 "bermain", "memainkan", "dimainkan", "dimain", "permainan", 
        #                 "sambutan", "menyambut", "perayaan", "disambut",
        #                 "tarian", "menari", 
        #                 "bahasa", "bercakap", "bertutur", "cakap", "ditutur",
        #                 "kepercayaan", "agama", "dipercayai", "percaya",
        #                 "suku", "suku kaum"]
        #     detected_lang = "ms" if re.search(r"[a-zA-Z]", processed_query) and any(kw in processed_query for kw in keywords) else "en"
        #     if detected_lang == "en":
        #         print("🔁 Translating question to Malay...")
        #         query_malay = translate_text(processed_query, target_lang="ms")
        #         print("Question in Malay:", query_malay)
        #     else:
        #         query_malay = processed_query
        # except:
        #     query_malay = processed_query

        try:
            detected_lang = detect(processed_query)
        except:
            detected_lang = "ms"

        if detected_lang == "en":
            print("🔁 Translating question to Malay...")
            query_malay = translate_text(processed_query, target_lang="ms")
            print("Question in Malay:", query_malay)
        else:
            query_malay = processed_query

        if is_kg_question(query_malay):
            print("🧠 Routing to Neo4j KG...")
            cypher, description = generate_cypher(query_malay)
            fallback_to_llm = False
            if cypher:
                try:
                    results = query_neo4j(cypher)
                    if results:
                        answer_malay = description + ": " + ", ".join(results)
                    else:
                        print("⚠️ No result from Neo4j, falling back to LLM...")
                        fallback_to_llm = True
                except Exception as e:
                    print(f"⚠️ Neo4j error: {e}, falling back to LLM...")
                    fallback_to_llm = True
            else:
                print("⚠️ No Cypher could be generated, falling back to LLM...")
                fallback_to_llm = True

            if fallback_to_llm:
                scores, indices, top_chunks = retrieve_chunks(query_malay, chunks, embedding_model, index)
                if scores[0][0] > 25.0:
                    answer_malay = "Saya tidak dapat menjawab soalan ini."
                else:
                    prompt = build_prompt(top_chunks, query_malay)
                    print("\n🤖 Generating answer from local LLM...")
                    answer_malay = ask_llm(prompt, llm)

        else:
            scores, indices, top_chunks = retrieve_chunks(query_malay, chunks, embedding_model, index)
            if scores[0][0] > 25.0:
                answer_malay = "Saya tidak dapat menjawab soalan ini."
            else:
                prompt = build_prompt(top_chunks, query_malay)
                print("\n🤖 Generating answer from local LLM...")
                answer_malay = ask_llm(prompt, llm)

        print("\n✅ Jawapan:")
        print(answer_malay)
        log_qa_to_csv(query, answer_malay)

        if detected_lang == "en":
            answer_english = translate_text(answer_malay, target_lang="en")
            print(answer_english)
