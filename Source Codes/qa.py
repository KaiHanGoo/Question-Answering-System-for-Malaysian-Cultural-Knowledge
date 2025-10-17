import os
import re
import pickle
import fitz
import faiss
import gradio as gr
import numpy as np
from llama_cpp import Llama
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import T5ForConditionalGeneration, AutoTokenizer
from neo4j import GraphDatabase
from langdetect import detect
import time
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# === Global Variables ===
csv_path = "qa_log.csv"
update_pdf_file = "user_updates.pdf"
pdf_file = "Training Document.pdf"
model_path = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
chunk_pickle_path = "chunks.pkl"
faiss_index_path = "faiss.index"

synonyms = {
    "kaum": "etnik",
    "etnik": "kaum",
    "menikmati": "makan",
    "makan": "menikmati"
}

# === Translation ===
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

def translate_text(text, target_lang='ms'):
    input_ids = tokenizer_t5.encode(f'terjemah ke {target_lang}: {text}', return_tensors='pt')
    outputs = model_t5.generate(input_ids, max_length=512)
    outputs = [i for i in outputs[0] if i not in [0, 1, 2]]
    return tokenizer_t5.decode(outputs, spaces_between_special_tokens=False)

def preprocess_query(query, synonyms):
    for word, replacement in synonyms.items():
        query = query.replace(word, replacement)
    return query

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

# === FAISS ===
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

# === Neo4j ===
neo4j_driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def is_kg_question(query):
    kg_keywords = ["kaum", "Malaysia", "tergolong", "berasal", "berasal dari", "negeri",
                   "makanan", "makan", "menikmati", "dinikmati",
                   "pakaian", "memakai", "dipakai",
                   "alat muzik", "muzik", "bermain", "memainkan", "dimainkan",
                   "permainan", "sambutan", "menyambut", "perayaan",
                   "tarian", "menari", "bahasa", "bercakap", "bertutur",
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

# === System Setup ===
print("Loading embedding model...")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_batch=32, use_mlock=True)

if os.path.exists(chunk_pickle_path) and os.path.exists(faiss_index_path):
    print("Loading FAISS index and chunks...")
    with open(chunk_pickle_path, "rb") as f:
        chunks = pickle.load(f)
    index = faiss.read_index(faiss_index_path)
else:
    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_file)
    general_text, faq_text = extract_faq_section(full_text)
    general_chunks = chunk_general_text(general_text)
    faq_chunks = extract_faq_chunks(faq_text)
    chunks = general_chunks + faq_chunks
    index, _ = build_faiss_index(chunks, embedding_model)
    with open(chunk_pickle_path, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, faiss_index_path)

# === Gradio Logic ===
def chat_interface(query, history):

    if not query or query.strip() == "":
        history.append(("Please enter a question before submitting.", ""))
        return history, history

    query = preprocess_query(query, synonyms)
    
    try:
        lang = detect(query)
    except:
        lang = "ms"
    query_malay = translate_text(query, "ms") if lang == "en" else query

    if is_kg_question(query_malay):
        print("Routing to Neo4j KG...")
        cypher, description = generate_cypher(query_malay)
        fallback_to_llm = False
        if cypher:
            try:
                results = query_neo4j(cypher)
                if results:
                    answer = description + ": " + ", ".join(results)
                else:
                    print("No result from Neo4j, falling back to LLM...")
                    fallback_to_llm = True
            except Exception as e:
                    print(f"Neo4j error: {e}, falling back to LLM...")
                    fallback_to_llm = True
        else:
            print("No Cypher could be generated, falling back to LLM...")
            fallback_to_llm = True
        
        if fallback_to_llm:
            scores, indices, top_chunks = retrieve_chunks(query_malay, chunks, embedding_model, index)
            if scores[0][0] > 25.0:
                answer = "Saya tidak dapat menjawab soalan ini."
            else:
                prompt = build_prompt(top_chunks, query_malay)
                print("\n🤖 Generating answer from local LLM...")
                answer = ask_llm(prompt, llm)

    else:
        scores, indices, top_chunks = retrieve_chunks(query_malay, chunks, embedding_model, index)
        if scores[0][0] > 25.0:
            answer = "Saya tidak dapat menjawab soalan ini."
        else:
            prompt = build_prompt(top_chunks, query_malay)
            print("\nGenerating answer from local LLM...")
            answer = ask_llm(prompt, llm)
    
    if lang == "en":
        answer = translate_text(answer, "en")

    # log to CSV
    log_qa_to_csv(query, answer)
    history.append((query, answer))
    
    log_df = gr.Dataframe(
        value=load_log(), 
        interactive=True, 
        headers=["Timestamp", "Question", "Answer", "Status"]
    )

    return history, history, log_df

def log_qa_to_csv(question, answer):
    new_row = {
        "Timestamp": datetime.now().isoformat(),
        "Question": question,
        "Answer": answer,
        "Status": "unchecked"
    }
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])
    df.to_csv(csv_path, index=False)

def load_log():
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["Timestamp", "Question", "Answer", "Status"])

def update_index(log_data):
    # Convert to DataFrame if not already
    df = pd.DataFrame(log_data, columns=["Timestamp", "Question", "Answer", "Status"])
    
    # Filter rows where status is "checked"
    selected = df[df["Status"] == "checked"]

    if selected.empty:
        return "No entries with Status = 'checked'."

    # Format the new Q&A entries
    new_chunks = [f"Question: {row['Question']}\nAnswer: {row['Answer']}" for _, row in selected.iterrows()]

    # Update global chunks and FAISS index
    global chunks, index
    chunks += new_chunks
    index, _ = build_faiss_index(chunks, embedding_model)

    # Save updated data
    with open(chunk_pickle_path, "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, faiss_index_path)

    os.remove("qa_log.csv")

    columns = ["Timestamp", "Question", "Answer", "Status"]
    empty_df = pd.DataFrame(columns=columns)
    empty_df.to_csv("qa_log.csv", index=False)

    new_log_df = gr.Dataframe(
        value=load_log(), 
        interactive=True, 
        headers=["Timestamp", "Question", "Answer", "Status"]
    )

    return new_log_df, "Update Successfully!"

# === Gradio UI ===
theme = gr.themes.Citrus(
    primary_hue="zinc",
).set(
    button_primary_background_fill='*primary_300'
)

with gr.Blocks(theme=theme) as demo:
    gr.Markdown("# 🤖 Soal Jawab Chatbot - Kekauman Malaysia")

    chatbot = gr.Chatbot()
    state = gr.State([])

    # user_input = gr.Textbox(label="Your Question", placeholder="Ask me anything... (Malay or English)")
    user_input = gr.Textbox(
        interactive=True,
        placeholder="Ask me anything... (Malay or English)",
        show_label=False,
        lines=1
    )
    # clear = gr.ClearButton([user_input, chatbot])

    with gr.Accordion("📋 QA Log Panel", open=False):
        log_df = gr.Dataframe(
            value=load_log(), 
            interactive=True, 
            headers=["Timestamp", "Question", "Answer", "Status"]
        )
        update_btn = gr.Button("🔄 Update")
        status_txt = gr.Textbox(visible=True, interactive=False, show_label=False)

    user_input.submit(chat_interface, [user_input, state], [chatbot, state, log_df])

    update_btn.click(fn=update_index, inputs=[log_df], outputs=[log_df, status_txt])

demo.launch()
