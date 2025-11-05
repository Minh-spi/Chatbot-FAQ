import streamlit as st
import chromadb
import numpy as np
import math
import os
import csv

# CƠ SỞ TRI THỨC Q&A (ĐỌC TỪ FILE)
knowledge_base = []
with open('knowledge_base.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        knowledge_base.append((row[0], row[1]))


# HÀM XỬ LÝ TF-IDF
def tokenize(text):
    return text.lower().split()

def compute_tf(doc):
    tf = {}
    words = tokenize(doc)
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    for w in tf:
        tf[w] /= len(words)
    return tf

def compute_idf(docs):
    idf = {}
    total_docs = len(docs)
    for doc in docs:
        for w in set(tokenize(doc)):
            idf[w] = idf.get(w, 0) + 1
    for w in idf:
        idf[w] = math.log(total_docs / (1 + idf[w]))
    return idf

def compute_tfidf(doc, idf):
    tf = compute_tf(doc)
    return {w: tf[w] * idf.get(w, 0) for w in tf}

def cosine_similarity(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)
    norm1 = math.sqrt(sum(v*v for v in vec1.values()))
    norm2 = math.sqrt(sum(v*v for v in vec2.values()))
    return dot / (norm1 * norm2 + 1e-9)

# KHỞI TẠO VECTOR DB (Chroma)
# Thư mục lưu vector database
persist_dir = os.path.join(os.getcwd(), "chroma_persist")
# Khởi tạo client có lưu trữ
client = chromadb.PersistentClient(path=persist_dir)
# Xóa collection cũ nếu có, rồi tạo mới
# Lấy hoặc tạo mới collection
try:
    collection = client.get_collection("faq")
except:
    collection = client.create_collection("faq")

# Tạo vector TF-IDF cho các câu hỏi
questions = [q for q, _ in knowledge_base]
idf = compute_idf(questions)
vectors = [compute_tfidf(q, idf) for q in questions]

# Lưu vào ChromaDB
# Lưu vào ChromaDB (xóa dữ liệu cũ nếu có)
try:
    client.delete_collection("faq")
except:
    pass
collection = client.create_collection("faq")

for i, q in enumerate(questions):
    collection.add(
        ids=[str(i)],
        documents=[q],
        metadatas=[{"answer": knowledge_base[i][1]}]
    )


# GIAO DIỆN CHATBOT
st.title("🤖 ElectroStore Chatbot")
st.write("Chào mừng bạn đến với ElectroStore! Hỏi gì cũng được nè 😄")

# Khởi tạo lịch sử chat nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị các tin nhắn đã có trong lịch sử
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Nhận input từ người dùng
if prompt := st.chat_input("Nhập câu hỏi của bạn:"):
    # 1. Thêm tin nhắn của người dùng vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Xử lý và lấy câu trả lời từ bot
    user_vec = compute_tfidf(prompt, idf)
    sims = [cosine_similarity(user_vec, v) for v in vectors]
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]

    if best_sim < 0.2:
        response = "Xin lỗi, tôi không tìm thấy câu trả lời cho câu hỏi này."
    else:
        response = knowledge_base[best_idx][1]

    # 3. Thêm câu trả lời của bot vào lịch sử và hiển thị
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)