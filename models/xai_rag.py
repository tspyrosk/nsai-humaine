from langchain_text_splitters import MarkdownTextSplitter
import outlines
import re
import os
# from langchain.text_splitter import MarkdownTextSplitter
# from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from paths import *
from dataset.minio_utils import minio_upload

#outlines_model = outlines.models.transformers("microsoft/Phi-3.5-mini-instruct")
outlines_model = outlines.models.openai(
    "gpt-4o",
    api_key= os.getenv("OPENAI_API_KEY")
)

llm = outlines.generate.text(outlines_model)

# embeddings_model_path = "output/gte-base-en-v1.5"
#
# if os.path.exists(embeddings_model_path):
#     embeddings_model = SentenceTransformer(embeddings_model_path, trust_remote_code=True)
# else:
#     embeddings_model = SentenceTransformer("Alibaba-NLP/gte-base-en-v1.5", trust_remote_code=True)
#     embeddings_model.save_pretrained(embeddings_model_path)
#
# rag_data_path = "input/bcw_rag_converted/"
# vectors_path = "output/rag_vectors.npy"

def pre_process_text(text):
    pattern = r"\(.*?\)|\[.*?\]|<.*?>"
    no_paren = re.sub(pattern, "", text)
    no_punct = re.sub(r'[^a-zA-Z0-9\s]', '', no_paren)
    return no_punct


def get_chunks(rag_data_path):
    all_texts = ""
    all_files = [f"{rag_data_path}{f}" for f in os.listdir(rag_data_path)]
    for file_path in all_files:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            all_texts += content

    all_texts = pre_process_text(all_texts)
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_text(all_texts)
    return chunks


def calc_rag_embeddings(embeddings, text):
    embedded = embeddings.encode(text, normalize_embeddings=True)
    return embedded


def form_query(concepts, important_features, pred_class, dataset_description=None):
    merged_concepts = " and ".join(concepts) if concepts else "no specific concepts identified"
    merged_features = " and ".join(important_features) if important_features else "no specific features"

    # Use dataset description for domain-aware query generation
    if dataset_description and dataset_description.get('domain'):
        domain = dataset_description.get('domain', '')
        row_desc = dataset_description.get('row_description', 'sample')
        prediction_target = dataset_description.get('prediction_target', '')
        class_desc = dataset_description.get('class_descriptions', '')

        # Build context-aware query
        context_info = f"Domain: {domain}. " if domain else ""
        context_info += f"Each observation represents: {row_desc}. " if row_desc else ""
        context_info += f"Prediction goal: {prediction_target}. " if prediction_target else ""
        context_info += f"Class meanings: {class_desc}. " if class_desc else ""

        query = f"""
        Context: {context_info}

        Given an observation classified as "{pred_class}" with the following characteristics: {merged_concepts}.
        What does this classification mean and why might these characteristics be relevant?
        What role do factors such as {merged_features} play in this classification?
        Provide a brief, domain-appropriate explanation.
        """
    else:
        # Fallback to generic query (not medical-specific)
        query = f"""
        Given an observation classified as "{pred_class}" with the following characteristics: {merged_concepts}.
        What does this classification mean?
        What role do factors such as {merged_features} play in this classification?
        Provide a brief explanation.
        """

    print("rag_query: ", query)
    return query


def extract_rag_explanation(concepts, important_features, pred_class, dataset_description=None):
    # chunks = get_chunks(rag_data_path)
    # if os.path.exists(vectors_path):
    #     rag_vectors = np.load(vectors_path)
    # else:
    #     rag_vectors = calc_rag_embeddings(embeddings_model, chunks)
    #     np.save(vectors_path, rag_vectors)
    query = form_query(concepts, important_features, pred_class, dataset_description)
    # query_embed = calc_rag_embeddings(embeddings_model, query)
    #
    # similarities = np.dot(rag_vectors, query_embed.T)
    # top_3_idx = np.argsort(similarities, axis=0)[-3:][::-1].tolist()
    #
    # most_similar = [chunks[idx] for idx in top_3_idx]

    context = ""
    # for (i, p) in enumerate(most_similar):
    #     context += p + "\n\n"

    # Build domain-aware prompt
    if dataset_description and dataset_description.get('domain'):
        domain = dataset_description.get('domain', '')
        prompt = f"""
        You are an expert in {domain}.
        Use the following CONTEXT or other trusted sources relevant to this domain to answer the QUESTION at the end.
        If the context is not sufficient but you have valid knowledge about this domain, provide it.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a brief, clear explanation that would be helpful for understanding this classification result.

        CONTEXT:{context}
        QUESTION:{query}
        """
    else:
        # Fallback to generic prompt (not medical-specific)
        prompt = f"""
        Use the following CONTEXT or your general knowledge to answer the QUESTION at the end.
        If the context is not sufficient but you have a valid answer, provide it.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Provide a brief, clear explanation for this classification result.

        CONTEXT:{context}
        QUESTION:{query}
        """

    rag_explanation = llm(prompt)
    return rag_explanation
