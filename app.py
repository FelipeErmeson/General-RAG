import gradio as gr
import spaces
import torch
import os
from huggingface_hub import snapshot_download
from utils import doc_converter, MSG_NENHUM_ARQUIVO_ENVIADO, MSG_TEXTO_NAO_EXTRAIDO
from rag_utils import create_split_doc, store_docs, create_rag_chain
import config

zero = torch.Tensor([0]).cuda()
print(zero.device)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

name_model = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
config.local_model_path = snapshot_download(
    repo_id=name_model,
    cache_dir="/root/.cache/huggingface",
    local_files_only=False
)
config.local_emb_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="/root/.cache/huggingface",
    local_files_only=False
)

def process_file(file):
    if file is None:
        return MSG_NENHUM_ARQUIVO_ENVIADO

    file_size = os.path.getsize(file)
    if file_size > MAX_FILE_SIZE:
        return f"O arquivo excede o limite. Por favor, realize o upload de um arquivo que contenha no máximo {MAX_FILE_SIZE/1024/1024:.1f}MB."

    texto_extraido = doc_converter(file)
    if texto_extraido is None:
        return MSG_TEXTO_NAO_EXTRAIDO
    
    return texto_extraido

@spaces.GPU
def ask_question(texto_extraido, question):
    
    # RAG
    docs_splitted = create_split_doc(texto_extraido)
    vector_store = store_docs(docs_splitted)
    rag_chain = create_rag_chain(vector_store)
    
    # resposta = rag_chain.run(question)
    response = rag_chain({"query": question})
    resposta = response["result"]
    docs_text = "\n\n\n===================================\n\n\n".join([doc.page_content for doc in response["source_documents"]])

    return resposta, docs_text

def update_ask_button(extracted_text, question):
    if extracted_text and MSG_NENHUM_ARQUIVO_ENVIADO not in extracted_text and MSG_TEXTO_NAO_EXTRAIDO not in extracted_text and question.strip():
        return gr.update(interactive=True)
    return gr.update(interactive=False)

def launch_app():
    with gr.Blocks(title="RAG", theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="slate")) as demo:
        gr.Markdown("# 🚀 Retrieval Augmented Generation - RAG")
        gr.Markdown("### ⚙️ Pergunte qualquer coisa para seu arquivo.")
        gr.Markdown(
            "🐶 Faça o upload do seu arquivo e pergunte qualquer coisa a ele! Este código é open source e disponível [aqui](https://github.com/FelipeErmeson/General-RAG) no GitHub. 😁"
        )

        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload da Imagem ou PDF", file_types=[".png", ".jpg", ".jpeg", ".pdf"])
                extracted_text = gr.Textbox(label="🌍 Texto extraído", lines=30)
            with gr.Column():
                question_input = gr.Textbox(label="📌 Faça uma pergunta ao seu documento!")
                ask_button = gr.Button("🔍 Perguntar", variant="primary", size="lg", interactive=False)
                answer_output = gr.Textbox(label="🎩 Resposta", lines=15)
                docs_sim = gr.Textbox(label="📎 Documentos similares a sua pergunta.", lines=15)

        # Conecta funções
        file_input.change(fn=process_file, inputs=file_input, outputs=extracted_text)
        # Sempre que o texto extraído ou a pergunta mudar, atualiza o botão
        extracted_text.change(fn=update_ask_button, inputs=[extracted_text, question_input], outputs=ask_button)
        question_input.change(fn=update_ask_button, inputs=[extracted_text, question_input], outputs=ask_button)

        # Chama o ask_question com o botão
        # question_input.submit(fn=ask_question, inputs=[extracted_text, question_input], outputs=[answer_output, docs_sim])
        ask_button.click(fn=ask_question, inputs=[extracted_text, question_input], outputs=[answer_output, docs_sim])

    demo.launch()

if __name__ == "__main__":
    launch_app()