# General-RAG

<div align="center">
    O General-RAG é um aplicativo com a finalidade de ajudar a responder qualquer pergunta relacionada ao seu documento.
  É um aplicativo feito em Gradio e segue uma estratégia técnica de Retrieval Augmented Generation - RAG para melhorar a precisão das perguntas e amenizar as alucinações.
  <br/>
  <br/>
  O app pode ser acessado
  <a href="https://huggingface.co/spaces/FelipeErmeson/projeto-rag" target="_blank">aqui</a>.
  </p>
</div>

### Visão Geral
<img src="assets/readme/general-rag-guide.gif">

### Detalhes técnicos:

* **LangChain** para orquestração de código para LLM e prompts para RAG.
* **FAISS** banco de dados vetorial para pesquisa e recuperação otimizada de documentos.
* **Docling** para reconhecimento de texto (OCR).
* **Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 (quantizado)** LLM (Large Language Model) para geração da resposta.
* **sentence-transformers/all-MiniLM-L6-v2** modelo para geração de embeddings para recuperação de documentos no banco de similaridade.
* **Gradio** para criação de componentes de interface e interação com o usuário.

### Limitações:

* Por questões de custo, o app General-RAG roda em uma máquina ZeroGPU do HuggingFace e portanto oferece limitações de vRAM, espaço em disco, rate limiting, throttling e outros.
* O LLM Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8 é um modelo quantizado, e portanto pode oferecer limitações de precisão nas respostas.
* Banco de similaridade FAISS é indicado apenas para protótipo.
* Documentos grandes podem apresentar problemas.
* Este projeto foi feito com a finalidade para estudos e atualmente não possui nenhuma intenção de realizar mais avanços.
