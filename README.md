## 🤖 GenAI Chat
GenAI Chat is an intelligent question-answering chatbot designed to help users interact with their data. Built on the Retrieval Augmented Generation (RAG) technique, it leverages the power of OpenAI’s large language models (LLMs) and Redis vector databases to provide accurate and context-aware answers to complex user queries.

<img src="data_indexing/images/genai_chat_ui.png" alt="GenAI Chat" width="500" style="border-radius: 10px;">

Key Features:
- Dynamic Knowledge Retrieval: It retrieve relevant documents from Redis Vector databases in real time, allowing them to respond with up-to-date and contextually accurate information.
- Natural language understanding: It utilizes Large language models (LLMs). LLMs can analyze complex patterns in language and can accurately interpret subtle meanings, respond appropriately to diverse queries, and adapt to various conversational styles, significantly enhancing the quality of human-computer communication.
- Multiple File Support: Project currently supports `.txt`, `.pdf`, `.docx`, `.json`, `.csv` and `.xlsx`.
- Multiple Data Source Support: Project supports 3 different sources for document indexing `Local`, `Azure` and `AWS`.

Technology Used:

[![My Skills](https://skillicons.dev/icons?i=python,flask,html,nginx,docker,redis,aws,azure,openai)](https://skillicons.dev)

For more detailed explanation of this project, including its design and implementation, check out the accompanying [Medium blog post](https://atinesh.medium.com/building-a-retrieval-augmented-generation-rag-chatbot-9a86c5b05691).


## 🚀 Updates
- **[11-05-2025]**: Added support for HNSW (Hierarchical Navigable Small World) Redis vector indexing and new improved user interface.
- **[11-03-2025]**: Added support for `.json`, `.csv` and `.xlsx` files.
- **[19-01-2025]**: Initial release of GenAI-Chat `v1.0`.

## 🔎 System Architecture

The chatbot consists of these core components:

- **Frontend**: Takes user queries and sends them to the backend. It's built with HTML + JavaScript and is running in a **Docker** container with **Nginx**.
- **Backend:** Takes user queries, fetches relevant documents from Redis Vector DB, builds prompts, and sends them to the LLM for generating response. Its built with **Flask** and is running in a **Docker** container.
- **Redis Vector Database:** Stores the document text, embedding vectors and session data. It’s also running in a **Docker** container.
- **OpenAI LLM**: Takes prompt and generates response. We will be using `gpt-4o` model for generating response and `text-embedding-3-large` model for generating embedding vectors (embedding dimension `3072`). These models are hosted in cloud Chatbot makes API call in order to communicate with the models. These models can be updated in `backend/config.py`.

<img src="data_indexing/images/genai_chat.png" alt="GenAI Chat" width="800" style="border-radius: 10px;">

## 🛠️ Installation 

Follow below steps in either on Mac and Linux (Ubuntu) machine.

**Step 1**: Install [Docker](https://www.docker.com/get-started/)

**Step 2**: Clone the repository
```
$ git clone https://github.com/atinesh/GenAI-Chat.git
```

**Step 3**: Configure OpenAI credentials

1. Login to [OpenAI](https://platform.openai.com)
2. In the upper right corner of the page, choose `Your profile`.
3. Create `API keys`.
4. Configure OpenAI key in `GenAI-Chat/.env` environment file as follows.

```
OPENAI_API_KEY=key
```

**Step 4**: Configure Vector index types (FLAT or HNSW) in `data_indexing/data_indexing.py` and `backend/config.py`.

- Choose the `FLAT` index type when you have small datasets (< 1M vectors) or when search accuracy is more important than search latency.
- Choose the `HNSW` (Hierarchical Navigable Small World) index type when you have larger datasets (> 1M documents) or when search performance and scalability are more important than search accuracy.

> Note: For `HNSW` index type optionally adjust `EF_RUNTIME` (Higher values increase accuracy, but also increase search latency.)

https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/

**Step 5**: Create a `data` directory in the project root if it doesn't already exist. This directory will be used to store the Redis dump file.

```
$ cd GenAI-Chat
$ mkdir data
```

**Step 6**: Build Images and Run Containers

```
$ cd GenAI-Chat
$ ./deploy.sh
```

**Step 7**: Index the data into Redis by following the instructions provided in the [README.md](/data_indexing/README.md) file.

**Step 8**: Once indexing is complete, you can interact with the frontend by visiting http://localhost:8080/.

> Note: RedisInsight can be accessed at http://localhost:8001/

## ⭐ Support and Contributions

If you found this repository helpful, please consider giving it a **star** ⭐ to show your support! It helps others discover the project and keeps me motivated to improve it further. If you'd like to support my work even more, consider buying me a coffee.

<a href='https://ko-fi.com/J3J4196KY7' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

### 🐛 Found a Bug?  
If you encounter any issues, please [open an issue](https://github.com/atinesh/GenAI-Chat/issues) with detailed steps to reproduce the problem. I’ll look into it as soon as possible.

### 💡 Have a Feature Request?  
I’m always looking to improve this project! If you have suggestions for new features or enhancements, feel free to [submit a feature request](https://github.com/atinesh/GenAI-Chat/issues).

---

Thank you for your support and contributions! 🙌

## 📝 License

This project is licensed under the `GNU General Public License v3.0`. See the [LICENSE](LICENSE) file for more details.
