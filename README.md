   # GEN-AI
![Add a heading](https://github.com/user-attachments/assets/56301b64-c635-4f54-a51d-1bc0696fafe6)


# GenAI Techniques
   ### 1. LLM - Large Language Model
   ### 2. Agents
   ### 3. RAG - Retrieval-Augmented Generation 
   ### 4. Langchain
   ### 5. MultiModel
   ### 6. Prompt Engineering

### 1. LLM - LARGE LANGUAGE MODEL
A LLM is a specialized type of AI that has been trained on vast amounts of text to understand existing content and generate original/new content.
A LLM is a type of AI program that can recognize and generate text, among other tasks. LLMs are trained on huge sets of data — hence the name "large." 
LLMs are built on machine learning: specifically, a type of neural network called a transformer model.

### 2. AGENTS
In the context of Generative AI (GenAI), agents refer to autonomous or semi-autonomous systems designed to perform specific tasks, often using machine learning models, including large language models (LLMs), to generate responses or actions. These agents are typically built to interact with users, process data, make decisions, and execute functions that can simulate intelligent behavior.
- Autonomy and Decision-making
Agents in GenAI are usually designed to operate independently to some extent. They can make decisions based on input data, a predefined set of rules, or real-time learning. For example, a GenAI agent might be programmed to assist users with specific queries, like a chatbot, or manage tasks like scheduling and email handling.
- Interaction with Environment
A core feature of these agents is their ability to interact with their environment. In the case of language models, this might involve conversing with users, retrieving data from external databases, or manipulating digital environments to achieve a task (e.g., creating documents, editing text, or performing computations).
- Task-Oriented Agents
Some agents are focused on specific tasks, such as:
     - Chatbots or virtual assistants (e.g., Siri, Alexa) that can help users by answering questions, providing recommendations, or executing commands.
     - Content Generators that create articles, summaries, or even entire books.
     - Personalized Agents that learn from users’ behavior to provide tailored suggestions, like Netflix recommendations or personalized email responses.
- Multi-Agent Systems
In more advanced implementations, multiple agents can work together in a coordinated manner to complete more complex tasks. This involves collaboration or competition between agents, which is often used in simulated environments for research, gaming, or complex problem-solving.
- Learning Capabilities
Some agents in GenAI are designed to learn over time, refining their strategies and improving task performance through experience, such as reinforcement learning-based agents. These can be used for dynamic tasks like game playing or financial forecasting.
- Human-AI Collaboration
While agents can work independently, they are also frequently designed to collaborate with humans. This could be in the form of providing suggestions, augmenting human decision-making, or enhancing creativity in fields like writing, design, or software development.

### 2. RAG - RETRIEVAL-AUGMENTED GENERATION
RAG is the process of optimizing the output of a large language model, so it references an authoritative knowledge base outside of its training data sources before generating a response. Large Language Models (LLMs) are trained on vast volumes of data and use billions of parameters to generate original output for tasks like answering questions, translating languages, and completing sentences. RAG extends the already powerful capabilities of LLMs to specific domains or an organization's internal knowledge base, all without the need to retrain the model. It is a cost-effective approach to improving LLM output so it remains relevant, accurate, and useful in various contexts.
##### RAG addresses some key challenges with large language models, including:
- Knowledge cutoff: LLMs have limited knowledge based on what they were trained on. RAG provides access to external knowledge, enabling LLMs to generate more accurate and reliable responses.
- Hallucination risks: LLMs may generate responses that are not factually accurate or relevant to the query. RAG allows LLMs to draw upon external knowledge sources to supplement their internal representation of information, reducing the risk of hallucinations.
- Contextual limitations: LLMs lack context from private data, leading to hallucinations when asked domain or company-specific questions. RAG provides up-to-date information about the world and domain-specific data to your GenAI applications, enabling them to generate more informed answers.
- Auditability: RAG allows GenAI to cite its sources and improves auditability, making it easier to track the sources of information used to generate responses.

### 4. Langchain
LangChain is an open-source framework for building applications based on LLMs.
LLMs are large deep-learning models pre-trained on large amounts of data that can generate responses to user queries—for example, answering questions or creating images from text-based prompts. 
LangChain provides tools and abstractions to improve the customization, accuracy, and relevancy of the information the models generate. 
For example, developers can use LangChain components to build new prompt chains or customize existing templates. LangChain also includes components that allow LLMs to access new data sets without retraining.
Langchain uses the **Prompt-Template** to write the multiple prompts for any problems, to get perfect solutions.
Multiple prompts:
   1. Zero-Short Prompt
   2. One-Short Prompts
   3. Few-Short Prompts  

### 5. MultiModel
In the context of Generative AI (GenAI), **multimodal** refers to the capability of AI models to process and generate information across multiple types or modalities of data. This means the model can understand and work with different kinds of input such as text, images, audio, video, and even more complex combinations of these.
For example, a **multimodal AI model** could:
1. **Understand Text and Images**: A multimodal model might be able to interpret a text description and generate a corresponding image based on that description.
2. **Combine Text and Audio**: The AI could process spoken language (audio) and generate text responses or even understand written text and produce audio output.
3. **Contextual Awareness Across Modalities**: The AI can use information from multiple types of data together, allowing for more complex and contextual reasoning. For instance, it could read a piece of text and understand the emotional tone from an accompanying image.
### Examples of Multimodal AI Models:
1. **CLIP (Contrastive Language-Image Pretraining)**: A model developed by OpenAI that can link text with images. It can interpret a written description and search for the most relevant images that match it.
2. **DALL·E**: Another OpenAI model, which generates images from textual descriptions (text-to-image).
3. **GPT-4 with Vision**: An advanced version of the GPT model that is multimodal, capable of understanding both text and images, and responding accordingly to queries about pictures.


