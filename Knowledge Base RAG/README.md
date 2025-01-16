# Amazon Bedrock Knowledge Base - Samples for building RAG workflows

## Contents to follow while using
- [0_create_ingest_documents_test_kb.ipynb](https://github.com/farzanaanjum/GEN-AI/blob/main/Knowledge%20Base%20RAG/Knowledge_Base_RAG/0_create_ingest_documents_test_kb.ipynb) - creates necessary role and policies required using the `utility.py` file. It uses the roles and policies to create Open Search Serverless vector index, knowledge base, data source, and then ingests the documents to the vector store. Once the documents are ingested it will then test the knowledge base using `RetrieveAndGenerate` API for question answering, and `Retrieve` API for fetching relevant documents. Finally, it deletes all the resources. If you want to continue with other notebooks, you can choose not to delete the resources and move to other notebooks. Please note, that if you do not delete the resources, you may be incurred cost of storing data in OpenSearch Serverless, even if you are not using it. Therefore, once you are done with trying out the sample code, make sure to delete all the resources. 

- [1_managed-rag-kb-retrieve-generate-api.ipynb](https://github.com/farzanaanjum/GEN-AI/blob/main/Knowledge%20Base%20RAG/Knowledge_Base_RAG/1_managed-rag-kb-retrieve-generate-api.ipynb) - Code sample for managed retrieval augmented generation (RAG) using `RetrieveAndGenerate` API from Knowledge Bases for Amazon Bedrock.

- [2_customized-rag-retrieve-api-claude-v2.ipynb](https://github.com/farzanaanjum/GEN-AI/blob/main/Knowledge%20Base%20RAG/Knowledge_Base_RAG/2_Langchain-rag-retrieve-api-mistral-and-claude-3-haiku.ipynb) - If you want to customize your RAG workflow, you can use the `retrieve` API provided by Knowledge Bases for Amazon Bedrock. Use this code sample as a starting point.

- [3_customized-rag-retrieve-api-langchain-claude-v2.ipynb](https://github.com/farzanaanjum/GEN-AI/blob/main/Knowledge%20Base%20RAG/Knowledge_Base_RAG/3_Langchain-rag-retrieve-api-claude-3.ipynb) - Code sample for using the `RetrieveQA` chain from LangChain and Amazon Knowledge Base as the retriever.

- Remember to use the [4_CLEAN_UP.ipynb](https://github.com/farzanaanjum/GEN-AI/blob/main/Knowledge%20Base%20RAG/Knowledge_Base_RAG/4_CLEAN_UP.ipynb)

Before following above code need to have AWS.
Steps:
- Create Amazon Bedrock Knowledge Base execution role with necessary policies for accessing data from S3 and writing embeddings into OSS.
- Create an empty OpenSearch serverless index.
- Download documents.
- Create Amazon Bedrock knowledge base.
- Create a data source within knowledge base which will connect to Amazon S3.
- Start an ingestion job using KB APIs which will read data from s3, chunk it, convert chunks into embeddings using Amazon Titan Embeddings model and then store these embeddings in AOSS. All of this without having to build, deploy and manage the data pipeline.

Add below Pre-requisites to policies:
- IAMFullAccess
- AWSLambda_FullAccess
- AmazonS3FullAccess
- AmazonBedrockFullAccess
- Custom policy for Amazon OpenSearch Serverless



