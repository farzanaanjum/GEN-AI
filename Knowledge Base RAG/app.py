import streamlit as st
import boto3
from botocore.client import Config

# Initialize the Streamlit app
st.title("Knowledge Base and RAG") 

# Initialize AWS clients
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime')
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config)
boto3_session = boto3.session.Session()
region_name = boto3_session.region_name

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"  # Modify as needed
region_id = region_name

def retrieveAndGenerate(input, kbId, sessionId=None, model_id=model_id, region_id=region_id):
    model_arn = f'arn:aws:bedrock:{region_id}::foundation-model/{model_id}'
    params = {
        'input': {'text': input},
        'retrieveAndGenerateConfiguration': {
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kbId,
                'modelArn': model_arn
            }
        }
    }
    if sessionId:
        params['sessionId'] = sessionId

    return bedrock_agent_client.retrieve_and_generate(**params)

# Streamlit input fields
query = st.text_input("Prompt:", "")
kb_id = "GMEBCJCKRT" # st.text_input("Enter Knowledge Base ID:", "")  # Provide a way to input Knowledge Base ID

# Button to trigger the action
if st.button("Generate Response"):
    if kb_id:
        try:
            response = retrieveAndGenerate(query, kb_id, model_id=model_id, region_id=region_id)
            generated_text = response['output']['text']
            st.write("Generated Text:")
            st.write(generated_text)
            
            citations = response.get("citations", [])
            contexts = []
            for citation in citations:
                retrievedReferences = citation.get("retrievedReferences", [])
                for reference in retrievedReferences:
                    contexts.append(reference["content"]["text"])
            
            st.write("Contexts:")
            for context in contexts:
                st.write(context)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a Knowledge Base ID.")
