import streamlit as st
import boto3
import json
import botocore
 
# Initialize Boto3 client for Bedrock
boto3_bedrock = boto3.client("bedrock-runtime")

def generate_email(prompt_data):
    body = json.dumps(
        {
            "inputText": prompt_data,
            "textGenerationConfig": {"topP": 0.95, "temperature": 0.1},
        }
    )
 
    modelId = "amazon.titan-tg1-large"
    accept = "application/json"
    contentType = "application/json"
    outputText = "\n"
 
    try:
        response = boto3_bedrock.invoke_model(
            body=body, modelId=modelId, accept=accept, contentType=contentType
        )
        response_body = json.loads(response.get("body").read())
        outputText = response_body.get("results")[0].get("outputText")
        return outputText
    except botocore.exceptions.ClientError as error:
        if error.response["Error"]["Code"] == "AccessDeniedException":
            st.error(
                f"{error.response['Error']['Message']}\n"
                "To troubleshoot this issue please refer to the following resources:\n"
                "https://docs.aws.amazon.com/IAM/latest/UserGuide/troubleshoot_access-denied.html\n"
                "https://docs.aws.amazon.com/bedrock/latest/userguide/security-iam.html"
            )
        else:
            st.error(f"An error occurred: {error}")
        return None
 
# Streamlit UI
st.title("Email Generator")
 
customer_name = st.text_input("Customer Name", "") #John Doe"
feedback_details = st.text_area("Feedback Details", "Please provide details of the negative feedback")
 
if st.button("Generate Email"):
    if customer_name and feedback_details:
        prompt_data = f"""
        Command: Write an email from Bob, Customer Service Manager, to the customer "{customer_name}" 
        who provided negative feedback on the service provided by our customer support engineer.
        Feedback Details: {feedback_details}
        """
        email_text = generate_email(prompt_data)
        if email_text:
            st.subheader("Generated Email")
            st.write(email_text)
    else:
        st.error("Please provide both the customer name and feedback details.")