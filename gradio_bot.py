# A quick GenAI chatbot with Gradio - an open-source Python package that allows you to quickly build a demo or web application for your machine learning model, API, or any arbitrary Python function.

# First, we import necessary tools:

import gradio as gr # Gradio helps create web interfaces
import boto3 # This helps connect to Amazon services
import json # This helps work with JSON data format

# Create the Bedrock client (Setting up connection to Amazon Bedrock:)
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

def get_response(message): # The main function that handles messages:
    request_body = json.dumps({  # Preparing the message: This function takes a message and returns a response from the AI model.
        "inputText": message,
        "textGenerationConfig": {
            "maxTokenCount": 256,
            "temperature": 0.7,
            "topP": 1,
            "stopSequences": []
        }
    })


# Sending request to the AI model:
    response = bedrock_runtime.invoke_model(
        modelId='amazon.titan-text-express-v1',
        body=request_body,
        contentType='application/json',
        accept='application/json'
    )


# Processing the response: This extracts the AI's response from the returned data.
    response_body = json.loads(response.get('body').read())
    return response_body.get('results')[0].get('outputText')


# Creating the web interface: Create the Gradio interface : This sets up a simple webpage where users can type messages.
# Gradio is a Python library that makes it easy to create web-based interfaces for your machine learning models, data processing pipelines, or any Python function. 

iface = gr.Interface( # This creates a web interface for your application
    fn=get_response,  # This tells the interface which function to run when users input something (in this case, a function called 'get_response')
    inputs="text",    # users will type in text as input
    outputs="text",   # the response will also be in text format
    title="Bedrock Chatbot", # The title that appears at the top of the interface
    description="Ask a question and get a response from Amazon Bedrock"
)

# Launch the interface : Starting the web server: This starts the webpage on port 7860 and makes it accessible to others.
iface.launch(server_name="0.0.0.0", server_port=7860, share=True)










