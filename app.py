
import boto3
import json
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain


from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.chains import ConversationChain

from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

# turn verbose to true to see the full logs and documents
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain

import ipywidgets as ipw
from IPython.display import display, clear_output

from langchain.document_loaders import UnstructuredFileLoader

import trans


class ChatUX:
    """ A chat UX using IPWidgets
    """
    def __init__(self, qa, retrievalChain = False):
        self.qa = qa
        self.name = None
        self.b=None
        self.retrievalChain = retrievalChain
        self.out = ipw.Output()


    def start_chat(self):
        print("Starting chat bot")
        display(self.out)
        self.chat(None)


    def chat(self, _):
        if self.name is None:
            prompt = ""
        else: 
            prompt = self.name.value
        if 'q' == prompt or 'quit' == prompt or 'Q' == prompt:
            print("Thank you , that was a nice chat !!")
            return
        elif len(prompt) > 0:
            with self.out:
                thinking = ipw.Label(value="Thinking...")
                display(thinking)
                try:
                    if self.retrievalChain:
                        result = self.qa.run({'question': prompt })
                    else:
                        result = self.qa.run({'input': prompt }) #, 'history':chat_history})
                except:
                    result = "No answer"
                thinking.value=""
                print(f"AI:{result}")
                self.name.disabled = True
                self.b.disabled = True
                self.name = None

        if self.name is None:
            with self.out:
                self.name = ipw.Text(description="You:", placeholder='q to quit')
                self.b = ipw.Button(description="Send")
                self.b.on_click(self.chat)
                display(ipw.Box(children=(self.name, self.b)))


bedrock_runtime = boto3.client(
    service_name='bedrock-runtime', 
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)
br_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-g1-text-02", client=bedrock_runtime)


titan_llm = Bedrock(
    model_id="amazon.titan-text-express-v1",
    client=bedrock_runtime
)
titan_llm.model_kwargs = {"temperature": 0.7, "maxTokenCount": 200, "topP": 1
}

# print("\n\n")

s3_path = 's3://guida.fiscality.eng/PDF/Passive e-invoicing - jGalileo Wiki.pdf'

# Create an S3 client
s3_client = boto3.client(
    service_name='s3', 
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)


# Parse the S3 URL
s3_path_parts = s3_path.split('/')
bucket_name = s3_path_parts[2]
object_key = '/'.join(s3_path_parts[3:])

try:
    s3_client.download_file(Bucket=bucket_name, Key=object_key, Filename="/home/tommal/Scrivania/titan/prova.pdf")
    print("Access granted: Object exists and can be accessed.")
  
except boto3.exceptions.NoCredentialsError:
    print("AWS credentials are missing or incorrect.")
except botocore.exceptions.ClientError as e:
    if e.response['Error']['Code'] == "404":
        print("Object not found: The specified object does not exist.")
    elif e.response['Error']['Code'] == "403":
        print("Access denied: The AWS credentials do not have permission to access the object.")
    else:
        print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")




# loader = CSVLoader("prova.csv") # --- > 219 docs with 400 chars
# documents_aws = loader.load() #


loader = UnstructuredFileLoader('prova.pdf')
documents_aws = loader.load()
# print(f"documents:loaded:size={len(documents_aws)}")
# print("\n\n")

docs = CharacterTextSplitter(chunk_size=4000, chunk_overlap=400).split_documents(documents_aws)

# print(f"Documents:after split and chunking size={len(docs)}")
# print("\n\n")

vectorstore_faiss_aws = FAISS.from_documents(
    documents=docs,
    embedding = br_embeddings, 
    #**k_args
)


# wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss_aws)
# print(wrapper_store_faiss.query("Describe to me what is the first column?", llm=titan_llm))




def create_prompt_template():
    _template = """{chat_history}

Answer only with the new question.
How would you ask the question considering the previous conversation: {question}
Question:"""
    CONVO_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    return CONVO_QUESTION_PROMPT

memory_chain = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
chat_history=[]


qa = ConversationalRetrievalChain.from_llm(
    llm=titan_llm, 
    retriever=vectorstore_faiss_aws.as_retriever(), 
    # retriever=vectorstore_faiss_aws.as_retriever(search_type='similarity', search_kwargs={"k": 8}),
    memory=memory_chain,
    verbose=True,
    # condense_question_prompt=CONDENSE_QUESTION_PROMPT, # create_prompt_template(), 
    chain_type='stuff', # 'refine',
    #max_tokens_limit=100
)


while(True):

    user_question = input("Enter your question: ")

    user_question = trans.tranlate('it', 'en', str(user_question))

    # Create a dictionary containing the chat history and user's response
    conversation_data = {
        "chat_history": chat_history,
        "question": user_question
    }
    # Generate a bot response based on the user's input
    bot_response = qa(conversation_data)  # Remove num_tokens from the assignment

    # Append the bot's response to the chat history
    chat_history.append(bot_response)

    final_answer= trans.tranlate('en','it',str(bot_response['answer']))

    print(final_answer)


