from flask import Flask, request, jsonify
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain



app = Flask(__name__)

llm = Ollama(model="llama3")

prompt = ChatPromptTemplate.from_template(
    """ 
    Answer the following question only based on the provided context.
    Note : evaluate the context carefully.
    <context>
    {context}
    </context>

    Question : {input}""")


document_chain = create_stuff_documents_chain(llm, prompt)


@app.route('/')
def home():
    return "Hello, Flask!"

documents = []
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def databa(chunks):

    vdb = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text",show_progress=True),
        collection_name="local-rag"
    )

    return vdb



@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    print("reach")
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part"}), 200

    # printing numbers of files on terminal
    files = request.files.getlist('files')
    print(files)
    

    
    for file in files:
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

        
            loader = PyPDFLoader(filepath)
            

            documents.extend(loader.load())

            
        else:
            return jsonify({"error": f"Invalid file type: {file.filename}"}), 200
        
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)


    global vector_db
    
    vector_db = databa(chunks)
    
    return jsonify({"texts": "done successfully "}), 200
    


@app.route('/ask_question', methods=['POST'])
def ask_question():
    question = request.json['question']
    retriver =  vector_db.as_retriever()
    
    retrival_chain = create_retrieval_chain(retriver, document_chain)
    
    res = vector_db.similarity_search(question)

    # printing extracted chunks on terminal

    for i in res:
        stt = i.page_content
        st = stt.replace("\n", " ")
        print(st, "\n")
    

    ans = retrival_chain.invoke({"input" : str(question)})



    return jsonify({'answer': ans['answer']}), 200


if __name__ == '__main__':

    app.run(debug=False)
