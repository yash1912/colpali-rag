from io import BytesIO
import chainlit as cl
from colpali_rag import ColPaliModel, QwenVLChat
import os
import asyncio

# Global variables to store uploaded and indexed PDF file paths
uploaded_files = []
indexed_files = []

@cl.cache
async def load_models():
    colpali_model_instance = ColPaliModel()
    qwen_chat_instance = QwenVLChat()

    return colpali_model_instance, qwen_chat_instance

# Initialize models globally
colpali_model_instance, qwen_chat_instance = asyncio.run(load_models())

@cl.on_chat_start
async def start():
    # Check if there are any uploaded files already
    if not uploaded_files:
        await cl.Message(content="Welcome! Please upload a PDF to get started.").send()
    else:
        await cl.Message(content="Welcome back! You can continue querying the uploaded PDFs.").send()

@cl.on_message
async def query(message: cl.Message):
    global uploaded_files, indexed_files  # Reference the global variables

    try:
        # Check if there are new files in the message elements
        pdf_files = [file for file in message.elements if "pdf" in file.mime]

        # Add new files to the uploaded_files list
        for pdf_file in pdf_files:
            if pdf_file.path not in uploaded_files:
                uploaded_files.append(pdf_file.path)
                # await cl.Message(content=f"Uploaded file: {pdf_file.name}", elements=[pdf_file]).send()
        
        # If there are no new files and uploaded_files is still empty, prompt the user to upload
        if not pdf_files and not uploaded_files:
            await cl.Message(content="Please upload a PDF to get started.").send()
            return

        # Determine which files need to be indexed (those not already indexed)
        files_to_index = [file for file in uploaded_files if file not in indexed_files]

        if files_to_index:
            # Show a loading animation while indexing the PDFs
            await cl.Message(content="Indexing the new PDF(s)... This may take a moment.").send()

            # Index the new PDFs using the ColPali model asynchronously
            await asyncio.to_thread(colpali_model_instance.index_pdf, files_to_index)

            # Add newly indexed files to the indexed_files list
            indexed_files.extend(files_to_index)

        # Run the query on the indexed PDFs
        response, images = await asyncio.to_thread(qwen_chat_instance.chat, message.content, colpali_model_instance)
        # Convert PIL images to bytes and send the response
        image_elements = []
        for image in images:
            byte_io = BytesIO()
            image.save(byte_io, format='PNG')  # Convert the image to bytes in PNG format
            image_bytes = byte_io.getvalue()
            image_elements.append(cl.Image(content=image_bytes))
        # Send the response
        await cl.Message(content=response + "\n\nImages: ", elements=image_elements).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()
