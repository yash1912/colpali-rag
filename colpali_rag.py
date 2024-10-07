import torch
import gc
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from colpali_engine.models import ColPaliProcessor, ColPali
from qwen_vl_utils import process_vision_info
from PIL import Image
from typing import List, Tuple, cast
from pdf2image import convert_from_path, pdfinfo_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
import warnings
import chainlit as cl
warnings.filterwarnings("ignore")
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage


class ColPaliModel:
    def __init__(self, model_name: str = "vidore/colpali-v1.2"):
        """
        Initialize the ColPali model.

        Args:
            model_name (str): Name of the pre-trained model to use. Defaults to "vidore/colpali-v1.2".
        """
        self.model_name = model_name
        self.colpali_model, self.colpali_processor = self.initialize_model()
        self.embeddings = []  # Store embeddings
        self.images = []  # Store images
    
    @cl.cache
    def initialize_model(self):
        """
        Initialize the ColPali model.

        This function is cached by ChainLit, so it will only be called once.

        Returns:
            Tuple[ColPali, ColPaliProcessor]: A tuple containing the ColPali model and processor.
        """
        colpali_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        colpali_model = cast(ColPali, ColPali.from_pretrained(
            self.model_name, 
            device_map="cuda",
            torch_dtype=torch.bfloat16,
            quantization_config=colpali_quant_config
        ))
        colpali_model = colpali_model.eval()
        colpali_processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))
        return colpali_model, colpali_processor

    def index_pdf(self, file_paths: List[str], dpi: int = 100):
        """
        Index the given PDF files and store the embeddings and images.

        Args:
            file_paths (List[str]): List of file paths to the PDF files to be indexed.
            dpi (int, optional): The DPI to use when converting the PDF files to images. Defaults to 100.

        Raises:
            PDFInfoNotInstalledError: If the poppler library is not installed.
            PDFPageCountError: If the PDF file is not valid.
            PDFSyntaxError: If the PDF file contains invalid syntax.
        """
        self.embeddings, self.images = [], []  # Reset embeddings and images
        for file_path in file_paths:
            print(f"Working on {file_path}")
            try:
                info = pdfinfo_from_path(file_path)
                max_pages = info["Pages"]
                images = convert_from_path(file_path, first_page=1, last_page=max_pages, dpi=dpi)
                self.images.extend(images)
            except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
                print(f"Error processing PDF file: {e}")
                continue

        dataloader = torch.utils.data.DataLoader(self.images, batch_size=1, collate_fn=lambda x: self.colpali_processor.process_images(images=x), shuffle=False)
    
        for batch in dataloader:
            with torch.no_grad():
                batch = {key: value.to("cuda") for key, value in batch.items()}
                embedding = self.colpali_model(**batch)
                self.embeddings.extend(list(torch.unbind(embedding.to("cpu"))))
        # print("Pdf indexing done!")
        self.clear_memory()

    def retrieve(self, query: str, k: int = 1) -> Tuple[List[int], List[Image.Image], List[float]]:
        resultant_images, query_embeddings = [], []

        with torch.no_grad():
            query_batch = self.colpali_processor.process_queries([query])
            query_batch = {key: value.to("cuda") for key, value in query_batch.items()}
            query_embedding = self.colpali_model(**query_batch)
            query_embeddings.extend(list(torch.unbind(query_embedding.to("cpu"))))

        similarity_scores = self.colpali_processor.score(query_embeddings, self.embeddings).cpu().numpy()
        top_pages = similarity_scores.argsort(axis=1)[0][-k:][::-1].tolist()

        for page in top_pages:
            resultant_images.append(self.images[page])

        return top_pages, resultant_images, similarity_scores

    @staticmethod
    def clear_memory():
        torch.cuda.empty_cache()
        gc.collect()


class QwenVLChat():
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.model_name = model_name
        self.qwen_model, self.qwen_processor = self.initialize_model()
        self.memory = ConversationBufferMemory(return_messages=True)
    @cl.cache
    def initialize_model(self):
        qwen_bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            quantization_config=qwen_bnb_config
        )
        qwen_processor = AutoProcessor.from_pretrained(self.model_name)
        return qwen_model, qwen_processor

    @staticmethod
    def clear_memory():
        torch.cuda.empty_cache()
        gc.collect()

    def chat(self, query: str, retriever: ColPaliModel):
        if len(self.memory.chat_memory.messages) > 1:
            retrieval_message = [{
                                "role": "user",
                                "content": [{
                                            "type": "text", 
                                            "text": f"Based on the previous conversation: {self.memory.buffer_as_str}, create a clear and concise standalone question that directly addresses the main point of the user's current query:\nHuman: {query}"
                                            }]
                                }]

            text = self.qwen_processor.apply_chat_template(retrieval_message, tokenize=False, add_generation_prompt=True)

            inputs = self.qwen_processor(
                text=text,
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to("cuda")

            generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
            generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            modified_query = self.qwen_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            best_index, retrieved_images, similarity_scores = retriever.retrieve(modified_query[0], k=1)
            QwenVLChat.clear_memory()
        else:
            best_index, retrieved_images, similarity_scores = retriever.retrieve(query, k=1)


        
        messages = [{
                    "role": "user",
                    "content": [
                        {
                                        "type": "text",
                                        "text": f"\n\nRefer to the given previous conversation when and if necesary: {self.memory.buffer_as_str}" if len(self.memory.chat_memory.messages) > 1 else ""
                                    }] +\
                        [{"type": "image", "image": retrieved_image} for retrieved_image in retrieved_images
                                ] + [
                                    {
                                        "type": "text",
                                        "text": f"Based only on the information provided in the images, please find the specific numerical data or facts that answer the following question. Try your best to answer the question. Only if you cannot find the answer in the images, respond with 'I don't know.'\n\nQuestion: {query} \n Thinking: I should try my best to answer the given question. \nAnswer: "
                                    }
                                ]
                    }]


        print("""Messages: """, messages, "\n"+ "*" * 10 + "\n")

        self.memory.chat_memory.messages.append(HumanMessage(content=query))


        text = self.qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.qwen_processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = inputs.to("cuda")

        generated_ids = self.qwen_model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        self.memory.chat_memory.messages.append(AIMessage(content=output_text[0]))

        ColPaliModel.clear_memory()
        QwenVLChat.clear_memory()

        return output_text[0], retrieved_images

