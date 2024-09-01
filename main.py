import fitz  
import csv
import os
import logging
import time
from langchain_community.llms import Ollama


llm = Ollama(model="mannix/phi3-mini-4k")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_first_page_text_from_pdf(pdf_file):
 
    try:
        logging.info("Starting text extraction from PDF.")
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        page = doc.load_page(0)  
        text = page.get_text()
        logging.info("Completed text extraction from PDF.")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

def few_shot_prompting(text):
  
    logging.info("Creating prompt for LLM.")
    prompt = f"""
    Context: {text}

    Based on the context above, please answer the following questions:

    Q1: What is the generic name?
    A1: 

    Q2: What is the brand name?
    A2: 

    Q3: What is the drug strength?
    A3: 

    Q4: What are the target species?
    A4: 

    Q5: What is the pharmaceutical form?
    A5: 
    """
    logging.info("Prompt creation completed.")
    return prompt

def extract_field_from_response(response, field_label):

    lines = response.splitlines()
    for line in lines:
        if line.startswith(field_label):
            return line.split(':', 1)[1].strip()
    return ""

def process_pdfs_and_save_to_csv(directory, output_csv):
 
    results = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            logging.info(f"Processing file: {filename}")

            start_time = time.time()

            with open(file_path, 'rb') as pdf_file:
                try:
                   
                    text_extraction_start = time.time()
                    text = extract_first_page_text_from_pdf(pdf_file)
                    text_extraction_end = time.time()
                    logging.info(f"Text extraction took {text_extraction_end - text_extraction_start:.2f} seconds.")

                    if text:
                        prompt = few_shot_prompting(text)

                  
                        llm_start = time.time()
                        try:
                            response = llm.invoke(prompt)
                        except Exception as e:
                            logging.error(f"Error getting response from LLM: {e}")
                        llm_end = time.time()
                        logging.info(f"LLM invocation took {llm_end - llm_start:.2f} seconds.")

        
                        results.append({
                            'file_name': filename,  
                            'generic_name': extract_field_from_response(response, 'A1'),
                            'brand_name': extract_field_from_response(response, 'A2'),
                            'drug_strength': extract_field_from_response(response, 'A3'),
                            'target_species': extract_field_from_response(response, 'A4'),
                            'pharmaceutical_form': extract_field_from_response(response, 'A5'),
                        })

                except Exception as e:
                    logging.error(f"Error processing file {filename}: {e}")

            end_time = time.time()
            logging.info(f"Finished processing {filename} in {end_time - start_time:.2f} seconds.")

  
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'generic_name', 'brand_name', 'drug_strength', 'target_species', 'pharmaceutical_form']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    logging.info('Information extracted and saved to CSV successfully.')

def main():
   
    pdf_directory = r'E:\canada\iter1_Team_1' 
    output_csv = 'extracted.csv'
    

    process_pdfs_and_save_to_csv(pdf_directory, output_csv)

if __name__ == "__main__":
    main()
