# Chat With Documents

## Overview

The system is a versatile platform that seamlessly integrates chat functionality with document handling. This system supports a wide range of file formats, including PDF, DOCX, TXT, XLSX, and CSV. Users can conveniently upload multiple files and engage in a chat interface, making it easy to collaborate and communicate.

This model employs LangChain and OpenAI API keys for its functionality.

## Installation and Usage

1. Clone the repository to your local machine.

   ``` git clone https://github.com/aditya09092003/Chat-With-Documents.git ```

3. Move to the folder "Chat-With-Documents" and install the required dependencies by using the following command:

   ``` pip install -r requirements.txt ```

4. Get an API key from OpenAI and add it to the .env file in the directory (enter the API key in the "" marks).

5. After installing all the dependecies and entering the API key in .env file, run the code by writing the below code line in the terminal.

   ``` streamlit run app.py ```

6. The site (http://localhost:8501) will get loaded and then you can upload the document and click on "Process".

7. You can upload files of format .pdf, .csv, .xlsx, .docx, and .txt.

8. Then after uploading you can ask any query related to the document.
