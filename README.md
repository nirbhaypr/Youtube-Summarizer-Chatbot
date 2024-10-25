# YouTube Video Summarizer and Chatbot

This project is a Streamlit app that allows users to input a YouTube video URL, retrieves the video transcript, summarizes it, and enables users to ask questions about the video content. The app leverages a Large Language Model (LLM) for summary and question answering, using Pinecone for storing and retrieving relevant transcript chunks for question answering.

## Preview
![image](https://github.com/user-attachments/assets/bd6fc627-fa12-40d3-90f4-66988aa9f5da)
____________________________________________________________________________________________
![image](https://github.com/user-attachments/assets/f24753d7-0696-49ae-b799-d341d1475444)




## Features
- **Transcript Extraction**: Extracts the transcript from a YouTube video.
- **Video Summarization**: Summarizes the video transcript.
- **Interactive Q&A**: Users can ask questions about the video, and the app responds using relevant transcript content.
- **Efficient Search**: Uses Pinecone to store transcript chunks and retrieve relevant information efficiently.

## Technology Stack
- **Python**: Backend logic.
- **Streamlit**: Frontend for user interaction.
- **Langchain**: Manages LLM-based processing.
- **Pinecone**: Vector storage and retrieval.
- **LLM (Llama 3.1-70b)**: Generates summaries and answers questions.

## Installation

1. Clone the repository.
   ```bash
   git clone https://github.com/nirbhaypr/Youtube-Summarizer-Chatbot.git
   cd Youtube-Summarizer-Chatbot
   ```
2. Install required packages.
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
    - Create a .env file in the root folder.
    - Add your Pinecone API key.

## Usage
1. Run the Streamlit app.
   ```bash
    streamlit run main.py
   ```
2. Enter a YouTube video URL in the input field.
3. After the transcript is fetched, the app will display a summary.
4. You can ask questions about the video, and the app will provide answers based on the transcript.

## License
This project is licensed under the MIT License.
