import streamlit as st
from langchain_code import get_transcript, summarize_video, split_transcript, store_in_pinecone, answer_question

def main():
    st.title("YouTube Video Summarizer and Chatbot")

    # Step 1: Input YouTube URL
    video_url = st.text_input("Enter YouTube Video URL")

    if video_url:
        # Step 2: Extract and display transcript
        st.write("Fetching transcript...")
        transcript = get_transcript(video_url)
        st.write("Transcript fetched!")

        # Step 3: Summarize the transcript
        st.write("Summarizing the video...")
        summary = summarize_video(transcript)
        st.write(f"Summary: {summary}")

        # Step 4: Prepare chatbot after summarization
        chunks = split_transcript(transcript)
        store_in_pinecone(chunks)

        # Step 5: Ask questions about the video
        st.write("Ask questions about the video:")
        question = st.text_input("Ask a question")

        if question:
            answer = answer_question(chunks, question)
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()
