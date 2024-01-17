#%%
import streamlit as st

# To run the page: streamlit run About.py --server.port 8888

#----------------------------------------------
# Main function

def main():
    st.title("ArXiv GPT")
    st.divider()
    
    st.sidebar.header("About")
    st.sidebar.write(
        "This is an Arxiv Q&A application powered by a large language model. Here's a brief overview of the process:"
    )

    st.sidebar.subheader("1. Topic Selection")
    st.sidebar.write(
        "Users input a topic of interest related to Arxiv, which serves as the basis for generating questions."
    )

    st.sidebar.subheader("2. Question Prompt")
    st.sidebar.write(
        "Users provide a specific question prompt related to the chosen topic. This prompt guides the model's response."
    )

    st.sidebar.subheader("3. Language Model Processing")
    st.sidebar.write(
        "The application uses a language model to process the user's question and generate a response. The model is designed to leverage relevant information from Arxiv documents."
    )

    st.sidebar.subheader("4. Response and Sources")
    st.sidebar.write(
        "The generated response, along with the titles and URLs of relevant source documents from Arxiv, are displayed to the user. This allows users to explore the source materials."
    )

    st.sidebar.subheader("About the Application")
    st.sidebar.write(
        "This application is designed to facilitate Q&A on Arxiv topics using advanced language models. It aims to provide users with informative responses and direct access to source documents for further exploration."
    )
    
if __name__ == "__main__":
    main()

#%%