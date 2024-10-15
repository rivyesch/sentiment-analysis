import streamlit as st
import pandas as pd
import openai
# import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from collections import Counter

# Load open api key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI model
def initialize_llm():
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    batch_prompt_template = """
    You are tasked with analyzing a batch of public comments regarding the Trump-Harris debate. 
    You must analyze **all comments in the batch** together and extract the following insights across the entire batch:

    1. **Sentiment**: 
        - Determine the percentage of comments that are **pro-Trump**, **pro-Harris**, **neutral**, or **unrelated**.
        - Provide reasoning for why comments fit into each category.

    2. **Criticisms and Allegations**:
        - Identify any recurring criticisms or allegations directed **at either Trump or Harris**.
        - Ensure that the accusations are correctly attributed (e.g., Harris was accused of using an earpiece).

    3. **Moderation Bias**: 
        - Analyze whether the comments mention **bias in the debate moderation** or claim that the debate was rigged in favor of one candidate.
        - Summarize any recurring complaints about the host or the structure of the debate.

    4. **General Themes**: 
        - Highlight any dominant themes in the batch of comments, whether they focus on candidate performance, policy discussion, emotional appeals, or other key issues.
        - Include any significant patterns or insights regarding how viewers perceive the debate as a whole.
    
    Provide a percentage breakdown of sentiment and a summary.
    """
    prompt = PromptTemplate(input_variables=['text'], template=batch_prompt_template)
    return LLMChain(llm=llm, prompt=prompt)

# Function to batch comments for processing
def batch_comments(comments, batch_size=50):
    for i in range(0, len(comments), batch_size):
        yield comments[i:i + batch_size]

# Function to extract insights from a batch of comments
def extract_insights_from_batch(llm_chain, comment_batch):
    comments_str = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comment_batch)])
    response = llm_chain.run({"text": comments_str})
    return response.strip()

# Function to aggregate insights after batch processing
def aggregate_insights(individual_results):
    combined_comments = "\n\n".join(individual_results)
    global_prompt = """
    Based on the analysis of all the comments in this batch, summarize the overall sentiment, key themes, and repeated insights or opinions.
    Be sure to include the following in your summary:
        
    1. **Sentiment Trends**: Identify which candidate received more positive feedback. Estimate percentages for **pro-Trump**, **pro-Harris**, **neutral**, and **unrelated** comments.

    2. **Criticisms**: Highlight any major criticisms or allegations (e.g., bias claims, rigging, earpiece allegations, fact-checking).

    3. **Debate Moderation**: Summarize public opinion on the moderators and debate format, noting any claims of bias or unfair treatment.

    4. **Conclusion**: Conclude who the majority of commenters felt won the debate based on the sentiment analysis, and note any significant patterns or recurring issues mentioned by the commenters.
    """

    llm_chain = initialize_llm()
    global_insights = llm_chain.run({"text": combined_comments})
    return global_insights.strip()

# Main function to perform sentiment analysis on the dataset
def extract_sentiment_insights(df, batch_size=25):
    llm_chain = initialize_llm()
    all_batch_insights = []

    # Batch process comments for individual sentiment and insights extraction
    comment_batches = batch_comments(df[0], batch_size=batch_size)
    for batch in comment_batches:
        insights_from_batch = extract_insights_from_batch(llm_chain, batch)
        all_batch_insights.append(insights_from_batch)

    # Aggregate insights and perform global analysis
    overall_insights = aggregate_insights(all_batch_insights)
    return overall_insights

# Streamlit App
def main():
    st.set_page_config(page_title="Public Sentiment Analysis", layout="centered")

    # App title and description
    st.title("Public Sentiment & Commentary Analysis")
    st.markdown("""
    This application analyzes public comments from events like debates or interviews. 
    By uploading a CSV file containing a list of comments, the system performs sentiment analysis, 
    identifies key themes, and provides insights on candidate or event performance.
    
    **Instructions**:
    - Upload a CSV file containing public comments.
    - Receive a detailed analysis of the overall sentiment, recurring criticisms, moderation feedback, and key themes.
    """, unsafe_allow_html=True)

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload your comments dataset", type="csv")

    if uploaded_file is not None:
        # Load CSV into a DataFrame
        df = pd.read_csv(uploaded_file, header=None)

        # Perform sentiment analysis
        st.info("Performing sentiment analysis... Please wait.")
        insights = extract_sentiment_insights(df, batch_size=40)

        # Display the analysis results
        st.subheader("Analysis Results")
        st.write(insights)

if __name__ == "__main__":
    main()