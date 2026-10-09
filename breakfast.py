import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from os import getenv

# 0. Set your OpenAI API key
OPENAI_API_KEY = getenv("OPENAI_API_KEY")

# 1. Load the secret API key from the .env file
load_dotenv()

# 2. Configure Streamlit Page Layout
st.set_page_config(page_title="GenAI Breakfast Assistant", page_icon="🍳")
st.title("🍳 AI Nutrition Assistant")
st.caption("Ask me your breakfast recipes!")

# 3. Initialize Conversation History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 4. Display Existing Chat Messages on Screen
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 5. Handle User Input via Streamlit Chat Input Box
user_prompt = st.chat_input("Ask for breakfast ideas or follow-up questions...")

if user_prompt:
    # Render user prompt immediately in UI
    with st.chat_message("user"):
        st.markdown(user_prompt)
    
    # Store user message in history
    st.session_state.messages.append(HumanMessage(content=user_prompt))

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # Prompt Template with Chat History Placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful nutrition assistant. Provide concise, friendly answers."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    chain = prompt | llm

    # Render assistant response container
    with st.chat_message("assistant"):
        # Helper generator function for real-time streaming
        def stream_response():
            for chunk in chain.stream({
                "chat_history": st.session_state.messages[:-1], # all history except current prompt
                "input": user_prompt
            }):
                yield chunk.content

        # st.write_stream consumes the generator and writes text token-by-token
        full_response = st.write_stream(stream_response())

    # Save the full AI response to chat history
    st.session_state.messages.append(AIMessage(content=full_response))