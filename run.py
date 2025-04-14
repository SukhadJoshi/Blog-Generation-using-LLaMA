import streamlit as st
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title=" Blog Generator",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ------------------ CUSTOM STYLING ------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #2c3e50, #4ca1af);
        color: white;
    }
    .css-1cpxqw2 {
        color: white;
    }
    input, .stSelectbox, .stButton button {
        background-color: #1e272e;
        color: white;
        border-radius: 8px;
    }
    .stButton>button {
        background-color: #00a8ff;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.header("📝 Blog Generator")
st.subheader("Need a blog post fast? Just tell me what it's about.")

def getllamaresponse(input_text, no_words, blog_style):
    # Loading LLaMA 2 model
    lama = CTransformers(
        model="Project/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        config={"max_new_tokens": 256, "temperature": 0.01}
    )

    # Prompt template
    template = """Write a blog for {blog_style} job profile on the topic 
    "{input_text}" within {no_words} words."""

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Generating the response
    formatted_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    )

    res = lama(formatted_prompt)
    return res

# ------------------ INPUT FIELDS ------------------
input_text = st.text_input("🧠 Enter the Blog Topic:")

col1, col2 = st.columns([5, 5])
with col1:
    no_words = st.text_input("✏️ Number of Words:")
with col2:
    blog_style = st.selectbox(
        "🎯 Writing the Blog for:",
        ("Casual", "Storytelling", "Technical"),
        index=0
    )

submit = st.button("Generate the Blog")

# ------------------ OUTPUT ------------------
if submit:
    with st.spinner("Generating your blog..."):
        response = getllamaresponse(input_text, no_words, blog_style)
        st.success("✅ Blog generated successfully!")
        st.markdown(response.replace("\n", "  \n"))

# ------------------ FOOTER ------------------
st.markdown("""---""")

