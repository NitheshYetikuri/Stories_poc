
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
from java_analyzer import clone_and_process_repo, delete_projects_folder
from stories import run_story_generation

st.set_page_config(page_title="DevStoryAI: Java Project Story Generator", layout="centered")
st.title("DevStoryAI: Java Project Story Generator")

# Step 1: Input GitHub URL
github_url = st.text_input("Enter the GitHub repository URL of the Java project:")

if github_url:
    if st.button("Clone & Analyze Project"):
        with st.spinner("Cloning and analyzing the repository. This may take a few minutes..."):
            try:
                clone_and_process_repo(github_url,os.environ.get("GOOGLE_API_KEY") )
                st.success("Repository cloned, parsed, and vector DB created!")
                st.session_state["repo_ready"] = True
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state["repo_ready"] = False


# Step 2: Chat-like Query Interface for Story Generation
if st.session_state.get("repo_ready", False):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.subheader("Chat with DevStoryAI")
    for chat in st.session_state["chat_history"]:
        if chat["role"] == "user":
            st.chat_message("user").write(chat["content"])
        else:
            st.chat_message("assistant").write(chat["content"])

    user_query = st.chat_input("Type your feature/change request and press Enter...")
    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.spinner("Generating user stories using AI agents..."):
            try:
                result = run_story_generation(user_query)
                # If result is a dict or contains multiple outputs, extract only the Tech Lead's user stories
                user_stories = None
                if isinstance(result, dict):
                    # Try to find the last or 'Tech Lead' output
                    for v in result.values():
                        if isinstance(v, str) and ("Developer User Story" in v or "Tester User Story" in v):
                            user_stories = v
                    if not user_stories:
                        # fallback: get last string value
                        user_stories = list(result.values())[-1]
                elif isinstance(result, str):
                    user_stories = result
                else:
                    user_stories = str(result)
                # Only print the user stories (final output)
                st.session_state["chat_history"].append({"role": "assistant", "content": user_stories})
                st.chat_message("assistant").write(user_stories)
            except Exception as e:
                st.session_state["chat_history"].append({"role": "assistant", "content": f"Error: {e}"})
                st.chat_message("assistant").write(f"Error: {e}")

# Step 3: Project Deletion Logic
if st.button("Exit & Delete Project Data", type="primary"):
    with st.spinner("Deleting all cloned project data and cleaning up..."):
        try:
            delete_projects_folder()
            st.success("All project data deleted. You can safely close the app.")
            st.session_state.clear()
        except Exception as e:
            st.error(f"Error: {e}")
