import streamlit as st
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = "http://127.0.0.1:8000"

# Initialize session state with minimal data
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "resume_keywords" not in st.session_state:
    st.session_state.resume_keywords = []
if "ui_initialized" not in st.session_state:
    st.session_state.ui_initialized = False

# Limit chat history to prevent memory issues
def trim_chat_history():
    if len(st.session_state.chat_history) > 20:  # Keep last 20 messages
        st.session_state.chat_history = st.session_state.chat_history[-20:]

st.title("💼 Career Chat AI")
st.markdown("Upload your resume and ask career questions to get personalized company recommendations!")

# Sidebar for resume management
with st.sidebar:
    st.header("Resume Upload")
    uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        with st.spinner("Uploading resume..."):
            try:
                res = requests.post(
                    f"{API_URL}/upload_resume",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                )
                res.raise_for_status()
                data = res.json()
                st.session_state.resume_keywords = data["keywords"]
                st.success("Resume uploaded! Keywords: " + ", ".join(data["keywords"][:5]) + "...")
                st.session_state.chat_history.append(("Assistant", f"Resume uploaded successfully. Keywords: {', '.join(data['keywords'][:5])}..."))
                trim_chat_history()
            except requests.RequestException as e:
                st.error(f"Error uploading resume: {str(e)}")
                st.session_state.chat_history.append(("Assistant", f"Error uploading resume: {str(e)}"))
                trim_chat_history()
            except json.JSONDecodeError:
                st.error("Error: Invalid response from server. Please try again.")
                st.session_state.chat_history.append(("Assistant", "Error: Invalid server response during resume upload."))
                trim_chat_history()
    if st.button("Clear Resume"):
        st.session_state.resume_keywords = []
        st.session_state.chat_history = []
        st.success("Resume and chat history cleared.")
        st.session_state.chat_history.append(("Assistant", "Resume and chat history cleared successfully."))

# Chat input
query = st.text_input("Ask a career question (e.g., 'Find startups who are hiring')")
num_companies = st.number_input("Number of companies to show", min_value=1, max_value=3, value=3)

if st.button("Send"):
    if not query and not st.session_state.resume_keywords:
        st.warning("Please upload a resume or enter a query.")
        st.session_state.chat_history.append(("Assistant", "Please upload a resume or enter a query to proceed."))
        trim_chat_history()
    else:
        with st.spinner("Searching for companies... This may take a moment."):
            st.session_state.chat_history.append(("User", query if query else "(default search)"))
            keywords_str = ",".join(st.session_state.resume_keywords) if st.session_state.resume_keywords else ""
            params = {"query": query, "keywords": keywords_str}
            try:
                logger.info(f"Sending search request: {query}")
                response = requests.get(f"{API_URL}/search_companies", params=params)
                response.raise_for_status()
                results = response.json()
                if isinstance(results, dict) and "detail" in results:
                    st.error(f"Error: {results['detail']}")
                    st.session_state.chat_history.append(("Assistant", f"Error: {results['detail']}"))
                    trim_chat_history()
                else:
                    answer_text = ""
                    company_names = [c["name"] for c in results[:num_companies]]
                    scores = [c["score"] for c in results[:num_companies]]
                    for c in results[:num_companies]:
                        link_md = f"[{c['name']}]({c['linkedin_url']})" if c.get("linkedin_url") else c["name"]
                        answer_text += f"**{link_md}** ({c['score']}% match)\n{c['about']}\n**Why?** {c['explanation']}\n"
                        if c.get("missing_skills"):
                            answer_text += f"**Improve Your Fit:** {', '.join([s['skill'] for s in c['missing_skills']])} ([Learn More]({c['missing_skills'][0]['resource']})).\n\n"
                    st.session_state.chat_history.append(("Assistant", answer_text))
                    trim_chat_history()
                    
                    # Career path suggestion
                    if st.session_state.resume_keywords:
                        with st.spinner("Generating career path..."):
                            try:
                                logger.info("Requesting career path")
                                career_response = requests.get(f"{API_URL}/career_path", params={"keywords": keywords_str})
                                career_response.raise_for_status()
                                career_data = career_response.json()
                                st.session_state.chat_history.append(("Assistant", f"**Career Path:** {career_data['path']}"))
                                trim_chat_history()
                            except (requests.RequestException, json.JSONDecodeError) as e:
                                logger.error(f"Career path request failed: {str(e)}")
                                st.session_state.chat_history.append(("Assistant", f"Sorry, I couldn't generate a career path: {str(e)}"))
                                trim_chat_history()
                    
                    # Bar chart for match scores (render only after results)
                    if company_names and scores:
                        st.markdown("**Company Match Scores**")
                        st.components.v1.html(f"""
                        <canvas id="matchChart"></canvas>
                        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                        <script>
                        new Chart(document.getElementById('matchChart'), {{
                            type: 'bar',
                            data: {{
                                labels: {json.dumps(company_names)},
                                datasets: [{{
                                    label: 'Match Score (%)',
                                    data: {json.dumps(scores)},
                                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                                    borderColor: '#36A2EB',
                                    borderWidth: 1
                                }}]
                            }},
                            options: {{
                                scales: {{ y: {{ beginAtZero: true, max: 100 }} }}
                            }}
                        }});
                        </script>
                        """, height=300)
            except requests.RequestException as e:
                st.error(f"Error: Search failed due to network issue: {str(e)}")
                st.session_state.chat_history.append(("Assistant", f"Error: Search failed due to network issue: {str(e)}"))
                trim_chat_history()
            except json.JSONDecodeError:
                st.error("Error: Invalid response from server. Please try again.")
                st.session_state.chat_history.append(("Assistant", "Error: Invalid server response during search."))
                trim_chat_history()

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "User":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown("**Career AI:**")
        st.markdown(message, unsafe_allow_html=True)

# Mark UI as initialized
st.session_state.ui_initialized = True