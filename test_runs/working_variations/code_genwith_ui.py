import streamlit as st
import pandas as pd
import re
from llama_cpp import Llama
import io
import sys

# Initialize the LLaMA model
llm = Llama(
    model_path="./Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf",
    max_tokens=8192,
    temperature=0.1,
    n_threads=8,
    n_ctx=2048
)

# Function to generate a prompt for the LLM
def generate_prompt(task_description, columns):
    prompt = f"""
You are an expert data analyst. Your task is to write Python code that solves data analysis problems based on the given description.

Try to understand the question. I had data load in df and now i need to find

Task: {task_description}

and I have these columns in my data  

Available columns in the dataset: {', '.join(columns)}

Please generate the Python code that solves this task using the pandas library. If the task requires filtering a DataFrame, make sure the filtering criteria are clearly mentioned in the code. If the task involves calculations like averages, sums, or groupings, include the corresponding code for those operations.

Be sure to include any necessary imports (like pandas) and explain the steps in the code comments.

The Python code should be concise and clear.
"""
    return prompt

# Function to extract Python code from LLM response
def extract_code_from_response(response):
    code_pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    match = code_pattern.search(response)
    if match:
        python_code = match.group(1).strip()
        python_code = re.sub(r"pd\.read_csv\([^\)]*\)\s*", "", python_code)
        return python_code
    else:
        return None

# Function to execute Python code
def execute_python_code(python_code, df):
    try:
        # Redirect stdout to capture printed output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Define local variables, including DataFrame `df`
        local_vars = {'df': df, 'pd': pd}
        
        # Execute the extracted Python code
        exec(python_code, {}, local_vars)
        
        sys.stdout = sys.__stdout__
        
        return captured_output.getvalue()  # Return captured output

    except Exception as e:
        return f"Error executing code: {e}"

# Streamlit app
def main():
    st.title("LLM Data Analysis Assistant")
    st.write("Upload a CSV file, ask questions about the data, and get Python code + results!")

    # Initialize session state for chat history and DataFrame
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "df" not in st.session_state:
        st.session_state.df = None

    # File uploader
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        # Load the CSV file into a DataFrame
        st.session_state.df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(st.session_state.df.head())

    # Chat interface
    if st.session_state.df is not None:
        user_input = st.text_input("Ask a question about your data:")

        # Check if user wants to exit
        if user_input.lower() in ["quit", "exit"]:
            st.write("Ending the chat. Goodbye!")
            st.session_state.chat_history = []
            st.session_state.df = None
            st.stop()  # Ends the script execution

        # Quit button to clear the session
        # if st.button("Quit"):
        #     st.session_state.chat_history = []
        #     st.session_state.df = None
        #     st.experimental_rerun()
        #     st.exper

        if st.button("Submit") and user_input:
            # Extract column names for prompt generation
            columns = st.session_state.df.columns.tolist()
            task_prompt = generate_prompt(user_input, columns)
            combined_prompt = f"{task_prompt}\nLLM:"

            # Get response from LLM
            response = llm(combined_prompt, max_tokens=8192, stop=["\nYou:"])
            llm_response = response['choices'][0]['text'].strip()
            
            # Add the user question and LLM response to chat history
            st.session_state.chat_history.append({"user": user_input, "llm": llm_response})

            # Extract and execute the Python code from LLM response
            python_code = extract_code_from_response(llm_response)
            answer = "No executable code generated."
            if python_code:
                answer = execute_python_code(python_code, st.session_state.df)

            # Display the response and result
            st.session_state.chat_history[-1]["answer"] = answer

        # Display chat history
        for chat in st.session_state.chat_history:
            st.write(f"You: {chat['user']}")
            st.write(f"LLM: {chat['llm']}")
            st.code(chat.get("answer", ""), language="python")

if __name__ == "__main__":
    main()
