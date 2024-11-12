import streamlit as st
import pandas as pd
import re
from llama_cpp import Llama
import io
import sys

# Initialize the LLaMA model
llm = Llama(
    model_path="./Qwen2.5.1-Coder-7B-Instruct-Q5_K_M.gguf",  # Replace with your model path
    max_tokens=8192,
    temperature=0.1,
    n_threads=8,
    n_ctx=2048
)

# Function to generate a prompt for the LLM
def generate_prompt(task_description, columns):
    prompt = f"""
You are an expert data analyst. Your task is to write Python code that solves data analysis problems based on the given description.

Try to understand the question. I had data load in df and now I need to find

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

    # Initialize session state for DataFrame and chat history
    if "df" not in st.session_state:
        st.session_state.df = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history as an empty list

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    
    if uploaded_file:
        # Load the CSV file into a DataFrame only once
        if st.session_state.df is None:  # Check if it's already loaded
            st.session_state.df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(st.session_state.df.head())

            # Add to chat history that file was uploaded
            st.session_state.chat_history.append({"author": "assistant", "message": "File uploaded successfully! Ask me questions about your data."})

    # **Display the conversation using chat_message directly**
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["author"]):
                st.write(msg["message"])
                if msg.get("answer"):
                    st.code(msg["answer"], language="python")
    
    # Input field for the user to type their message
    user_message = st.text_input("You: ", "")

    # Check if the user wants to exit
    if user_message.lower() in ["quit", "exit"]:
        with st.chat_message("assistant"):
            st.write("Goodbye! The session is ending.")
        return  # Stop further processing

    if user_message:
        # Add the user's message to the chat history and display it
        with st.chat_message("user"):
            st.write(user_message)

        st.session_state.chat_history.append({"author": "user", "message": user_message})

        # Generate a task-specific prompt for the LLM
        columns = st.session_state.df.columns.tolist()
        task_prompt = generate_prompt(user_message, columns)
        combined_prompt = f"{task_prompt}\nLLM:"

        # Get response from LLM
        response = llm(combined_prompt, max_tokens=8192, stop=["\nYou:"])
        llm_response = response['choices'][0]['text'].strip()

        # Add the assistant's response to the chat history
        with st.chat_message("assistant"):
            st.write(llm_response)
        st.session_state.chat_history.append({"author": "assistant", "message": llm_response})

        # Extract and execute the Python code from the LLM response
        python_code = extract_code_from_response(llm_response)
        answer = "No executable code generated."
        if python_code:
            answer = execute_python_code(python_code, st.session_state.df)

        # Add the answer (output or error message) to the chat history
        with st.chat_message("assistant"):
            st.write("Here is the code:")
            st.code(answer, language="python")
        st.session_state.chat_history.append({"author": "assistant", "message": "Here is the code:", "answer": answer})

# Run the app
if __name__ == "__main__":
    main()
