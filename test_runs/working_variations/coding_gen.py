import pandas as pd
import re
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from llama_cpp import Llama

# Load the dataset
file_path = "data/world_bank_dataset.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Initialize LLM (Llama 2 model as per your request)
# llm = CTransformers(
#     model="llama-2-7b-chat.ggmlv3.q5_K_M.bin",
#     model_type="llama",
#     config={
#         'max_new_tokens': 1024,
#         'temperature': 0.1,
#         'context_length': 2048
#     }
# )

llm = Llama(
    model_path="./mistral-7b-instruct-v0.2-code-ft.Q5_K_M.gguf",
    max_tokens=1024,     # Set default max_tokens
    temperature=0.1,     # Set default temperature
    n_threads=8          # Set default number of threads
)


# Step 1: Generate Python Code from the Question
def generate_code(question: str) -> str:
    prompt = f"""
    You are an expert data analyst. I have a dataset with several columns, and I need your help in answering the following question:

    Question: {question}

    Please feel free to understand the question fully, and use Python to generate a solution. You may use pandas or any necessary Python libraries to answer this. The solution should be in Python code that can directly work with a dataset represented as a pandas DataFrame.

    Your answer should be a Python code snippet that solves the problem effectively.
    """
    response = llm(prompt)  # Get code from LLaMA
    code = response['choices'][0]['text'] if 'choices' in response else ""
    print(code)
    return code

# Step 2: Extract and Review the Code (Proofreading)
def extract_columns_from_code(code: str) -> list:
    # Extract column names from the generated Python code using regex
    column_names = re.findall(r"['\"](\w+)['\"]", code)
    return column_names

def proofread_code(columns_in_code: list, df: pd.DataFrame) -> str:
    # Step 3: Check columns in the code and compare them to the dataset columns
    dataset_columns = df.columns.tolist()
    incorrect_columns = [col for col in columns_in_code if col not in dataset_columns]
    
    if incorrect_columns:
        # Generate proofreading corrections based on the dataset
        prompt = f"The following columns are used in the Python code: {columns_in_code}. Check and correct any mistakes based on this dataset's columns: {dataset_columns}."
        corrected_columns = llm(prompt)  # Get correction suggestions from LLaMA
        return corrected_columns
    else:
        return "No corrections needed."

# Step 4: Execute the corrected code
def execute_corrected_code(code: str, df: pd.DataFrame):
    try:
        # Execute the corrected Python code
        # Define the scope for exec to ensure safe execution
        local_scope = {"df": df}
        
        # Execute the code within the context of the pandas DataFrame
        exec(code, {}, local_scope)

        # Return the result from local scope after execution
        # We'll check if the local scope contains a variable representing the result
        result = local_scope.get('result', None)

        if result is not None:
            return result
        else:
            # If 'result' is not defined, just return the dataframe (or other result) 
            return "The code ran successfully but no result was returned from the executed code."

    except Exception as e:
        # Return a detailed error message
        return f"Error executing code: {str(e)}"

# Step 5: Explain the results (using LLaMA to explain the output)
def explain_results(results: pd.DataFrame) -> str:
    prompt = f"Explain the following data analysis result: {results.to_string()}"
    explanation = llm(prompt)  # Use LLaMA to explain the results
    return explanation

# Main function to handle the entire process
def handle_data_analysis(question: str):
    # Step 1: Generate Python code based on the user question
    generated_code = generate_code(question)
    print("Generated Python code:")
    print(generated_code)
    
    # Step 2: Extract columns from the generated code
    columns_in_code = extract_columns_from_code(generated_code)
    print("Columns found in the code:", columns_in_code)
    
    # Step 3: Proofread and validate the columns
    proofreading_result = proofread_code(columns_in_code, df)
    print("Proofreading result (corrections):")
    print(proofreading_result)
    
    # Correct the code based on proofreading results
    corrected_code = generated_code  # Here, apply corrections to the generated code if needed
    
    # Step 4: Execute the corrected code
    result = execute_corrected_code(corrected_code, df)
    print("Result of the groupby operation:")
    print(result)
    
    # Step 5: Explain the results
    explanation = explain_results(result)
    print("Explanation of the results:")
    print(explanation)

# Example user question
question = "what is brazil average population?"

# Call the main function to handle the entire data analysis process
handle_data_analysis(question)
