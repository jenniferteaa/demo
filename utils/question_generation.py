"""
question_generation.py
 
Module for generating fact-checking questions for atomic statements in the RARR framework.
"""
 
import os
import time
import logging
from typing import List, Dict, Optional
 
import openai
from openai import OpenAIError  # Import the specific error class you need

 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
 
 
def parse_api_response(api_response: str, required_questions: int = 3) -> List[str]:
    """
    Extract up to `required_questions` questions from the GPT response text.
    Ensures we return exactly `required_questions` questions,
    padding with generic placeholders if fewer are found.
    """
    try:
        questions = []
        # Split on new lines, looking for lines starting with a digit (e.g., "1.", "2)").
        for line in api_response.split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                # Find the first space after the digit or punctuation
                split_index = line.find(" ")
                if split_index != -1:
                    # Clean up leftover punctuation
                    question = line[split_index:].strip().lstrip(".- )")
                    # Ensure question starts uppercase for consistency
                    if question and question[0].isupper():
                        questions.append(question)
 
        # Remove duplicates (preserving first occurrence)
        unique_questions = list(dict.fromkeys(questions))
 
        # If we got more than needed, truncate
        if len(unique_questions) > required_questions:
            return unique_questions[:required_questions]
 
        # If fewer, pad with fallback questions
        while len(unique_questions) < required_questions:
            if not unique_questions:
                unique_questions.append("What specific evidence supports this statement?")
            else:
                base_q = unique_questions[-1].rstrip("?")
                new_q = f"What sources or evidence can verify that {base_q.lower()}?"
                unique_questions.append(new_q)
        
        return unique_questions
 
    except Exception as e:
        logger.error(f"Error parsing API response: {e}")
        # Fallback if parsing fails
        return [
            "What evidence supports this statement?",
            "What sources can verify this claim?",
            "How can this information be validated?"
        ][:required_questions]
 
 
def generate_contextual_prompt(
    claim: str,
    context: Optional[str] = None,
    previous_questions: Optional[List[str]] = None
) -> str:
    """
    Builds a user prompt for GPT, including context (previous statements)
    and previously asked questions to avoid repetition.
    """
    prompt_parts = []
    
    if context:
        prompt_parts.append(
            "Previous context for reference:\n"
            f"{context}\n\n"
            "Current statement to verify:\n"
            f"{claim}\n"
        )
    else:
        prompt_parts.append(f"Statement to verify:\n{claim}\n")
    
    prompt_parts.append(
        "\nGenerate exactly 3 specific questions that:"
        "\n1. Focus on verifiable facts and claims"
        "\n2. Target specific details that can be fact-checked"
        "\n3. Can be answered with credible sources"
    )
    
    if previous_questions:
        prompt_parts.append("\nPreviously asked questions (avoid repeating):")
        for q in previous_questions:
            prompt_parts.append(f"- {q}")
    
    prompt_parts.append("\nNew questions:")
    return "\n".join(prompt_parts)
 
 
def are_questions_similar(q1: str, q2: str, similarity_threshold: float = 0.8) -> bool:
    """
    Simple word-overlap-based similarity check to prevent repeated questions.
    If overlap is above `similarity_threshold`, they're considered similar.
    """
    try:
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words1 = {w.lower() for w in q1.split() if w.lower() not in common_words}
        words2 = {w.lower() for w in q2.split() if w.lower() not in common_words}
        
        if not words1 or not words2:
            return False
            
        overlap = len(words1.intersection(words2))
        similarity = overlap / max(len(words1), len(words2))
        
        return similarity > similarity_threshold
 
    except Exception as e:
        logger.error(f"Error comparing questions: {e}")
        return False
 
 
def run_rarr_question_generation(
    claim: str,
    model: str,
    prompt: str,
    temperature: float = 0.7,
    num_retries: int = 5,
    context: Optional[str] = None,
    previous_questions: Optional[List[str]] = None,
    required_questions: int = 3
) -> List[str]:
    """
    Generates `required_questions` fact-checking questions for a given claim.
    - Leverages `context` (previous statements) and `previous_questions` to avoid duplicates.
    - Retries up to `num_retries` times on transient errors.
    - Ensures we end up with exactly `required_questions` unique questions.
    """
    try:
        # Build the final user prompt
        contextual_prompt = generate_contextual_prompt(claim, context, previous_questions)
        system_prompt = (
            "You are a fact-checking assistant specialized in generating specific, "
            "focused questions to verify factual claims."
        )
        gpt_input = f"{prompt}\n{contextual_prompt}".strip()
 
        all_questions = set()
        attempts = 0
        max_rounds = 3  # The number of GPT call rounds
 
        # Attempt multiple GPT calls, stopping early if we reach the required number of questions
        while len(all_questions) < required_questions and attempts < max_rounds:
            for attempt_i in range(num_retries):
                try:
                    # Create a ChatCompletion with the 'system' + 'user' message
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": gpt_input}
                        ],
                        temperature=temperature
                    )
 
                    # Parse the GPT output
                    new_questions = parse_api_response(
                        response.choices[0].message.content,
                        required_questions
                    )
                    
                    # Filter out questions too similar to what's already generated
                    for q in new_questions:
                        if not any(are_questions_similar(q, existing_q) for existing_q in all_questions):
                            all_questions.add(q)
                    break  # Successfully retrieved questions, break the retry loop
 
                except OpenAIError as oe:
                    logger.warning(f"OpenAI API Error (attempt {attempt_i + 1}/{num_retries}): {oe}")
                    if attempt_i < num_retries - 1:
                        time.sleep(2)  # short backoff before next attempt
                except Exception as e:
                    logger.warning(f"API Error (attempt {attempt_i + 1}/{num_retries}): {e}")
                    if attempt_i < num_retries - 1:
                        time.sleep(2)  # short backoff before next attempt
 
            attempts += 1
 
        # Limit to required_questions
        questions_list = list(all_questions)[:required_questions]
 
        # If still fewer, fill with placeholders
        while len(questions_list) < required_questions:
            questions_list.append(
                f"What evidence supports the claim that {claim.lower().rstrip('.')}?"
            )
 
        return questions_list
 
    except Exception as e:
        logger.error(f"Error in question generation: {e}")
        # Fallback if all fails
        return [
            f"What evidence supports the claim that {claim.lower().rstrip('.')}?",
            "What sources can verify this information?",
            "How can these specific details be validated?"
        ][:required_questions]
 
 
def process_atomic_statements(
    statements: List[str],
    model: str,
    base_prompt: str,
    temperature: float = 0.7
) -> Dict[str, List[str]]:
    """
    For each statement in `statements`, generates 3 questions using `run_rarr_question_generation`.
    Maintains a set of previously asked questions across statements to reduce duplicates.
    Returns a dict: {statement: [questions]}
    """
    try:
        results = {}
        # Accumulate all previously generated questions to avoid repeats
        all_previous_questions = []
        
        for i, statement in enumerate(statements):
            logger.info(f"Processing statement {i + 1}/{len(statements)}")
 
            # For context, we can pass all prior statements (unrevised)
            context = " ".join(statements[:i]) if i > 0 else ""
 
            questions = run_rarr_question_generation(
                claim=statement,
                model=model,
                prompt=base_prompt,
                temperature=temperature,
                num_retries=3,
                context=context,
                previous_questions=all_previous_questions,
                required_questions=3
            )
            
            results[statement] = questions
            # Add newly generated questions to the global list to avoid duplication later
            all_previous_questions.extend(questions)
            logger.info(f"Generated {len(questions)} questions for statement {i + 1}")
        
        return results
 
    except Exception as e:
        logger.error(f"Error processing atomic statements: {e}")
        fallback = [
            "What evidence supports this statement?",
            "What sources can verify this information?",
            "How can these claims be validated?"
        ]
        return {statement: fallback for statement in statements}
 
 
# Example direct usage:
if __name__ == "__main__":
    # Example usage
    example_statements = [
        "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
        "She received three consecutive Emmy nominations for the role.",
        "She became one of the highest-paid television actors by 2017."
    ]
    
    example_prompt = "Generate specific questions to verify the following information."
    
    results = process_atomic_statements(
        statements=example_statements,
        model="gpt-3.5-turbo",
        base_prompt=example_prompt,
        temperature=0.7
    )
    
    print("\nGenerated Questions per Statement:")
    for statement, questions in results.items():
        print(f"\nStatement: {statement}")
        for i, question in enumerate(questions, 1):
            print(f"{i}. {question}")
 
 




# """Utils for running question generation."""
# import os
# import time
# from typing import List, Any
# import logging
# import openai

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_openai_client():
#     """Sets up the OpenAI client using the API key from the environment."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     if hasattr(openai, 'OpenAI'):
#         # Newer version of the library
#         return openai.OpenAI(api_key=api_key)
#     else:
#         # Older version of the library
#         openai.api_key = api_key
#         return openai

# # Initialize the OpenAI client
# client = setup_openai_client()

# def parse_api_response(api_response: str) -> List[str]:
#     """Extract questions from the GPT API response."""
#     search_string = "I googled:"
#     questions = [line.split(search_string)[1].strip() for line in api_response.split("\n") if search_string in line]
#     return questions

# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_rounds: int,
#     context: str = None,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim."""
#     if client is None:
#         client = setup_openai_client()

#     gpt3_input = prompt.format(context=context, claim=claim).strip() if context else prompt.format(claim=claim).strip()

#     questions = set()
#     max_attempts = num_rounds * 4  # Define maximum attempts based on rounds
#     total_attempts = 0

#     while len(questions) < 5 and total_attempts < max_attempts:
#         for attempt in range(num_retries):
#             try:
#                 # Adapt to new or old OpenAI library version
#                 if hasattr(client, 'chat'):
#                     # New OpenAI client version for chat models
#                     response = client.chat.completions.create(
#                         model=model,
#                         messages=[
#                             {"role": "system", "content": "Generate unique, diverse questions about the given claim."},
#                             {"role": "user", "content": gpt3_input}
#                         ],
#                         temperature=min(temperature + (total_attempts * 0.1), 1.0),  # Gradually increase temperature, capped at 1.0
#                         max_tokens=256,
#                         n=2,  # Generate 2 completions per request
#                     )
#                     response_texts = [choice.message.content.strip() for choice in response.choices]
#                 else:
#                     response = client.Completion.create(
#                         model=model,
#                         prompt=gpt3_input,
#                         temperature=min(temperature + (total_attempts * 0.1), 1.0),  # Gradually increase temperature
#                         max_tokens=256,
#                         n=2,
#                     )
#                     response_texts = [choice.text.strip() for choice in response.choices]

#                 # Parse and add new unique questions
#                 for response_text in response_texts:
#                     cur_round_questions = parse_api_response(response_text)
#                     new_questions = set(cur_round_questions) - questions
#                     questions.update(new_questions)
                
#                 logging.info(f"Attempt {total_attempts + 1}: Generated {len(new_questions)} new unique questions")

#                 if len(questions) >= 5:
#                     break
#             except Exception as e:
#                 logging.warning(f"Attempt {total_attempts + 1}, Try {attempt + 1}: {str(e)}. Retrying...")
#                 time.sleep(1)

#             if attempt == num_retries - 1:
#                 logging.error(f"Failed to generate questions in attempt {total_attempts + 1} after {num_retries} tries")
                
#         total_attempts += 1

#     questions = list(sorted(questions))[:5]  # Limit to 5 unique questions
#     logging.info(f"Total unique questions generated: {len(questions)}")
#     if len(questions) < 5:
#         logging.warning(f"Failed to generate 5 unique questions. Only generated {len(questions)} questions.")
#     return questions


# """Utils for running question generation."""
# import os
# import time
# from typing import List, Any
# import logging
# import openai

# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_openai_client():
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     if hasattr(openai, 'OpenAI'):
#         # New version of the library
#         return openai.OpenAI(api_key=api_key)
#     else:
#         # Old version of the library
#         openai.api_key = api_key
#         return openai

# # Initialize the OpenAI client
# client = setup_openai_client()

# def parse_api_response(api_response: str) -> List[str]:
#     """Extracts questions from the GPT-3 API response.

#     This assumes questions are formatted as a list in the API response.
#     Modify this if the format is different in your responses.
    
#     Args:
#         api_response: The response from GPT-3 as a string.
    
#     Returns:
#         A list of questions extracted from the response.
#     """
#     questions = []
#     for line in api_response.split("\n"):
#         if line.strip():
#             questions.append(line.strip())
#     return questions

# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_rounds: int,
#     context: str = None,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim.

#     Args:
#         claim: The claim to generate questions about.
#         model: The OpenAI GPT model to use.
#         prompt: The prompt template for question generation.
#         temperature: Sampling temperature to control randomness.
#         num_rounds: Number of rounds to generate questions.
#         context: Additional context for the claim, if any.
#         num_retries: Number of retries in case of API failure.
#         client: The OpenAI client instance.
    
#     Returns:
#         A list of unique, generated questions.
#     """
#     if client is None:
#         client = setup_openai_client()

#     if context:
#         gpt3_input = prompt.format(context=context, claim=claim).strip()
#     else:
#         gpt3_input = prompt.format(claim=claim).strip()

#     questions = set()
#     max_attempts = num_rounds * 4  # Allow up to 4 attempts per round
#     total_attempts = 0

#     while len(questions) < 5 and total_attempts < max_attempts:
#         for attempt in range(num_retries):
#             try:
#                 if hasattr(client, 'chat'):
#                     # Using the chat-based OpenAI model
#                     response = client.chat.completions.create(
#                         model=model,
#                         messages=[
#                             {"role": "system", "content": "Generate unique, diverse questions about the given claim."},
#                             {"role": "user", "content": gpt3_input}
#                         ],
#                         temperature=temperature + (total_attempts * 0.1),  # Gradually increase temperature
#                         max_tokens=256,
#                         n=2,  # Generate 2 completions per request
#                     )
#                     response_texts = [choice.message.content.strip() for choice in response.choices]
#                 else:
#                     # Using the older OpenAI completion API
#                     response = client.Completion.create(
#                         model=model,
#                         prompt=gpt3_input,
#                         temperature=temperature + (total_attempts * 0.1),  # Gradually increase temperature
#                         max_tokens=256,
#                         n=2,  # Generate 2 completions per request
#                     )
#                     response_texts = [choice.text.strip() for choice in response.choices]

#                 # Parse the generated responses
#                 for response_text in response_texts:
#                     cur_round_questions = parse_api_response(response_text)
#                     new_questions = set(cur_round_questions) - questions
#                     questions.update(new_questions)
                
#                 logging.info(f"Attempt {total_attempts + 1}: Generated {len(new_questions)} new unique questions")
                
#                 if len(questions) >= 5:
#                     break
#             except Exception as e:
#                 logging.warning(f"Attempt {total_attempts + 1}, Retry {attempt + 1}: {str(e)}. Retrying...")
#                 if attempt == num_retries - 1:
#                     logging.error(f"Failed to generate questions in attempt {total_attempts + 1} after {num_retries} tries")
#                 time.sleep(1)
#         total_attempts += 1

#     # Ensure the function only returns 5 unique questions
#     questions = list(sorted(questions))[:5]
#     logging.info(f"Total unique questions generated: {len(questions)}")
    
#     if len(questions) < 5:
#         logging.warning(f"Failed to generate 5 unique questions. Only generated {len(questions)} questions.")
    
#     return questions




# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_rounds: int,
#     context: str = None,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim.
    
#     Given a piece of text (claim), we use GPT-3 to generate questions that question the
#     information in the claim. We run num_rounds of sampling to get a diverse set of questions.
    
#     Args:
#         claim: Text to generate questions off of.
#         model: Name of the OpenAI GPT model to use.
#         prompt: The prompt template to query GPT with.
#         temperature: Temperature to use for sampling questions. 0 represents greedy decoding.
#         num_rounds: Number of times to sample questions.
#         context: Optional context for the claim.
#         num_retries: Number of retries in case of API errors.
#         client: OpenAI client instance.
#     Returns:
#         questions: A list of questions.
#     """
#     if client is None:
#         client = setup_openai_client()

#     if context:
#         gpt3_input = prompt.format(context=context, claim=claim).strip()
#     else:
#         gpt3_input = prompt.format(claim=claim).strip()

#     questions = set()
#     for round in range(num_rounds):
#         for attempt in range(num_retries):
#             try:
#                 if hasattr(client, 'chat'):
#                     # New OpenAI client
#                     response = client.chat.completions.create(
#                         model=model,
#                         messages=[{"role": "user", "content": gpt3_input}],
#                         temperature=temperature,
#                         max_tokens=256,
#                     )
#                     cur_round_questions = parse_api_response(response.choices[0].message.content.strip())
#                 else:
#                     # Old OpenAI client
#                     response = client.Completion.create(
#                         model=model,
#                         prompt=gpt3_input,
#                         temperature=temperature,
#                         max_tokens=256,
#                     )
#                     cur_round_questions = parse_api_response(response.choices[0].text.strip())

#                 questions.update(cur_round_questions)
#                 logging.info(f"Round {round + 1}: Generated {len(cur_round_questions)} questions")
#                 break
#             except Exception as e:
#                 logging.warning(f"Round {round + 1}, Attempt {attempt + 1}: {str(e)}. Retrying...")
#                 if attempt == num_retries - 1:
#                     logging.error(f"Failed to generate questions in round {round + 1} after {num_retries} attempts")
#                 time.sleep(1)

#     questions = list(sorted(questions))
#     logging.info(f"Total unique questions generated: {len(questions)}")
#     return questions
# """Utils for running question generation."""
# import os
# import time
# from typing import List

# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


# def parse_api_response(api_response: str) -> List[str]:
#     """Extract questions from the GPT-3 API response.

#     Our prompt returns questions as a string with the format of an ordered list.
#     This function parses this response in a list of questions.

#     Args:
#         api_response: Question generation response from GPT-3.
#     Returns:
#         questions: A list of questions.
#     """
#     search_string = "I googled:"
#     questions = []
#     for question in api_response.split("\n"):
#         # Remove the search string from each question
#         if search_string not in question:
#             continue
#         question = question.split(search_string)[1].strip()
#         questions.append(question)

#     return questions


# def run_rarr_question_generation(
#     claim: str,
#     model: str,
#     prompt: str,
#     temperature: float,
#     num_rounds: int,
#     context: str = None,
#     num_retries: int = 5,
# ) -> List[str]:
#     """Generates questions that interrogate the information in a claim.

#     Given a piece of text (claim), we use GPT-3 to generate questions that question the
#     information in the claim. We run num_rounds of sampling to get a diverse set of questions.

#     Args:
#         claim: Text to generate questions off of.
#         model: Name of the OpenAI GPT-3 model to use.
#         prompt: The prompt template to query GPT-3 with.
#         temperature: Temperature to use for sampling questions. 0 represents greedy deconding.
#         num_rounds: Number of times to sample questions.
#     Returns:
#         questions: A list of questions.
#     """
#     if context:
#         gpt3_input = prompt.format(context=context, claim=claim).strip()
#     else:
#         gpt3_input = prompt.format(claim=claim).strip()

#     questions = set()
#     for _ in range(num_rounds):
#         for _ in range(num_retries):
#             try:
#                 response = openai.Completion.create(
#                     model=model,
#                     prompt=gpt3_input,
#                     temperature=temperature,
#                     max_tokens=256,
#                 )
#                 cur_round_questions = parse_api_response(
#                     response.choices[0].text.strip()
#                 )
#                 questions.update(cur_round_questions)
#                 break
#             except openai.error.OpenAIError as exception:
#                 print(f"{exception}. Retrying...")
#                 time.sleep(1)

#     questions = list(sorted(questions))
#     return questions