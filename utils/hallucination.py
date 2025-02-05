import itertools
import os
import time
import logging
from typing import Any, Dict, List, Tuple, Optional
 
import torch
from sentence_transformers import CrossEncoder
import openai
 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
 
# Initialize cross-encoder for evidence scoring
PASSAGE_RANKER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    max_length=512,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
 
def compute_score_matrix(
    questions: List[str],
    evidences: List[str],
    context: Optional[str] = None
) -> List[List[float]]:
    """
    Computes a score matrix where:
      - Rows = Questions
      - Columns = Evidence passages
    Each cell is the relevance score from the CrossEncoder model.
    Optionally prepends a shared `context` to each question for better scoring.
    """
    try:
        score_matrix = []
        for q in questions:
            # If context is provided, prepend it to the question for additional clarity
            contextualized_q = f"{context} {q}" if context else q
            evidence_pairs = [(contextualized_q, e) for e in evidences]
            
            with torch.no_grad():
                evidence_scores = PASSAGE_RANKER.predict(evidence_pairs).tolist()
            score_matrix.append(evidence_scores)
        return score_matrix
 
    except Exception as e:
        logger.error(f"Error computing score matrix: {e}")
        # Return a matrix of zeros if there's an error
        return [[0.0] * len(evidences) for _ in questions]
 
def question_coverage_objective_fn(
    score_matrix: List[List[float]],
    evidence_indices: List[int],
    question_groups: Optional[List[List[int]]] = None
) -> float:
    """
    Given a `score_matrix` and a list of selected evidence indices, computes a
    coverage-based objective to quantify how well the selected evidences cover
    all questions.
 
    If `question_groups` is provided, each group is treated as belonging to a
    specific atomic statement. The coverage for each group is averaged.
    """
    try:
        total = 0.0
 
        if question_groups:
            # Process questions by atomic statement or grouping
            for group in question_groups:
                group_scores = [score_matrix[q_idx] for q_idx in group]
                # For each question in the group, get the maximum coverage from selected evidences
                group_total = sum(
                    max(scores_for_question[j] for j in evidence_indices)
                    for scores_for_question in group_scores
                )
                # Normalize by the number of questions in the group
                total += group_total / len(group)
        else:
            # Process all questions together, each row in the score_matrix is a question
            for scores_for_question in score_matrix:
                total += max(scores_for_question[j] for j in evidence_indices)
        
        return total
 
    except Exception as e:
        logger.error(f"Error computing coverage objective: {e}")
        return 0.0
 
def select_evidences(
    example: Dict[str, Any],
    max_selected: int = 5,
    prefer_fewer: bool = False,
    atomic_statements: Optional[List[str]] = None,
    questions_per_statement: Optional[List[List[str]]] = None
) -> List[Dict[str, Any]]:
    """
    Selects an optimal subset of evidence passages to maximize question coverage.
    - `example`: A dictionary containing at least one `revisions` item with `evidences`.
    - `max_selected`: Maximum number of evidences to select.
    - `prefer_fewer`: If True, tries from 1 up to `max_selected`; otherwise tries exactly `max_selected`.
    - `atomic_statements`: List of statements (optional) for context.
    - `questions_per_statement`: If provided, indicates a per-statement grouping for questions.
 
    Returns a list of dicts: [{"text": evidence_text, "score": coverage_score}, ...]
    """
    try:
        # Gather questions
        if questions_per_statement:
            # Flatten questions and build groups to track each statement's questions
            questions = [q for sublist in questions_per_statement for q in sublist]
            question_groups = []
            current_idx = 0
            for statement_questions in questions_per_statement:
                group = list(range(current_idx, current_idx + len(statement_questions)))
                question_groups.append(group)
                current_idx += len(statement_questions)
        else:
            # No per-statement grouping; just use all questions in one set
            questions = sorted(set(example.get("questions", [])))
            question_groups = None
 
        # Extract evidence text from the example
        evidences = []
        if example.get("revisions") and example["revisions"][0].get("evidences"):
            # Gather unique evidence texts
            evidences = sorted(
                set(e["text"] for e in example["revisions"][0]["evidences"] if e.get("text"))
            )
        
        if not evidences:
            logger.warning("No evidences found in the provided example.")
            return []
 
        # Build optional context from atomic statements (excluding the last to avoid repetition)
        context = " ".join(atomic_statements[:-1]) if atomic_statements else None
        score_matrix = compute_score_matrix(questions, evidences, context)
 
        # Find the best combination of evidences to maximize coverage
        best_combo = tuple()
        best_objective_value = float("-inf")
        max_selected = min(max_selected, len(evidences))
 
        # If `prefer_fewer` is True, start from 1 up to max_selected
        min_selected = 1 if prefer_fewer else max_selected
 
        for num_selected in range(min_selected, max_selected + 1):
            for combo in itertools.combinations(range(len(evidences)), num_selected):
                objective_value = question_coverage_objective_fn(
                    score_matrix, combo, question_groups
                )
                if objective_value > best_objective_value:
                    best_combo = combo
                    best_objective_value = objective_value
 
        # Return selected evidence texts with the best coverage score
        return [
            {"text": evidences[idx], "score": float(best_objective_value)}
            for idx in best_combo
        ]
 
    except Exception as e:
        logger.error(f"Error selecting evidences: {e}")
        return []
 
def run_evidence_hallucination(
    query: str,
    model: str,
    prompt: str,
    context: Optional[str] = None,
    atomic_statement: Optional[str] = None,
    num_retries: int = 5
) -> Dict[str, str]:
    """
    Generates textual 'evidence' for a given query using the OpenAI API.
    - If context and atomic_statement are provided, they become part of the user/system messages.
    - Returns a dict with the 'text' field containing generated evidence.
    """
    try:
        # Build system prompt
        if context and atomic_statement:
            system_content = (
                "You are a fact-checking assistant. Generate evidence based on factual "
                "information. Focus on verifiable facts and reliable sources."
            )
            full_context = (
                f"Context from previous statements: {context}\n"
                f"Current statement to verify: {atomic_statement}\n"
                f"Query: {query}"
            )
            gpt_input = prompt.format(context=full_context).strip()
        else:
            system_content = "You are a helpful assistant focused on factual accuracy."
            # If no context/statement is available, format with query only
            gpt_input = prompt.format(query=query).strip()
 
        # Attempt multiple retries in case of transient OpenAI errors
        for attempt in range(num_retries):
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": gpt_input},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                # Extract generated evidence text
                evidence = response.choices[0].message["content"].strip()
                
                if evidence:
                    return {
                        "text": evidence,
                        "query": query,
                        "context": context or "",
                        "statement": atomic_statement or "",
                        "source": "generated",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
 
            except openai.error.OpenAIError as e:
                logger.warning(
                    f"OpenAI API error on attempt {attempt + 1}/{num_retries}: {e}"
                )
                # Exponential backoff
                if attempt < num_retries - 1:
                    time.sleep(2 * (attempt + 1))
 
        # If all retries fail
        logger.error("Failed to generate evidence after all retries.")
        return {
            "text": "",
            "query": query,
            "context": context or "",
            "statement": atomic_statement or "",
            "error": "Failed to generate evidence after all retries"
        }
 
    except Exception as e:
        logger.error(f"Unexpected error in evidence generation: {e}")
        return {
            "text": "",
            "query": query,
            "context": context or "",
            "statement": atomic_statement or "",
            "error": str(e)
        }
 
def batch_generate_evidence(
    questions: List[str],
    model: str,
    prompt: str,
    atomic_statements: Optional[List[str]] = None,
    current_statement_idx: Optional[int] = None
) -> List[Dict[str, str]]:
    """
    Generates evidence for a batch of questions, optionally anchored to a specific
    statement within a list of `atomic_statements`.
    - `atomic_statements`: All statements in the document.
    - `current_statement_idx`: Index of the current statement for context.
    """
    try:
        context = None
        current_statement = None
 
        # Build context from previous statements (if any)
        if atomic_statements and current_statement_idx is not None:
            if current_statement_idx > 0:
                context = " ".join(atomic_statements[:current_statement_idx])
            current_statement = atomic_statements[current_statement_idx]
 
        evidences = []
        for query in questions:
            evidence = run_evidence_hallucination(
                query=query,
                model=model,
                prompt=prompt,
                context=context,
                atomic_statement=current_statement
            )
            evidences.append(evidence)
        
        # Filter out any empty or error-laden evidence responses
        valid_evidences = [e for e in evidences if e.get("text") and not e.get("error")]
        if not valid_evidences:
            logger.warning("No valid evidences generated in batch.")
        
        return valid_evidences
 
    except Exception as e:
        logger.error(f"Error in batch evidence generation: {e}")
        return []
 
 


# """Utils for generating fake evidence given a query."""
# import os
# import time
# import logging
# from typing import Dict, Any
# import openai

# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_openai_client():
#     """Sets up the OpenAI client using API key from environment."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     if hasattr(openai, 'OpenAI'):
#         # For new OpenAI library version
#         return openai.OpenAI(api_key=api_key)
#     else:
#         # For old OpenAI library version
#         openai.api_key = api_key
#         return openai

# # Initialize the OpenAI client
# client = setup_openai_client()

# def run_evidence_hallucination(
#     query: str,
#     model: str,
#     prompt: str,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> Dict[str, str]:
#     """Generates a synthetic piece of evidence via LLM given a query.

#     Args:
#         query: The query to generate evidence for.
#         model: The OpenAI GPT model to use.
#         prompt: The prompt template to structure the query.
#         num_retries: Number of retries if the API call fails.
#         client: OpenAI client instance (optional).
#     Returns:
#         output: A dictionary with hallucinated evidence text and the query.
#     """
#     if client is None:
#         client = setup_openai_client()

#     gpt3_input = prompt.format(query=query).strip()

#     for attempt in range(num_retries):
#         try:
#             # Choose the appropriate client API method based on the library version
#             if hasattr(client, 'chat'):
#                 # New OpenAI client version for chat models
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": gpt3_input}],
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n", "\n\n"],
#                 )
#                 hallucinated_evidence = response.choices[0].message.content.strip()
#             else:
#                 # Old OpenAI client version
#                 response = client.Completion.create(
#                     model=model,
#                     prompt=gpt3_input,
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n", "\n\n"],
#                 )
#                 hallucinated_evidence = response.choices[0].text.strip()

#             # Log successful generation of hallucinated evidence
#             logging.info(f"Generated hallucinated evidence for query: {query[:50]}...")

#             # Return the hallucinated evidence
#             output = {"text": hallucinated_evidence, "query": query}
#             return output

#         except Exception as e:
#             # Log failure and retry
#             logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in 2 seconds...")
#             time.sleep(2)

#     # Log error after all retries have failed
#     logging.error(f"Failed to generate hallucinated evidence after {num_retries} attempts")
#     return {"text": "", "query": query}

# def raise_hallucinate_evidence_warning():
#     """Warns the user that LLM-generated evidence may lead to hallucinations."""
#     logging.warning(
#         "WARNING: LLM-generated evidence may lead to hallucinations. "
#         "This should not be used for improving attribution, as evidence may be "
#         "inaccurate. This is provided only for experimentation without a search API."
#     )

# # Display the warning immediately upon importing this module
# raise_hallucinate_evidence_warning()





# """Utils for generating fake evidence given a query."""
# import os
# import time
# import logging
# from typing import Dict, Any
# import openai

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

# def run_evidence_hallucination(
#     query: str,
#     model: str,
#     prompt: str,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> Dict[str, str]:
#     """Generates a fake piece of evidence via LLM given the question.

#     Args:
#         query: Query to guide the validity check.
#         model: Name of the OpenAI GPT model to use.
#         prompt: The prompt template to query GPT with.
#         num_retries: Number of times to retry OpenAI call in the event of an API failure.
#         client: OpenAI client instance.
#     Returns:
#         output: A potentially inaccurate piece of evidence.
#     """
#     if client is None:
#         client = setup_openai_client()

#     gpt3_input = prompt.format(query=query).strip()
    
#     for attempt in range(num_retries):
#         try:
#             if hasattr(client, 'chat'):
#                 # New OpenAI client
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": gpt3_input}],
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n", "\n\n"],
#                 )
#                 hallucinated_evidence = response.choices[0].message.content.strip()
#             else:
#                 # Old OpenAI client
#                 response = client.Completion.create(
#                     model=model,
#                     prompt=gpt3_input,
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n", "\n\n"],
#                 )
#                 hallucinated_evidence = response.choices[0].text.strip()

#             logging.info(f"Successfully generated hallucinated evidence for query: {query[:50]}...")
#             output = {"text": hallucinated_evidence, "query": query}
#             return output

#         except Exception as e:
#             logging.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying...")
#             time.sleep(2)

#     logging.error(f"Failed to generate hallucinated evidence after {num_retries} attempts")
#     return {"text": "", "query": query}

# def raise_hallucinate_evidence_warning():
#     logging.warning(
#         "WARNING!! YOU ARE USING A LLM TO GENERATE EVIDENCE POTENTIALLY WITH "
#         "HALLUCINATIONS INSTEAD OF RETRIEVING EVIDENCE. \n\nThis should NEVER be "
#         "done when trying to improve attribution as evidence may be inaccurate "
#         "and is only provided to quickly experiment with repository setting up "
#         "the search API first.\n"
#     )

# # Call this function when the module is imported
# raise_hallucinate_evidence_warning()





# """Utils for generating fake evidence given a query."""
# import os
# import time
# from typing import Dict

# import openai

# openai.api_key = os.getenv("OPENAI_API_KEY")


# def run_evidence_hallucination(
#     query: str,
#     model: str,
#     prompt: str,
#     num_retries: int = 5,
# ) -> Dict[str, str]:
#     """Generates a fake piece of evidence via LLM given the question.

#     Args:
#         query: Query to guide the validity check.
#         model: Name of the OpenAI GPT-3 model to use.
#         prompt: The prompt template to query GPT-3 with.
#         num_retries: Number of times to retry OpenAI call in the event of an API failure.
#     Returns:
#         output: A potentially inaccurate piece of evidence.
#     """
#     gpt3_input = prompt.format(query=query).strip()
#     for _ in range(num_retries):
#         try:
#             response = openai.Completion.create(
#                 model=model,
#                 prompt=gpt3_input,
#                 temperature=0.0,
#                 max_tokens=256,
#                 stop=["\n", "\n\n"],
#             )
#             break
#         except openai.error.OpenAIError as exception:
#             print(f"{exception}. Retrying...")
#             time.sleep(2)

#     hallucinated_evidence = response.choices[0].text.strip()
#     output = {"text": hallucinated_evidence, "query": query}
#     return output