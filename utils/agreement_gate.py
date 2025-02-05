"""Agreement gate for fact verification and revision."""
import os
import time
import json
import logging
from typing import Dict, Optional, Tuple, Any
 
import openai
import torch
from sentence_transformers import CrossEncoder
from openai import OpenAIError  # Import the specific error class you need

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
# Set the OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
 
# Initialize a CrossEncoder model to compute relevance scores between text pairs
RELEVANCE_SCORER = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
 
def compute_relevance_score(query: str, evidence: str) -> float:
    """
    Computes a relevance score for (query, evidence) using the cross-encoder.
    Higher scores => more relevant.
    """
    try:
        with torch.no_grad():
            score = RELEVANCE_SCORER.predict([(query, evidence)])
        return float(score[0])
    except Exception as e:
        logger.error(f"Error computing relevance: {e}")
        return 0.0
 
def format_prompt(
    claim: str,
    query: str,
    evidence: str,
    context: Optional[str] = None,
    relevance_threshold: float = 0.1
) -> Tuple[str, bool]:
    """
    Builds the base prompt for analyzing whether 'claim' needs revision given 'evidence'.
    Also checks if evidence meets the `relevance_threshold`.
    Returns:
      - A textual prompt
      - A boolean indicating whether the evidence is relevant enough to consider.
    """
    # Compute relevance
    relevance = compute_relevance_score(query, evidence)
    logger.info(f"Evidence relevance score: {relevance:.2f}")
 
    is_relevant = (relevance >= relevance_threshold)
 
    # If not relevant, we don't include the context or evidence in the final prompt
    # (or set them to blank).
    if not is_relevant:
        logger.info(f"Evidence ignored (low relevance: {relevance:.2f})")
        context = None
 
    # Build the basic instruction
    base_prompt = (
        "Task: Determine if the evidence contradicts or requires changes to the claim.\n"
        "Format: JSON with the following fields:\n"
        "- needs_revision: true/false\n"
        "- explanation: Short reason for or against revision\n"
        "- revised_claim: the updated text if revision is needed, else the same claim\n\n"
        f"Claim: {claim}\n"
        f"Query: {query}\n"
        f"Evidence: {evidence}\n"
    )
 
    if context:
        base_prompt = f"Context: {context}\n{base_prompt}"
 
    return base_prompt, is_relevant
 
def run_agreement_gate(
    claim: str,
    query: str,
    evidence: str,
    context: Optional[str] = None,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.1,
    max_retries: int = 3,
    relevance_threshold: float = 0.1,
    prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main function to determine if a claim needs revision based on the evidence.
    Steps:
      1. Format a JSON-based system prompt.
      2. Parse whether "needs_revision" is True/False.
      3. If True, we run a second revision prompt to finalize the updated claim.
      4. Return a dict capturing the final status of the revision process.
    """
    try:
        # 1) Create a base prompt using `format_prompt`
        formatted_prompt, is_relevant = format_prompt(
            claim=claim,
            query=query,
            evidence=evidence,
            context=context,
            relevance_threshold=relevance_threshold
        )
 
        # If the user provided an additional prompt, prepend it
        if prompt:
            formatted_prompt = f"{prompt}\n\n{formatted_prompt}"
 
        # 2) Attempt multiple times in case of transient OpenAI errors
        for attempt in range(max_retries):
            try:
                # The system message sets the role/perspective
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a fact-checking assistant analyzing evidence versus claims. "
                                "Return JSON as instructed."
                            )
                        },
                        {
                            "role": "user",
                            "content": formatted_prompt
                        }
                    ],
                    temperature=temperature,
                )
                result_raw = response.choices[0].message.content.strip()
                logger.debug(f"Agreement gate raw response:\n{result_raw}")
 
                # 3) Parse the JSON
                try:
                    result = json.loads(result_raw)
                except json.JSONDecodeError:
                    # If we can't parse JSON, assume no revision needed
                    logger.warning("Could not parse JSON from the response.")
                    result = {
                        "needs_revision": False,
                        "explanation": "No valid JSON returned",
                        "revised_claim": claim
                    }
 
                needs_revision = bool(result.get("needs_revision", False))
                explanation = result.get("explanation", "")
                revised_claim_in_result = result.get("revised_claim", "").strip()
 
                final_revised_claim = claim  # default to original
                if needs_revision:
                    # The model's answer might already have a "revised_claim",
                    # but we can refine with a second revision prompt:
                    revision_prompt = (
                        "You are a fact-checking assistant that revises claims based on evidence.\n"
                        f"Original claim: {claim}\n"
                        f"Evidence: {evidence}\n"
                        f"Reason for revision: {explanation}\n\n"
                        "Provide a concise, accurate revised claim."
                    )
                    revision_response = openai.ChatCompletion.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a fact-checking assistant."},
                            {"role": "user", "content": revision_prompt},
                        ],
                        temperature=temperature
                    )
                    second_pass_claim = revision_response.choices[0].message.content.strip()
                    final_revised_claim = second_pass_claim or revised_claim_in_result or claim
 
                # Compute a final relevance score for the revised claim
                revision_score = compute_relevance_score(final_revised_claim, evidence)
 
                return {
                    "is_open": needs_revision,
                    "original_claim": claim,
                    "revised_claim": final_revised_claim,
                    "explanation": explanation,
                    "context_used": is_relevant,
                    "evidence_used": evidence,
                    "revision_score": revision_score
                }
 
            except OpenAIError as e:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
 
        raise Exception(f"Failed after {max_retries} attempts.")
 
    except Exception as e:
        logger.error(f"Agreement gate error: {e}")
        # If everything fails, return a fallback
        return {
            "is_open": False,
            "original_claim": claim,
            "revised_claim": claim,
            "explanation": str(e),
            "context_used": False,
            "evidence_used": evidence,
            "revision_score": 0.0
        }
 
if __name__ == "__main__":
    # Quick test
    test_claim = "The Wright brothers made their first flight in 1902."
    test_query = "When did the Wright brothers first achieve powered flight?"
    test_evidence = "They achieved their first powered, sustained flight on December 17, 1903."
 
    result = run_agreement_gate(
        claim=test_claim,
        query=test_query,
        evidence=test_evidence,
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_retries=2,
        relevance_threshold=0.1
    )
 
    print("\n--- Agreement Gate Test ---")
    print(f"Original claim: {result['original_claim']}")
    print(f"Gate status: {'open' if result['is_open'] else 'closed'}")
    print(f"Revised claim: {result['revised_claim']}")
    print(f"Explanation: {result['explanation']}")
    print(f"Evidence used: {result['evidence_used']}")
    print(f"Final revision score: {result['revision_score']:.3f}")


# import os
# import time
# import logging
# from typing import Any, Dict, Tuple
# import openai

# # Configure logging for more detailed output and tracking computation
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def setup_openai_client():
#     """Sets up the OpenAI client with API key validation."""
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY environment variable is not set")

#     if hasattr(openai, 'OpenAI'):
#         return openai.OpenAI(api_key=api_key)
#     else:
#         openai.api_key = api_key
#         return openai

# # Initialize the OpenAI client
# client = setup_openai_client()

# def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
#     """Parses the API response to check if the agreement gate should be open.

#     Args:
#         api_response: Agreement gate response from GPT-3.
#     Returns:
#         is_open: Boolean indicating if the agreement gate is open.
#         reason: Reasoning for the gate's state.
#         decision: Final decision on the gate status in string form.
#     """
#     api_response = api_response.strip().split("\n")
#     if len(api_response) < 2:
#         reason, decision, is_open = "Failed to parse.", None, False
#     else:
#         reason = api_response[0]
#         decision = api_response[1].split("Therefore:")[-1].strip()
#         is_open = "disagrees" in api_response[1]
#     return is_open, reason, decision

# def run_agreement_gate(
#     claim: str,
#     query: str,
#     evidence: str,
#     model: str,
#     prompt: str,
#     context: str = None,
#     num_retries: int = 5,
#     client: Any = None,
# ) -> Dict[str, Any]:
#     """Checks if the provided evidence contradicts the claim given a query.

#     Args:
#         claim: The claim text to validate.
#         query: The query guiding the validity check.
#         evidence: Evidence text to validate the claim against.
#         model: OpenAI model to use for the agreement gate.
#         prompt: Prompt template for the GPT query.
#         context: Optional context for claim refinement.
#         num_retries: Max number of retries in case of API failure.
#         client: OpenAI client instance.
#     Returns:
#         gate: Dictionary containing the gate's state and reasoning.
#     """
#     if client is None:
#         client = setup_openai_client()

#     gpt3_input = prompt.format(
#         context=context or "", claim=claim, query=query, evidence=evidence
#     ).strip()
    
#     total_time, attempts = 0, 0  # To track total time spent and attempts

#     for attempt in range(1, num_retries + 1):
#         attempts += 1
#         start_time = time.time()

#         try:
#             if hasattr(client, 'chat'):
#                 response = client.chat.completions.create(
#                     model=model,
#                     messages=[{"role": "user", "content": gpt3_input}],
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n\n"],
#                 )
#                 api_response = response.choices[0].message.content
#             else:
#                 response = client.Completion.create(
#                     model=model,
#                     prompt=gpt3_input,
#                     temperature=0.0,
#                     max_tokens=256,
#                     stop=["\n\n"],
#                     logit_bias={"50256": -100},
#                 )
#                 api_response = response.choices[0].text

#             end_time = time.time()
#             total_time += end_time - start_time

#             # Parse the API response
#             is_open, reason, decision = parse_api_response(api_response)
#             gate = {"is_open": is_open, "reason": reason, "decision": decision}
            
#             # Logging details for complexity analysis
#             logging.info(f"Attempt {attempt} completed in {end_time - start_time:.2f} seconds.")
#             return gate

#         except Exception as e:
#             end_time = time.time()
#             total_time += end_time - start_time
#             logging.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in 2 seconds...")
#             time.sleep(2)
    
#     # Log the total time and attempts if retries fail
#     logging.error(f"Failed to get a response after {num_retries} attempts, total time: {total_time:.2f} seconds.")
#     return {"is_open": False, "reason": "API call failed", "decision": None}

# # """Utils for running the agreement gate."""
# # import os
# # import time
# # from typing import Any, Dict, Tuple

# # import openai

# # openai.api_key = os.getenv("OPENAI_API_KEY")


# # def parse_api_response(api_response: str) -> Tuple[bool, str, str]:
# #     """Extract the agreement gate state and the reasoning from the GPT-3 API response.

# #     Our prompt returns questions as a string with the format of an ordered list.
# #     This function parses this response in a list of questions.

# #     Args:
# #         api_response: Agreement gate response from GPT-3.
# #     Returns:
# #         is_open: Whether the agreement gate is open.
# #         reason: The reasoning for why the agreement gate is open or closed.
# #         decision: The decision of the status of the gate in string form.
# #     """
# #     api_response = api_response.strip().split("\n")
# #     if len(api_response) < 2:
# #         reason = "Failed to parse."
# #         decision = None
# #         is_open = False
# #     else:
# #         reason = api_response[0]
# #         decision = api_response[1].split("Therefore:")[-1].strip()
# #         is_open = "disagrees" in api_response[1]
# #     return is_open, reason, decision


# # def run_agreement_gate(
# #     claim: str,
# #     query: str,
# #     evidence: str,
# #     model: str,
# #     prompt: str,
# #     context: str = None,
# #     num_retries: int = 5,
# # ) -> Dict[str, Any]:
# #     """Checks if a provided evidence contradicts the claim given a query.

# #     Checks if the answer to a query using the claim contradicts the answer using the
# #     evidence. If so, we open the agreement gate, which means that we allow the editor
# #     to edit the claim. Otherwise the agreement gate is closed.

# #     Args:
# #         claim: Text to check the validity of.
# #         query: Query to guide the validity check.
# #         evidence: Evidence to judge the validity of the claim against.
# #         model: Name of the OpenAI GPT-3 model to use.
# #         prompt: The prompt template to query GPT-3 with.
# #         num_retries: Number of times to retry OpenAI call in the event of an API failure.
# #     Returns:
# #         gate: A dictionary with the status of the gate and reasoning for decision.
# #     """
# #     if context:
# #         gpt3_input = prompt.format(
# #             context=context, claim=claim, query=query, evidence=evidence
# #         ).strip()
# #     else:
# #         gpt3_input = prompt.format(claim=claim, query=query, evidence=evidence).strip()

# #     for _ in range(num_retries):
# #         try:
# #             response = openai.Completion.create(
# #                 model=model,
# #                 prompt=gpt3_input,
# #                 temperature=0.0,
# #                 max_tokens=256,
# #                 stop=["\n\n"],
# #                 logit_bias={"50256": -100},  # Don't allow <|endoftext|> to be generated
# #             )
# #             break
# #         except openai.error.OpenAIError as exception:
# #             print(f"{exception}. Retrying...")
# #             time.sleep(2)

# #     is_open, reason, decision = parse_api_response(response.choices[0].text)
# #     gate = {"is_open": is_open, "reason": reason, "decision": decision}
# #     return gate