import itertools
import logging
import tqdm
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from sentence_transformers import CrossEncoder
 
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
 
class EvidenceSelector:
    """
    A class to handle evidence selection using a cross-encoder for relevance scoring.
    Capable of both atomic (statement-wise) and non-atomic evidence selection.
    """
    def __init__(self):
        """Initialize the EvidenceSelector with a CrossEncoder passage ranker."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.passage_ranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                max_length=512,
                device=self.device
            )
            logger.info(f"Initialized passage ranker on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder: {e}")
            self.passage_ranker = None
 
    def compute_score_matrix(
        self,
        questions: List[str],
        evidences: List[str],
        context: Optional[str] = None,
        statement: Optional[str] = None,
        batch_size: int = 16,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Compute a score matrix measuring how each (question + context/statement) pair
        aligns with each evidence text. Returns a (len(questions) x len(evidences)) matrix.
        """
        if not self.passage_ranker:
            logger.error("Passage ranker not initialized.")
            return np.array([])
 
        if not questions or not evidences:
            logger.warning("Empty questions or evidences.")
            return np.array([])
 
        try:
            # Build contextualized questions
            contextualized_questions = []
            for q in questions:
                parts = []
                if context:
                    parts.append(f"Context: {context}")
                if statement:
                    parts.append(f"Statement: {statement}")
                parts.append(f"Question: {q}")
                contextualized_questions.append(" ".join(parts))
 
            # Prepare (question, evidence) pairs
            pairs = [(q, e) for q in contextualized_questions for e in evidences]
 
            # Break into batches
            batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
 
            all_scores = []
            # Optionally display progress with tqdm
            batch_iterator = tqdm.tqdm(batches, desc="Computing scores") if show_progress else batches
 
            for batch in batch_iterator:
                with torch.no_grad():
                    scores = self.passage_ranker.predict(batch)
                    all_scores.extend(scores)
 
            # Reshape scores into a matrix
            score_matrix = np.array(all_scores).reshape(len(questions), len(evidences))
            return self._normalize_scores(score_matrix)
 
        except Exception as e:
            logger.error(f"Error computing scores: {e}")
            return np.array([])
 
    def _normalize_scores(self, matrix: np.ndarray) -> np.ndarray:
        """
        Normalizes each row to the [0, 1] range. If min == max, uses a default
        fallback (0.5 if max > 0, else 0.0) for the entire row.
        """
        try:
            normalized = np.zeros_like(matrix)
            for i, row in enumerate(matrix):
                min_val, max_val = row.min(), row.max()
                if max_val > min_val:
                    normalized[i] = (row - min_val) / (max_val - min_val)
                else:
                    # Fallback for uniform or zero row
                    normalized[i] = np.ones_like(row) * (0.5 if max_val > 0 else 0.0)
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing scores: {e}")
            return matrix
 
    def _filter_irrelevant_evidence(
        self,
        evidences: List[Dict[str, Any]],
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filters out evidence items whose 'score' is below `min_score`.
        """
        return [e for e in evidences if e.get('score', 0) >= min_score]
 
    def select_evidences_for_atomic_statement(
        self,
        questions: List[str],
        evidences: List[str],
        context: Optional[str] = None,
        statement: Optional[str] = None,
        max_selected: int = 5,
        min_score_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Select evidences relevant to a single atomic statement.
        Each evidence must meet the `min_score_threshold`.
        Returns a list of evidences (dicts) sorted by descending score.
        """
        try:
            if not questions or not evidences:
                return []
 
            # Compute a (questions x evidences) relevance matrix
            score_matrix = self.compute_score_matrix(
                questions,
                evidences,
                context=context,
                statement=statement
            )
            if score_matrix.size == 0:
                return []
 
            # For each evidence (column), take the max score across all questions (rows)
            max_scores = np.max(score_matrix, axis=0)
 
            evidence_info = []
            for i, evidence in enumerate(evidences):
                if max_scores[i] >= min_score_threshold:
                    # Gather question-level relevance for any question above threshold
                    relevant_qs = [
                        {
                            'question': questions[q],
                            'relevance_score': float(score_matrix[q, i])
                        }
                        for q in np.where(score_matrix[:, i] >= min_score_threshold)[0]
                    ]
                    if relevant_qs:
                        evidence_info.append({
                            'text': evidence,
                            'score': float(max_scores[i]),
                            'relevant_questions': relevant_qs
                        })
 
            # Sort by descending overall score
            evidence_info.sort(key=lambda x: x['score'], reverse=True)
 
            # Return the top `max_selected`
            return evidence_info[:max_selected]
 
        except Exception as e:
            logger.error(f"Error in select_evidences_for_atomic_statement: {e}")
            return []
 
    def process_atomic_statements(
        self,
        statements: List[str],
        questions_per_statement: List[List[str]],
        evidences: List[str],
        max_evidences_per_statement: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Iterate over each statement, fetch relevant evidences for it, and maintain
        a running context of previously processed statements.
        """
        results = []
        context = ""
 
        for i, (statement, questions) in enumerate(zip(statements, questions_per_statement)):
            logger.info(f"Processing statement {i + 1}/{len(statements)}")
 
            # For each statement, build context from all previously processed statements
            selected_evidences = self.select_evidences_for_atomic_statement(
                questions=questions,
                evidences=evidences,
                context=context if i > 0 else None,
                statement=statement,
                max_selected=max_evidences_per_statement
            )
            
            results.append({
                "statement": statement,
                "questions": questions,
                "evidences": selected_evidences,
                "context_used": bool(context)
            })
            
            # Update context to include current statement as well
            context = " ".join(statements[:i + 1])
 
        return results
 
    def select_evidences(
        self,
        example: Dict[str, Any],
        max_selected: int = 5,
        atomic_processing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Main public method to select evidences from an `example`. If `atomic_processing` is True
        and `example` contains `atomic_statements` and `questions_per_statement`, it processes them
        statement-by-statement. Otherwise, falls back to a simpler (non-atomic) approach.
        """
        try:
            if atomic_processing and "atomic_statements" in example:
                statements = example["atomic_statements"]
                questions_per_statement = example.get("questions_per_statement", [])
                evidences = self._extract_evidences(example)
                
                if not evidences:
                    logger.warning("No evidences found in example.")
                    return []
                
                return self.process_atomic_statements(
                    statements=statements,
                    questions_per_statement=questions_per_statement,
                    evidences=evidences,
                    max_evidences_per_statement=max_selected
                )
            else:
                # Non-atomic fallback
                logger.info("Using non-atomic processing mode.")
                questions = sorted(set(example.get("questions", [])))
                evidences = self._extract_evidences(example)
                
                if not questions or not evidences:
                    return []
 
                # Compute scores in a single pass
                score_matrix = self.compute_score_matrix(questions, evidences)
                if score_matrix.size == 0:
                    return []
 
                # For each evidence, take the max score across questions, then pick top `max_selected`
                max_scores = np.max(score_matrix, axis=0)
                best_indices = np.argsort(-max_scores)[:max_selected]
                return [
                    {
                        "text": evidences[i],
                        "score": float(max_scores[i])
                    }
                    for i in best_indices
                ]
 
        except Exception as e:
            logger.error(f"Error in select_evidences: {e}")
            return []
 
    def _extract_evidences(self, example: Dict[str, Any]) -> List[str]:
        """
        Helper to retrieve evidence strings from the 'revisions' field of an example.
        Ensures each piece of evidence is non-empty and of sufficient length.
        """
        try:
            revisions = example.get("revisions", [])
            if not revisions or "evidences" not in revisions[0]:
                return []
            
            # Gather all distinct evidence texts
            evidences = []
            for e in revisions[0]["evidences"]:
                if isinstance(e, dict) and e.get("text"):
                    text = e["text"].strip()
                    # A basic check to exclude extremely short or invalid entries
                    if text and len(text) > 10:
                        evidences.append(text)
            
            # Return unique sorted evidences
            return sorted(set(evidences))
 
        except Exception as e:
            logger.error(f"Error extracting evidences: {e}")
            return []
 
 
if __name__ == "__main__":
    # Example usage with Lena Headey claims
    selector = EvidenceSelector()
 
    sample_statements = [
        "Lena Headey portrayed Cersei Lannister in HBO's Game of Thrones since 2011.",
        "She received three consecutive Emmy nominations for Outstanding Supporting Actress.",
        "By 2017, she became one of the highest-paid television actors."
    ]
 
    sample_questions = [
        [
            "When did Lena Headey start playing Cersei Lannister?",
            "Was she the original actress cast for the role?",
            "How long did she play the character?"
        ],
        [
            "Which years did she receive Emmy nominations?",
            "Were the nominations consecutive?",
            "What was the exact category for her nominations?"
        ],
        [
            "What was her reported salary by 2017?",
            "How did her salary compare to other TV actors?",
            "What factors contributed to her salary increase?"
        ]
    ]
 
    # Example usage of evidence selection
    test_example = {
        "atomic_statements": sample_statements,
        "questions_per_statement": sample_questions,
        "revisions": [{
            "evidences": [{"text": "Sample evidence " + str(i)} for i in range(5)]
        }]
    }
 
    results = selector.select_evidences(test_example, atomic_processing=True)
    
    print("\nResults per Statement:")
    for result in results:
        print(f"\nStatement: {result['statement']}")
        print("Selected Evidences:")
        for evidence in result.get('evidences', []):
            print(f"- Evidence: {evidence['text']}")
            score_str = f"{evidence.get('score', 0.0):.3f}"
            print(f"  Score: {score_str}")
            if 'relevant_questions' in evidence:
                print("  Relevant Questions:")
                for q_info in evidence['relevant_questions']:
                    print(f"    * {q_info['question']} (score: {q_info['relevance_score']:.3f})")
 
 





# import itertools
# import logging
# from typing import Any, Dict, List

# import torch
# from sentence_transformers import CrossEncoder

# # Configure logging with detailed information for tracking complexity
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # Initialize the passage ranker model
# PASSAGE_RANKER = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     max_length=512,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# )

# def compute_score_matrix(
#     questions: List[str], 
#     evidences: List[str]
# ) -> List[List[float]]:
#     """Enhanced scoring for evidence relevance"""
#     score_matrix = []
#     for q in questions:
#         # Normalize and enhance scores
#         evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences])
#         normalized_scores = (evidence_scores - evidence_scores.min()) / (evidence_scores.max() - evidence_scores.min())
#         score_matrix.append(normalized_scores.tolist())
#     return score_matrix

# # def compute_score_matrix(
# #     questions: List[str], evidences: List[str]
# # ) -> List[List[float]]:
# #     """Scores the relevance of all evidence against all questions using a CrossEncoder.

# #     Args:
# #         questions: A list of unique questions.
# #         evidences: A list of unique evidences.
# #     Returns:
# #         score_matrix: A 2D list of question X evidence relevance scores.
# #     """
# #     score_matrix = []
# #     try:
# #         for q in questions:
# #             # Use the CrossEncoder model to predict relevance scores
# #             evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
# #             logging.info(f"Scores for question '{q}': {evidence_scores}")  # Log the scores
# #             score_matrix.append(evidence_scores)
# #         return score_matrix
# #     except Exception as e:
# #         logging.error(f"Error in compute_score_matrix: {str(e)}")
# #         return []

# def question_coverage_objective_fn(
#     score_matrix: List[List[float]], evidence_indices: List[int]
# ) -> float:
#     """Calculate coverage score for the selected evidence subset based on question relevance.

#     Args:
#         score_matrix: 2D list of question X evidence relevance scores.
#         evidence_indices: Indices of the evidence subset for coverage scoring.
#     Returns:
#         total: The combined coverage score for the subset.
#     """
#     try:
#         total = sum(max(scores_for_question[j] for j in evidence_indices) for scores_for_question in score_matrix)
#         return total
#     except Exception as e:
#         logging.error(f"Error in question_coverage_objective_fn: {str(e)}")
#         return 0.0

# def select_evidences(
#     example: Dict[str, Any], max_selected: int = 5, prefer_fewer: bool = False
# ) -> List[Dict[str, Any]]:
#     """Selects the optimal subset of evidence that maximizes question coverage.

#     Args:
#         example: The result of running the editing pipeline on one claim.
#         max_selected: Maximum number of evidences to select.
#         prefer_fewer: If True, selects fewer evidence if it reaches maximum coverage.
#     Returns:
#         selected_evidences: The best subset of evidence.
#     """
#     try:
#         # Retrieve unique questions and evidence texts
#         questions = sorted(set(example["questions"]))
#         evidences = sorted(set(e["text"] for e in example["revisions"][0]["evidences"]))
#         num_evidences = len(evidences)

#         if not num_evidences:
#             logging.warning("No evidences found. Returning empty list.")
#             return []

#         # Compute score matrix and validate
#         score_matrix = compute_score_matrix(questions, evidences)
#         if not score_matrix:
#             logging.warning("Failed to compute score matrix. Returning empty list.")
#             return []

#         # Adjust the maximum and minimum selections
#         max_selected = min(max_selected, num_evidences)
#         min_selected = 1 if prefer_fewer else max_selected

#         # Track best combination and score
#         best_combo, best_objective_value = (), float("-inf")

#         # Iteratively find the best combination of evidence indices
#         for num_selected in range(min_selected, max_selected + 1):
#             for combo in itertools.combinations(range(num_evidences), num_selected):
#                 objective_value = question_coverage_objective_fn(score_matrix, combo)
#                 if objective_value > best_objective_value:
#                     best_combo, best_objective_value = combo, objective_value

#         # Prepare final selection based on best combination
#         selected_evidences = [{"text": evidences[idx]} for idx in best_combo]
#         logging.info(f"Selected {len(selected_evidences)} evidences out of {num_evidences}, with a score of {best_objective_value}")
#         return selected_evidences

#     except Exception as e:
#         logging.error(f"Error in select_evidences: {str(e)}")
#         return []





# """Utils for selecting the most relevant evidences."""
# import itertools
# import logging
# from typing import Any, Dict, List

# import torch
# from sentence_transformers import CrossEncoder

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PASSAGE_RANKER = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     max_length=512,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# )

# def compute_score_matrix(
#     questions: List[str], evidences: List[str]
# ) -> List[List[float]]:
#     """Scores the relevance of all evidence against all questions using a CrossEncoder.

#     Args:
#         questions: A list of unique questions.
#         evidences: A list of unique evidences.
#     Returns:
#         score_matrix: A 2D list of question X evidence relevance scores.
#     """
#     score_matrix = []
#     try:
#         for q in questions:
#             evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
#             score_matrix.append(evidence_scores)
#         return score_matrix
#     except Exception as e:
#         logging.error(f"Error in compute_score_matrix: {str(e)}")
#         return []

# def question_coverage_objective_fn(
#     score_matrix: List[List[float]], evidence_indices: List[int]
# ) -> float:
#     """Given (query, evidence) scores and a subset of evidence, return the coverage.

#     Given all pairwise query and evidence scores, and a subset of the evidence
#     specified by indices, return a value indicating how well this subset of evidence
#     covers (i.e., helps answer) all questions.

#     Args:
#         score_matrix: A 2D list of question X evidence relevance scores.
#         evidence_indices: A subset of the evidence to get the coverage score of.
#     Returns:
#         total: The coverage we would get by using the subset of evidence in
#             `evidence_indices` over all questions.
#     """
#     # Compute sum_{question q} max_{selected evidence e} score(q, e).
#     # This encourages all questions to be explained by at least one evidence.
#     try:
#         total = 0.0
#         for scores_for_question in score_matrix:
#             total += max(scores_for_question[j] for j in evidence_indices)
#         return total
#     except Exception as e:
#         logging.error(f"Error in question_coverage_objective_fn: {str(e)}")
#         return 0.0

# def select_evidences(
#     example: Dict[str, Any], max_selected: int = 5, prefer_fewer: bool = False
# ) -> List[Dict[str, Any]]:
#     """Selects the set of evidence that maximizes information coverage over the claim.

#     Args:
#         example: The result of running the editing pipeline on one claim.
#         max_selected: Maximum number of evidences to select.
#         prefer_fewer: If True and the maximum objective value can be achieved by
#             fewer evidences than `max_selected`, prefer selecting fewer evidences.
#     Returns:
#         selected_evidences: Selected evidences that serve as the attribution report.
#     """
#     try:
#         questions = sorted(set(example["questions"]))
#         evidences = sorted(set(e["text"] for e in example["revisions"][0]["evidences"]))
#         num_evidences = len(evidences)
        
#         if not num_evidences:
#             logging.warning("No evidences found. Returning empty list.")
#             return []

#         score_matrix = compute_score_matrix(questions, evidences)
        
#         if not score_matrix:
#             logging.warning("Failed to compute score matrix. Returning empty list.")
#             return []

#         best_combo = tuple()
#         best_objective_value = float("-inf")
#         max_selected = min(max_selected, num_evidences)
#         min_selected = 1 if prefer_fewer else max_selected

#         for num_selected in range(min_selected, max_selected + 1):
#             for combo in itertools.combinations(range(num_evidences), num_selected):
#                 objective_value = question_coverage_objective_fn(score_matrix, combo)
#                 if objective_value > best_objective_value:
#                     best_combo = combo
#                     best_objective_value = objective_value

#         selected_evidences = [{"text": evidences[idx]} for idx in best_combo]
#         logging.info(f"Selected {len(selected_evidences)} evidences out of {num_evidences}")
#         return selected_evidences

#     except Exception as e:
#         logging.error(f"Error in select_evidences: {str(e)}")
#         return []
    


# import itertools
# from typing import Any, Dict, List

# import torch
# from sentence_transformers import CrossEncoder

# PASSAGE_RANKER = CrossEncoder(
#     "cross-encoder/ms-marco-MiniLM-L-6-v2",
#     max_length=512,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
# )


# def compute_score_matrix(
#     questions: List[str], evidences: List[str]
# ) -> List[List[float]]:
#     """Scores the relevance of all evidence against all questions using a CrossEncoder.

#     Args:
#         questions: A list of unique questions.
#         evidences: A list of unique evidences.
#     Returns:
#         score_matrix: A 2D list list of question X evidence relevance scores.
#     """
#     score_matrix = []
#     for q in questions:
#         evidence_scores = PASSAGE_RANKER.predict([(q, e) for e in evidences]).tolist()
#         score_matrix.append(evidence_scores)
#     return score_matrix


# def question_coverage_objective_fn(
#     score_matrix: List[List[float]], evidence_indices: List[int]
# ) -> float:
#     """Given (query, evidence) scores and a subset of evidence, return the coverage.

#     Given all pairwise query and evidence scores, and a subset of the evidence
#     specified by indices, return a value indicating how well this subset of evidence
#     covers (i.e., helps answer) all questions.

#     Args:
#         score_matrix: A 2D list list of question X evidence relevance scores.
#         evidence_indicies: A subset of the evidence to to get the coverage score of.
#     Returns:
#         total: The coverage we would get by using the subset of evidence in
#             `evidence_indices` over all questions.
#     """
#     # Compute sum_{question q} max_{selected evidence e} score(q, e).
#     # This encourages all questions to be explained by at least one evidence.
#     total = 0.0
#     for scores_for_question in score_matrix:
#         total += max(scores_for_question[j] for j in evidence_indices)
#     return total


# def select_evidences(
#     example: Dict[str, Any], max_selected: int = 5, prefer_fewer: bool = False
# ) -> List[Dict[str, Any]]:
#     """Selects the set of evidence that maximizes information converage over the claim.

#     Args:
#         example: The result of running the editing pipeline on one claim.
#         max_selected: Maximum number of evidences to select.
#         prefer_fewer: If True and the maximum objective value can be achieved by
#             fewer evidences than `max_selected`, prefer selecting fewer evidences.
#     Returns:
#         selected_evidences: Selected evidences that serve as the attribution report.
#     """
#     questions = sorted(set(example["questions"]))
#     evidences = sorted(set(e["text"] for e in example["revisions"][0]["evidences"]))
#     num_evidences = len(evidences)
#     if not num_evidences:
#         return []

#     score_matrix = compute_score_matrix(questions, evidences)

#     best_combo = tuple()
#     best_objective_value = float("-inf")
#     max_selected = min(max_selected, num_evidences)
#     min_selected = 1 if prefer_fewer else max_selected
#     for num_selected in range(min_selected, max_selected + 1):
#         for combo in itertools.combinations(range(num_evidences), num_selected):
#             objective_value = question_coverage_objective_fn(score_matrix, combo)
#             if objective_value > best_objective_value:
#                 best_combo = combo
#                 best_objective_value = objective_value

#     selected_evidences = [{"text": evidences[idx]} for idx in best_combo]
#     return selected_evidences