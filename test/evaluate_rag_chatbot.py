import os
import sys
import json
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Tuple, Any
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import re

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Now import your chatbot
from rag_chatbot import (
    detect_language, 
    rag_answer, 
    retriever, 
    translate_text,
    NEPALI_PATTERN
)

class RAGChatbotEvaluator:
    def __init__(self):
        self.metrics = {}
        self.NEPALI_PATTERN = NEPALI_PATTERN
    
    def detect_language_eval(self, text: str) -> str:
        """Language detection for evaluation"""
        return detect_language(text)
    
    def evaluate_language_detection(self, test_cases: List[Dict]) -> Dict:
        """Evaluate language detection accuracy"""
        print("Evaluating language detection...")
        
        predictions = []
        true_labels = []
        
        for case in test_cases:
            text = case['text']
            true_lang = case['language']
            pred_lang = self.detect_language_eval(text)
            
            predictions.append(pred_lang)
            true_labels.append(true_lang)
        
        accuracy = accuracy_score(true_labels, predictions)
        
        # Language-wise metrics
        lang_metrics = {}
        for lang in ['english', 'nepali']:
            lang_true = [1 if l == lang else 0 for l in true_labels]
            lang_pred = [1 if l == lang else 0 for l in predictions]
            precision, recall, f1, _ = precision_recall_fscore_support(
                lang_true, lang_pred, average='binary', zero_division=0
            )
            lang_metrics[lang] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        self.metrics['language_detection'] = {
            'overall_accuracy': accuracy,
            'language_wise_metrics': lang_metrics,
            'confusion_matrix': pd.crosstab(
                pd.Series(true_labels), 
                pd.Series(predictions), 
                rownames=['Actual'], 
                colnames=['Predicted']
            ).to_dict()
        }
        
        return self.metrics['language_detection']
    
    def evaluate_response_quality(self, qa_test_cases: List[Dict]) -> Dict:
        """Evaluate response quality and language consistency"""
        print("Evaluating response quality...")
        
        results = []
        
        for case in qa_test_cases:
            question = case['question']
            expected_lang = case.get('expected_language', self.detect_language_eval(question))
            
            start_time = time.time()
            answer, detected_lang = rag_answer(question)
            response_time = time.time() - start_time
            
            # Language consistency check
            answer_lang = self.detect_language_eval(answer)
            lang_consistent = (answer_lang == expected_lang)
            
            result = {
                'question': question,
                'answer': answer,
                'expected_language': expected_lang,
                'detected_language': detected_lang,
                'answer_language': answer_lang,
                'language_consistent': lang_consistent,
                'response_time': response_time,
                'has_content': len(answer.strip()) > 0,
                'answer_length': len(answer)
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Calculate metrics
        language_consistency = df['language_consistent'].mean()
        avg_response_time = df['response_time'].mean()
        success_rate = df['has_content'].mean()
        avg_answer_length = df['answer_length'].mean()
        
        # Language-specific metrics
        lang_specific = {}
        for lang in ['english', 'nepali']:
            lang_df = df[df['expected_language'] == lang]
            if len(lang_df) > 0:
                lang_specific[lang] = {
                    'consistency': lang_df['language_consistent'].mean(),
                    'avg_response_time': lang_df['response_time'].mean(),
                    'avg_answer_length': lang_df['answer_length'].mean()
                }
        
        self.metrics['response_quality'] = {
            'language_consistency_rate': language_consistency,
            'average_response_time_seconds': avg_response_time,
            'success_rate': success_rate,
            'average_answer_length': avg_answer_length,
            'language_specific_metrics': lang_specific,
            'detailed_results': results
        }
        
        return self.metrics['response_quality']
    
    def evaluate_retrieval_quality(self, retrieval_test_cases: List[Dict]) -> Dict:
        """Evaluate retrieval component performance"""
        print("Evaluating retrieval quality...")
        
        results = []
        
        for case in retrieval_test_cases:
            question = case['question']
            expected_topics = case.get('expected_topics', [])
            
            # Test retrieval
            docs = retriever.invoke(question)
            retrieved_content = [doc.page_content for doc in docs]
            
            # Simple relevance check
            relevance_score = self._calculate_relevance_score(question, retrieved_content)
            
            result = {
                'question': question,
                'num_documents_retrieved': len(docs),
                'relevance_score': relevance_score,
                'documents_retrieved': len(retrieved_content) > 0
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        
        self.metrics['retrieval_quality'] = {
            'avg_documents_retrieved': df['num_documents_retrieved'].mean(),
            'avg_relevance_score': df['relevance_score'].mean(),
            'retrieval_success_rate': df['documents_retrieved'].mean(),
            'detailed_results': results
        }
        
        return self.metrics['retrieval_quality']
    
    def _calculate_relevance_score(self, question: str, documents: List[str]) -> float:
        """Calculate simple relevance score based on keyword matching"""
        question_keywords = set(question.lower().split())
        total_score = 0
        
        for doc in documents:
            doc_keywords = set(doc.lower().split())
            common_keywords = question_keywords.intersection(doc_keywords)
            score = len(common_keywords) / len(question_keywords) if question_keywords else 0
            total_score += score
        
        return total_score / len(documents) if documents else 0
    
    def evaluate_translation_quality(self, translation_test_cases: List[Dict]) -> Dict:
        """Evaluate translation accuracy"""
        print("Evaluating translation quality...")
        
        results = []
        
        for case in translation_test_cases:
            source_text = case['source_text']
            expected_translation = case['expected_translation']
            direction = case['direction']  # 'en_to_np' or 'np_to_en'
            
            try:
                if direction == 'en_to_np':
                    translated = translate_text(source_text, "nepali")
                else:
                    translated = translate_text(source_text, "english")
                
                # Simple similarity metric
                similarity = self._calculate_text_similarity(translated, expected_translation)
                
                result = {
                    'source_text': source_text,
                    'translated_text': translated,
                    'expected_translation': expected_translation,
                    'similarity_score': similarity,
                    'direction': direction
                }
                results.append(result)
                
            except Exception as e:
                print(f"Translation error for '{source_text}': {e}")
                continue
        
        df = pd.DataFrame(results)
        
        translation_metrics = {}
        for direction in ['en_to_np', 'np_to_en']:
            dir_df = df[df['direction'] == direction]
            if len(dir_df) > 0:
                translation_metrics[direction] = {
                    'avg_similarity_score': dir_df['similarity_score'].mean(),
                    'num_tests': len(dir_df)
                }
        
        self.metrics['translation_quality'] = {
            'overall_avg_similarity': df['similarity_score'].mean(),
            'direction_metrics': translation_metrics,
            'detailed_results': results
        }
        
        return self.metrics['translation_quality']
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def run_comprehensive_evaluation(self, test_datasets: Dict) -> Dict:
        """Run all evaluation components"""
        print("Starting comprehensive evaluation...")
        
        # Language detection evaluation
        if 'language_detection_cases' in test_datasets:
            self.evaluate_language_detection(test_datasets['language_detection_cases'])
        
        # Response quality evaluation
        if 'qa_cases' in test_datasets:
            self.evaluate_response_quality(test_datasets['qa_cases'])
        
        # Retrieval quality evaluation
        if 'retrieval_cases' in test_datasets:
            self.evaluate_retrieval_quality(test_datasets['retrieval_cases'])
        
        # Translation quality evaluation
        if 'translation_cases' in test_datasets:
            self.evaluate_translation_quality(test_datasets['translation_cases'])
        
        # Calculate overall score
        self._calculate_overall_score()
        
        return self.metrics
    
    def _calculate_overall_score(self):
        """Calculate overall performance score"""
        weights = {
            'language_detection': 0.2,
            'response_quality': 0.4,
            'retrieval_quality': 0.3,
            'translation_quality': 0.1
        }
        
        overall_score = 0
        components_used = 0
        
        for component, weight in weights.items():
            if component in self.metrics:
                if component == 'language_detection':
                    score = self.metrics[component]['overall_accuracy']
                elif component == 'response_quality':
                    score = self.metrics[component]['language_consistency_rate']
                elif component == 'retrieval_quality':
                    score = self.metrics[component]['avg_relevance_score']
                elif component == 'translation_quality':
                    score = self.metrics[component]['overall_avg_similarity']
                
                overall_score += score * weight
                components_used += weight
        
        self.metrics['overall_performance'] = {
            'overall_score': overall_score,
            'max_possible_score': components_used,
            'normalized_score': overall_score / components_used if components_used > 0 else 0
        }
    
    def generate_report(self, output_file: str = None):
        """Generate evaluation report"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': self.metrics
        }
        
        if output_file:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Print summary
        self._print_summary()
        
        return report
    
    def _print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        if 'overall_performance' in self.metrics:
            overall = self.metrics['overall_performance']
            print(f"Overall Score: {overall['normalized_score']:.2%}")
        
        if 'language_detection' in self.metrics:
            ld = self.metrics['language_detection']
            print(f"Language Detection Accuracy: {ld['overall_accuracy']:.2%}")
        
        if 'response_quality' in self.metrics:
            rq = self.metrics['response_quality']
            print(f"Language Consistency: {rq['language_consistency_rate']:.2%}")
            print(f"Average Response Time: {rq['average_response_time_seconds']:.2f}s")
        
        if 'retrieval_quality' in self.metrics:
            rq = self.metrics['retrieval_quality']
            print(f"Retrieval Success Rate: {rq['retrieval_success_rate']:.2%}")
            print(f"Average Relevance Score: {rq['avg_relevance_score']:.2f}")
        
        print("="*50)

# Sample test data
def create_sample_test_data():
    """Create sample test data for evaluation"""
    
    test_data = {
        'language_detection_cases': [
            {'text': 'What are the government services?', 'language': 'english'},
            {'text': 'सरकारी सेवाहरू के के हुन्?', 'language': 'nepali'},
            {'text': 'How to apply for citizenship?', 'language': 'english'},
            {'text': 'नागरिकताको लागि कसरी आवेदन गर्ने?', 'language': 'nepali'},
            {'text': 'This is a mixed text with नेपाली characters', 'language': 'nepali'},
        ],
        
        'qa_cases': [
            {'question': 'What is digital citizen charter?', 'expected_language': 'english'},
            {'question': 'डिजिटल नागरिक घोषणापत्र भनेको के हो?', 'expected_language': 'nepali'},
            {'question': 'How to get birth certificate?', 'expected_language': 'english'},
            {'question': 'जन्म दर्ता कागजात कसरी प्राप्त गर्ने?', 'expected_language': 'nepali'},
        ],
        
        'retrieval_cases': [
            {'question': 'government services', 'expected_topics': ['services', 'government']},
            {'question': 'citizenship application', 'expected_topics': ['citizenship', 'application']},
        ],
        
        'translation_cases': [
            {
                'source_text': 'Government services are available online',
                'expected_translation': 'सरकारी सेवाहरू अनलाइन उपलब्ध छन्',
                'direction': 'en_to_np'
            },
            {
                'source_text': 'नागरिकताको लागि आवेदन गर्नुहोस्',
                'expected_translation': 'Apply for citizenship',
                'direction': 'np_to_en'
            }
        ]
    }
    
    return test_data

# Main execution
if __name__ == "__main__":
    try:
        # Initialize evaluator
        evaluator = RAGChatbotEvaluator()
        
        # Load test data
        test_data = create_sample_test_data()
        
        # Run comprehensive evaluation
        print("Starting RAG Chatbot Evaluation...")
        metrics = evaluator.run_comprehensive_evaluation(test_data)
        
        # Generate report in test directory
        report_path = os.path.join(os.path.dirname(__file__), 'evaluation_report.json')
        report = evaluator.generate_report(report_path)
        
        print(f"\nEvaluation completed! Check {report_path} for detailed results.")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()