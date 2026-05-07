# test/test_data_creation.py
import json
import os

def create_comprehensive_test_dataset():
    """Create a comprehensive test dataset for Digital Citizen Charter chatbot"""
    
    test_dataset = {
        "language_detection_cases": [
            {"text": "What services are available?", "language": "english"},
            {"text": "के सेवाहरू उपलब्ध छन्?", "language": "nepali"},
            {"text": "How to register for voting?", "language": "english"},
            {"text": "मतदानको लागि कसरी दर्ता गर्ने?", "language": "nepali"},
            {"text": "Digital services portal", "language": "english"},
            {"text": "डिजिटल सेवा पोर्टल", "language": "nepali"},
        ],
        
        "qa_cases": [
            {
                "question": "What is the Digital Citizen Charter?",
                "expected_language": "english"
            },
            {
                "question": "डिजिटल नागरिक घोषणापत्र भनेको के हो?",
                "expected_language": "nepali"
            },
            {
                "question": "How can I complain about government services?",
                "expected_language": "english"
            },
            {
                "question": "सरकारी सेवाको बारेमा कसरी गुनासो गर्ने?",
                "expected_language": "nepali"
            }
        ],
        
        "retrieval_cases": [
            {
                "question": "citizenship requirements",
                "expected_topics": ["citizenship", "requirements"]
            },
            {
                "question": "नागरिकताको आवश्यकता",
                "expected_topics": ["नागरिकता", "आवश्यकता"]
            }
        ],
        
        "translation_cases": [
            {
                "source_text": "Government services are available online",
                "expected_translation": "सरकारी सेवाहरू अनलाइन उपलब्ध छन्",
                "direction": "en_to_np"
            },
            {
                "source_text": "Apply for digital services",
                "expected_translation": "डिजिटल सेवाको लागि आवेदन गर्नुहोस्",
                "direction": "en_to_np"
            },
            {
                "source_text": "नागरिकताको लागि आवेदन गर्नुहोस्",
                "expected_translation": "Apply for citizenship",
                "direction": "np_to_en"
            }
        ]
    }
    
    # Save test dataset
    output_path = os.path.join(os.path.dirname(__file__), 'test_dataset.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"Test dataset created: {output_path}")
    
    return test_dataset

if __name__ == "__main__":
    create_comprehensive_test_dataset()