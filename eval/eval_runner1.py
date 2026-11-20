from RAG1.agent import answer_question
import json


with open("datasets/eval_dataset.jsonl", "r", encoding="utf-8") as file:
    test_cases = [
        json.loads(line)
        for line in file
    ]
    
    results = []
    
    for test_case in test_cases:
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        print(f"\n Question: {question}")
        
        rag_results = answer_question(question)
        result = {
            "question": question,
            "expected_answer": expected_answer,
            "actual_answer": rag_results.answer,
            "context": rag_results.context
        }
        
        results.append(result)
        print(f"Answer: {rag_results.answer}")

with open("results1.json", "w", encoding="utf-8") as file:
    for result in results:
        file.write(
            json.dumps(result)+"\n"
        )
        
        
print("\nDone!, Results1 saved as results1.json")