import pandas as pd
from eval import REG_Evaluator

def evaluate_csv_direct(csv_path, embedding_model='dmis-lab/biobert-v1.1', spacy_model='en_core_sci_lg'):
    # Load CSV directly
    df = pd.read_csv(csv_path)
    
    eval_pairs = list(zip(df["Ground Truths"], df["Generated Reports"]))
    
    # Run REG evaluation
    evaluator = REG_Evaluator(embedding_model=embedding_model, spacy_model=spacy_model)
    score = evaluator.evaluate_dummy(eval_pairs)
    
    # Save result
    results_df = pd.DataFrame([{
        "Average Ranking Score": score
    }])
    results_df.to_csv("results.csv", index=False)
    
    print(f"REG Score: {score:.4f} (n={len(eval_pairs)})")
    return score

csv_path = '/home/woody/iwi5/iwi5204h/HistGen/MyExperiments/REG2025_Format/seed43/TITAN/17/gen_vs_gt.csv'

score = evaluate_csv_direct(csv_path)
