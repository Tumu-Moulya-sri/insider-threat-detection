import pandas as pd
import random
import os

def generate_survey(input_path="data/escalated.csv", output_path="data/human_evaluation_survey.md"):
    if not os.path.exists(input_path):
        print(f"File {input_path} not found. Please run the pipeline first.")
        return
        
    df = pd.read_csv(input_path)
    
    # Filter for interesting alerts (Alert or Log)
    alerts = df[df["escalation_decision"].isin(["Alert", "Log"])]
    
    if len(alerts) == 0:
        print("No alerts found in the dataset to survey.")
        return
        
    # Sample up to 10 alerts
    sample_size = min(10, len(alerts))
    sampled = alerts.sample(n=sample_size, random_state=42)
    
    markdown_lines = [
        "# UEBA Evidence Chain - Human Evaluation Survey",
        "",
        "**Instructions for Analyst:**",
        "Please read the provided natural-language evidence explanation for each alert.",
        "Rate each explanation on a scale of 1 to 5 based on how well it helps you decide what action to take (e.g., how interpretable, actionable, and transparent it is).",
        "",
        "**Scale:**",
        "1 - Completely unhelpful / confusing",
        "2 - Provides minimal context, still requires heavy manual investigation",
        "3 - Somewhat helpful, gives a general idea of the anomaly",
        "4 - Very helpful, isolates the issue clearly",
        "5 - Perfect, gives me exactly what I need to make a confident decision",
        "",
        "---",
        ""
    ]
    
    for idx, (i, row) in enumerate(sampled.iterrows(), 1):
        markdown_lines.append(f"### Alert {idx}")
        markdown_lines.append(f"**Action Recommended by System:** {row['escalation_decision']}")
        markdown_lines.append(f"**Predicted Behavioral Stage:** {row['predicted_stage']}")
        markdown_lines.append(f"**Evidence Chain Explanation:**")
        markdown_lines.append(f"> {row['evidence_chain_explanation']}")
        markdown_lines.append("")
        markdown_lines.append("**Your Rating (1-5):** [   ]")
        markdown_lines.append("**Comments (optional):**")
        markdown_lines.append("________________________________________________________")
        markdown_lines.append("________________________________________________________")
        markdown_lines.append("")
        markdown_lines.append("---")
        markdown_lines.append("")
        
    with open(output_path, "w") as f:
        f.write("\n".join(markdown_lines))
        
    print(f"Generated human evaluation survey at {output_path}")

if __name__ == "__main__":
    generate_survey()
