import pandas as pd
import os

def load_data():
    stage_a = pd.read_csv('outputs/stage_a_results_summary.csv')
    stage_b = pd.read_csv('outputs_stage_b/stage_b_results_summary.csv')
    
    stage_a['Stage'] = 'Stage A (Unconstrained/Baseline)'
    stage_b['Stage'] = 'Stage B (Constrained/Advanced)'
    
    # Combine data
    df = pd.concat([stage_a, stage_b], ignore_index=True)
    return df

def generate_markdown(df):
    subsets = ['EA1', 'EA7', 'EA14', 'Full']
    
    md_content = ["# Comprehensive Experimental Results\n"]
    md_content.append("This document contains exhaustive tabular results from all model evaluations across Stage A and Stage B.\n")
    md_content.append("Metrics included:\n- **RMSE**: Root Mean Squared Error (Lower is better)\n- **MAE**: Mean Absolute Error (Lower is better)\n- **R²**: Coefficient of Determination (Higher is better, max 1.0)\n\n")
    
    for subset in subsets:
        md_content.append(f"## Subset: {subset}")
        md_content.append(f"*{'1-Day' if subset == 'EA1' else '7-Day' if subset == 'EA7' else '14-Day' if subset == 'EA14' else '1-365 Days'} Compressive Strength*\n")
        
        subset_df = df[df['Subset'] == subset].copy()
        subset_df = subset_df.sort_values(by=['Stage', 'R2_mean'], ascending=[True, False])
        
        table = "| Stage | Model | RMSE (Mean ± Std) | MAE (Mean ± Std) | R² (Mean ± Std) |\n"
        table += "| :--- | :--- | :--- | :--- | :--- |\n"
        
        for _, row in subset_df.iterrows():
            rmse_str = f"{row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}"
            mae_str = f"{row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}"
            r2_str = f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}"
            
            # Make the best model within each stage bold? 
            # We will just list them.
            table += f"| {row['Stage']} | {row['Model']} | {rmse_str} | {mae_str} | {r2_str} |\n"
            
        md_content.append(table)
        md_content.append("\n")
        
    return "".join(md_content)

def main():
    df = load_data()
    md_text = generate_markdown(df)
    
    # Write to artifact directory
    out_path = r'C:\Users\MRaza\.gemini\antigravity-ide\brain\e0252ed4-7b83-41fe-bdf9-0cb109a7ee4c\all_experimental_results.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md_text)
    print(f"Generated comprehensive report at {out_path}")

if __name__ == '__main__':
    main()
