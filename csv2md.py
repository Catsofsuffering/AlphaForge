import pandas as pd
from openai import OpenAI  # Uncomment this line for actual OpenAI API calls
import os
from data_collection.alphalogger import AlphaLogger

logger= AlphaLogger()
# --- Configuration ---
CSV_FILE = "./out/test_csi500_2020_0/csv_zoo_final.csv"
OUTPUT_MARKDOWN_FILE = "output.md"

# For actual OpenAI API calls, uncomment the line below and set your API key
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")  # It's best practice to use environment variables
OPENAI_MODEL = "deepseek-ai/DeepSeek-V3"

# --- System Prompt for the AI ---
SYSTEM_PROMPT = """
请将以下 Python 代码中定义的运算，与其对应的数学或统计学公式（如果适用）转换为 **LaTeX 格式**。对于滚动（`rolling_ops` 和 `rolling_binary_ops`）和引用（`Ref`）操作，请提供其概念解释，并用文字描述其作用，因为它们通常没有单一的通用公式。

**代码:**

```python
unary_ops = [Inv, S_log1p]
binary_ops = [Add, Sub, Mul, Div, Pow,]
rolling_ops = [Ref, ts_mean, ts_sum, ts_std, ts_var, ts_max, ts_min,ts_med, ts_mad,ts_div, ts_pctchange, ts_delta, ts_wma, ts_ema,]
rolling_binary_ops = [ts_cov, ts_corr]
```

**输出要求:**

* **一元运算 (`unary_ops`):**
    * `Inv`: (倒数) -> LaTeX 公式
    * `S_log1p`: (对数) -> LaTeX 公式
* **二元运算 (`binary_ops`):**
    * `Add`: (加法) -> LaTeX 公式
    * `Sub`: (减法) -> LaTeX 公式
    * `Mul`: (乘法) -> LaTeX 公式
    * `Div`: (除法) -> LaTeX 公式
    * `Pow`: (幂运算) -> LaTeX 公式
* **滚动运算 (`rolling_ops`):**
    * `Ref`: (引用/滞后) -> 文字解释
    * `ts_mean`: (滚动平均值) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_sum`: (滚动求和) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_std`: (滚动标准差) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_var`: (滚动方差) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_max`: (滚动最大值) -> 文字解释（注明 $N$ 为窗口大小）
    * `ts_min`: (滚动最小值) -> 文字解释（注明 $N$ 为窗口大小）
    * `ts_med`: (滚动中位数) -> 文字解释（注明 $N$ 为窗口大小）
    * `ts_mad`: (滚动平均绝对偏差) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_div`: (滚动除法) -> 文字解释（注明可能的场景）
    * `ts_pctchange`: (滚动百分比变化) -> LaTeX 公式（注明 $N$ 为周期）
    * `ts_delta`: (滚动差分) -> LaTeX 公式（注明 $N$ 为周期）
    * `ts_wma`: (滚动加权移动平均) -> LaTeX 公式（注明权重 $w_i$）
    * `ts_ema`: (滚动指数移动平均) -> LaTeX 公式（注明平滑因子 $\alpha$）
* **滚动二元运算 (`rolling_binary_ops`):**
    * `ts_cov`: (滚动协方差) -> LaTeX 公式（注明 $N$ 为窗口大小）
    * `ts_corr`: (滚动相关系数) -> LaTeX 公式（注明 $N$ 为窗口大小）
"""


# --- Function to Convert Expression to LaTeX using AI ---
def convert_expr_to_latex(expression_code: str, client) -> str:
    """
    Uses the AI to convert a code-form expression into LaTeX format or provide an explanation.
    Supports streaming response.
    """
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请将以下运算表达式转换为 LaTeX 格式或进行解释：\n{expression_code}"}
            ],
            # max_tokens=200,
            temperature=0.1,
            stream=True
        )
        
        latex_output = ""
        for chunk in response:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                latex_output += chunk.choices[0].delta.content
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                latex_output += chunk.choices[0].delta.reasoning_content
        return latex_output
    except Exception as e:
        logger.error(f"Error calling AI for '{expression_code}': {e}")
        return f"Conversion failed: `{expression_code}`"


# --- Main Processing Function ---
def process_csv_and_generate_markdown(csv_file: str, output_file: str):
    """
    Reads CSV data line by line, converts 'exprs' column, and saves to a Markdown file.
    Ensures UTF-8 encoding for all file operations.
    """
    try:
        # Read CSV with UTF-8 encoding
        df = pd.read_csv(csv_file, encoding="utf-8")
    except FileNotFoundError:
        logger.error(f"Error: CSV file '{csv_file}' not found. Please ensure the path is correct.")
        return
    except UnicodeDecodeError:
        logger.error(f"Error: Could not decode '{csv_file}' with UTF-8. Try a different encoding if you know it (e.g., 'latin1').")
        return

    # Initialize OpenAI client (or Mock client for testing)
    client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Expression Conversion Results\n\n")

        # Iterate through each row in the DataFrame
        row_counter = 0
        total_rows = len(df)
        for index, row in df.iterrows():
            row_counter += 1
            expr_id = row["id"]
            logger.info(f"Processing row {row_counter}: ID={expr_id}")
            expr_code = str(row["exprs"]).strip()  # Ensure it's a string and remove leading/trailing whitespace
            expr_id = row["id"]
            score = row["scores"]

            # Call AI for conversion
            latex_expr = convert_expr_to_latex(expr_code, client)
            processed_latex = latex_expr.replace("\n", " ")  # Pre-process to remove newlines

            # Write results to Markdown table
            # Extract operation type for heading
            op_type = expr_code.split('(')[0].strip()
            op_name = {
                'Add': '加法运算',
                'Sub': '减法运算',
                'Mul': '乘法运算',
                'Div': '除法运算',
                'Pow': '幂运算',
                'Inv': '倒数运算',
                'S_log1p': '对数运算',
                'Ref': '引用运算',
                'ts_mean': '滚动平均',
                'ts_sum': '滚动求和',
                'ts_std': '滚动标准差',
                'ts_var': '滚动方差',
                'ts_max': '滚动最大值',
                'ts_min': '滚动最小值',
                'ts_med': '滚动中位数',
                'ts_mad': '滚动平均绝对偏差',
                'ts_div': '滚动除法',
                'ts_pctchange': '滚动百分比变化',
                'ts_delta': '滚动差分',
                'ts_wma': '滚动加权平均',
                'ts_ema': '滚动指数平均',
                'ts_cov': '滚动协方差',
                'ts_corr': '滚动相关系数'
            }.get(op_type, op_type)
            
            # Split LaTeX output into formula and explanation
            parts = processed_latex.split('->')
            latex_formula = parts[0].strip() if len(parts) > 0 else ""
            explanation = parts[1].strip() if len(parts) > 1 else processed_latex
            
            f.write(f"## {op_name}\n\n")
            f.write(f"- **公式**: {latex_formula}\n")
            f.write(f"- **解释**: {explanation}\n")
            f.write(f"- **原始表达式**: `{expr_code}`\n")
            f.write(f"- **评分**: {score}\n\n")

    logger.info(f"Processed {row_counter}/{total_rows} rows total")
    logger.info(f"Conversion complete. Results saved to '{output_file}'.")


# --- Main execution block ---
if __name__ == "__main__":
    # Create a sample CSV file if it doesn't exist for easy testing
    if not os.path.exists(CSV_FILE):
        raise
    process_csv_and_generate_markdown(CSV_FILE, OUTPUT_MARKDOWN_FILE)
