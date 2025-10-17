from fpdf import FPDF
import pandas as pd
import os

def convert_checked_csv_to_pdf(csv_file="qa_log.csv", pdf_file="user_updates.pdf"):
    if not os.path.exists(csv_file):
        print(f"❌ File {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)

    # 过滤出 checked 状态的记录
    df_checked = df[df["status"].str.lower() == "checked"]

    if df_checked.empty:
        print("⚠️ No 'checked' records found in CSV.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for index, row in df_checked.iterrows():
        pdf.multi_cell(0, 10, f"Question: {row['question']}")
        pdf.ln(1)
        pdf.multi_cell(0, 10, f"Answer: {row['answer']}")
        pdf.ln(5)

    pdf.output(pdf_file)
    print(f"✅ PDF saved as {pdf_file}")

    os.remove("qa_log.csv")

if __name__ == "__main__":
    convert_checked_csv_to_pdf("qa_log.csv", "user_updates.pdf")

