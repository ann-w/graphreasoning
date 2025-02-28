import csv


def save_response_to_csv(csv_path, question_text, response):
    """Save the question and response to a CSV file."""
    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["question", "answer"])
        writer.writerow([question_text, response])