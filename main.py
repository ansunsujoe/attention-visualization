from attention.model import load_model_from_file
from attention.analyzer import AttentionAnalyzer


def main():
    model, tokenizer = load_model_from_file("models/qwen3-0.6b")
    analyzer = AttentionAnalyzer(model=model, tokenizer=tokenizer)
    analyzer.analyze(
        prompt="What is the tallest mountain in the world?",
        system_prompt="You are an experienced moutaineer.",
        max_tokens=100
    )


if __name__ == "__main__":
    main()
