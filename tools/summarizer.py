from smolagents import tool

@tool
def summarize_news(text: str) -> str:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "VietAI/vit5-base-vietnews-summarization"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

    formatted_text = "vietnews: " + text + " </s>"
    encoding = tokenizer(formatted_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            max_length=256,
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


@tool
def classify_topic(text: str, topic: str) -> bool:
    from transformers import pipeline
    import torch

    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "zero-shot-classification",
        model="vicgalle/xlm-roberta-large-xnli-anli",
        device=device,
        trust_remote_code=True
    )

    candidate_labels = [topic, f"không liên quan {topic}"]
    result = classifier(text, candidate_labels)
    return result["labels"][0] == topic
