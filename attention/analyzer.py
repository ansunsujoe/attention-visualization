from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import string


def group_tokens_to_words(tokens, scores):
    words = []
    word_scores = []

    current_word = ""
    current_score = 0.0

    for token, score in zip(tokens, scores):
        is_new = (
            token.startswith("▁")
            or token.startswith("Ġ")
            or token.startswith("Ċ")
            or token.startswith("<|")
        )
        clean = token.lstrip("▁Ġ").lstrip("Ċ")

        if is_new and current_word:
            if current_word == "<|im_start|>system":
                words.append("&lt;system&gt;")
                word_scores.append(0)
            elif current_word == "<|im_end|>":
                words.append("&lt;end&gt;")
                word_scores.append(0)
            elif current_word == "<|im_start|>assistant":
                words.append("&lt;assistant&gt;")
                word_scores.append(0)
            elif current_word == "<|im_start|>user":
                words.append("&lt;user&gt;")
                word_scores.append(0)
            else:
                words.append(current_word)
                word_scores.append(current_score)
            current_word = clean
            current_score = score.item()
        else:
            current_word += clean
            current_score += score.item()

    if current_word:
        words.append(current_word)
        word_scores.append(current_score)
    return words, word_scores


def is_punctuation(word):
    return all(c in string.punctuation for c in word)


class AttentionAnalyzer:
    def __init__(
        self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def get_formatted_prompt(self, prompt: str, system_prompt: str) -> None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return formatted_prompt

    def render_attention_html(
        self, words, scores, min_alpha=0.05, max_alpha=0.6
    ):
        html_parts = []

        for word, score in zip(words, scores):
            alpha = min_alpha + (max_alpha - min_alpha) * score
            alpha = min(alpha, max_alpha)

            span = f"""
        <span style="
            position: relative;
            background-color: {"rgba(30, 144, 255, " + str(alpha) + ")" if not word.startswith("&lt;") else "black"};
            color: {"black" if not word.startswith("&lt;") else "white"};
            border-radius: 4px;
            padding: 2px 4px;
            margin: 1px;
            display: inline-block;
            font-family: sans-serif;
            font-weight: {500 if score < 0.1 else 600};
        " title="attention: {score:.6f}">
            {word}
            <span style="
                position: absolute;
                top: -1.5em;
                left: 0.2em;
                font-size: 0.54em;
                color: {"rgba(30, 144, 255, 0.85)" if not word.startswith("&lt;") else "white"};
                pointer-events: none;
                font-weight: {500 if score < 0.1 else 700};
            ">
                {score:.3f}
            </span>
        </span>
        """
            html_parts.append(span)

        return " ".join(html_parts)

    def analyze(
        self, prompt: str, system_prompt: str, max_tokens: int = 10
    ) -> None:
        device = self._model.device

        # Tokenize prompt.
        formatted_prompt = self.get_formatted_prompt(
            prompt=prompt, system_prompt=system_prompt
        )
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(
            device
        )
        input_ids = inputs["input_ids"]

        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()
        generated_token_texts = []
        all_token_attentions = []

        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self._model(
                    input_ids=generated_ids,
                    output_attentions=True,
                    return_dict=True,
                )

            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            token_text = self._tokenizer.decode(
                next_token_id[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_token_texts.append(token_text)

            # Attention extraction from last layer.
            attentions = outputs.attentions
            last_layer_attn = attentions[-1]
            token_attn = last_layer_attn[:, :, -1, :]

            # Take only the parts that are relevant to the prompt.
            prompt_attn = token_attn[:, :, :prompt_len]
            prompt_attn_mean = prompt_attn.mean(dim=1).squeeze(0)
            all_token_attentions.append(prompt_attn_mean.cpu())
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        # Convert full prompt tokens.
        tokens = self._tokenizer.convert_ids_to_tokens(input_ids[0])

        html_outputs = []
        for step, attn in enumerate(all_token_attentions):
            generated_token = generated_token_texts[step].strip().replace("<", "&lt;").replace(">", "&gt;")
            if generated_token == "":
                continue

            words, scores = group_tokens_to_words(tokens, attn)

            total = sum(scores)
            scores = [s / total if total > 0 else 0.0 for s in scores]

            html = self.render_attention_html(words, scores)
            html_outputs.append(
                f"""
                <div style="margin-bottom: 24px;">
                    <h4>{generated_token}</h4>
                    <p style="line-height: 1.6; font-family: sans-serif;">
                        {html}
                    </p>
                </div>
                """
            )
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8">
        <title>Attention Visualization</title>
        </head>
        <body style="padding: 24px;">
        <h2>Prompt Attention Attribution</h2>
        {"".join(html_outputs)}
        </body>
        </html>
        """

        with open("attention.html", "w") as f:
            f.write(full_html)
