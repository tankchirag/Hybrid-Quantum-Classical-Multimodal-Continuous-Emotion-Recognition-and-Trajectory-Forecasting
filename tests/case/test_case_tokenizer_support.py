from models.tokenizers.tokenizer import (
    Tokenizer,
)

t = Tokenizer()

print()
print("SUPPORTED TOKENS")
print("-" * 40)

for token in t.supported_tokens:
    print(token)