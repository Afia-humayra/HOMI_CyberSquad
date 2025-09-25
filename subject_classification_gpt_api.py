import openai
import subprocess

# Set your API key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Define mapping from categories to games
game_mapping = {
    "math": "python3 math_game.py",
    "word": "python3 word_game.py",
    "memory": "python3 memory_game.py"
}

def classify_sentence(sentence):
    # Ask GPT to classify
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # lightweight and fast for classification
        messages=[
            {"role": "system", "content": "You are a classifier. Classify the input into: math, word, memory."},
            {"role": "user", "content": sentence}
        ],
        max_tokens=10
    )
    category = response["choices"][0]["message"]["content"].strip().lower()
    return category

def launch_game(category):
    if category in game_mapping:
        print(f"Launching {category} game...")
        subprocess.run(game_mapping[category], shell=True)
    else:
        print("No matching game found for this category.")

if __name__ == "__main__":
    sentence = input("Enter a sentence to classify: ")
    category = classify_sentence(sentence)
    print(f"GPT classified this as: {category}")
    launch_game(category)