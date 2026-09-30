from sentence_transformers import SentenceTransformer
import faiss
import os
import ollama


# --------------------------------
# 1. Load embedding model
# --------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------
# 2. Load documents
# --------------------------------

documents = []
sources = []

folder = "documents"

for filename in os.listdir(folder):

    if filename.endswith(".txt"):

        path = os.path.join(folder, filename)

        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        # Split into chunks
        words = text.split()

        chunk_size = 100

        for i in range(0, len(words), chunk_size):

            chunk = " ".join(words[i:i + chunk_size])

            documents.append(chunk)
            sources.append(filename)


# --------------------------------
# 3. Convert documents to embeddings
# --------------------------------

embeddings = model.encode(documents)

print("Number of chunks:", len(documents))


# --------------------------------
# 4. Create FAISS index
# --------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)


# --------------------------------
# 5. Take a claim
# --------------------------------

claim = input("\nEnter a claim: ")


# --------------------------------
# 6. Convert claim to embedding
# --------------------------------

claim_embedding = model.encode([claim])


# --------------------------------
# 7. Search for relevant evidence
# --------------------------------

k = min(3, len(documents))

distances, results = index.search(
    claim_embedding,
    k
)


# --------------------------------
# 8. Relevance threshold
# --------------------------------

threshold = 2.0


# --------------------------------
# 9. Collect relevant evidence
# --------------------------------

print("\nRetrieved Evidence:\n")

found_evidence = False
evidence_text = ""

for position, i in enumerate(results[0]):

    distance = distances[0][position]

    if distance <= threshold:

        found_evidence = True

        print("--------------------------------")
        print("Evidence", position + 1)
        print("Source:", sources[i])
        print("Distance:", distance)
        print("Text:")
        print(documents[i])

        evidence_text += (
            "\nSource: " + sources[i] +
            "\n" + documents[i] + "\n"
        )


# --------------------------------
# 10. Generate explanation using Qwen3
# --------------------------------

if found_evidence:

    prompt = f"""
You are an evidence verification assistant.

Claim:
{claim}

Retrieved evidence:
{evidence_text}

Based ONLY on the retrieved evidence, analyze the claim.

Explain whether the evidence supports, contradicts, or is insufficient
to verify the claim.

Do not use outside knowledge.
"""

    print("\n--------------------------------")
    print("Qwen3 Analysis:")
    print("--------------------------------")

    response = ollama.chat(
        model="qwen3:4b-q4_K_M",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print(response["message"]["content"])


else:

    print("No sufficiently relevant evidence found.")