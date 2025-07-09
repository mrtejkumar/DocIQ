from transformers import pipeline

classifier = pipeline("zero-shot-classification")

# sequence = "When I say you're my life, I hope you know that it's true, I love you with all my heart, I love you with my soul."
sequence = "What makes a nation’s pillars high, And its foundations strong? What makes it mighty to defy, The foes that round it throng? "
labels = ["patriotism", "love", "anger", "sadness"]

result = classifier(sequence, candidate_labels=labels)
print(result)
