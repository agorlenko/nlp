import time
import torch
import tqdm
from nltk.translate.bleu_score import corpus_bleu


def calc_metrics(model, predict_model, get_original_text, get_generated_text, trg_field, test_iterator, n_examples=3):
    start_time = time.time()
    original_text = []
    generated_text = []

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):
            src = batch.src
            trg = batch.trg
            output = predict_model(model, src, trg)
            output = output.argmax(dim=-1)

            original_text.extend(get_original_text(trg_field, trg))
            generated_text.extend(get_generated_text(trg_field, output))

    elapsed = time.time() - start_time
    print(f'elapsed: {elapsed}')
    batch_size = test_iterator.batch_size
    print(f'time for batch with size 32 = {(elapsed / len(test_iterator)) * (32 / batch_size)} s')
    print(f'bleu: {corpus_bleu([[text] for text in original_text], generated_text) * 100}\n')

    examples = []
    seen = set()
    for generated_text_example, original_text_example in zip(generated_text, original_text):
        if tuple(original_text_example) in seen:
            continue
        seen.add(tuple(original_text_example))
        score = corpus_bleu([[original_text_example]], [generated_text_example]) * 100
        examples.append({
            'original_text': original_text_example,
            'generated_text': generated_text_example,
            'score': score
        })
    examples = sorted(examples, key=lambda item: item['score'], reverse=True)
    print(f'best {n_examples} translations:\n')
    for example in examples[:n_examples]:
        print(f"\toriginal: {' '.join(example['original_text'])}")
        print(f"\tgenerated: {' '.join(example['generated_text'])}")
        print(f"\tscore = {example['score']}\n")

    print(f'worst {n_examples} translations:\n')
    for example in examples[-n_examples:]:
        print(f"\toriginal: {' '.join(example['original_text'])}")
        print(f"\tgenerated: {' '.join(example['generated_text'])}")
        print(f"\tscore = {example['score']}\n")

    return examples
