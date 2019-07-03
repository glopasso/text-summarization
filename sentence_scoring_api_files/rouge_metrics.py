import rouge


def rouge_scoring(hypothesis, reference, max_n=1, alpha=0.5, score='F1'):
    evaluator = rouge.Rouge(metrics=['rouge-n'],
                            max_n=max_n,
                            limit_length=False,
                            alpha=alpha,  # Default F1_score
                            stemming=True)
    if score == 'F1':
        score_entry = 'f'
    elif score == 'Precision':
        score_entry = 'p'
    else:
        score == 'Recall'
        score_entry = 'r'

    rouge_score = evaluator.get_scores(hypothesis, reference)['rouge-' + str(max_n)][score_entry]

    return rouge_score