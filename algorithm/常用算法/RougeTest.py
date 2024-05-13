
import jieba
from rouge import Rouge
#ROUGE-L是基于最长公共子序列（LCS）的重叠度量，
#它计算候选摘要和参考摘要之间最长的公共单词序列的长度和比例。
#LCS可以捕捉到单词顺序的信息，但不要求完全匹配；


def preprocess_chinese_text(text):
    # 使用jieba进行中文分词
    tokens = jieba.cut(text)
    return " ".join(tokens)

def preprocess_text2(text):
    return " ".join(text)

def score_rouge(hypothesis,reference):
    hypothesis=preprocess_text2(hypothesis)
    reference=preprocess_text2(reference)
    print(hypothesis)
    print(reference)
    rouger = Rouge()
    scores = rouger.get_scores(hypothesis, reference)
    print(scores)
    return  max(scores[0]['rouge-l']['r'],  scores[0]['rouge-l']['p'])

if __name__ == '__main__':
    print(score_rouge("上海寻梦信息技术2","上海寻梦信息技术"))
    #
    # system_summary = "This is a sample summary produced by the system."
    # reference_summary = "This is a reference summary written by a human."
    #
    # # Create a Rouge object
    # rouge = Rouge()
    #
    # # Compute Rouge scores
    # scores = rouge.get_scores(system_summary, reference_summary)
    #
    # # Print the scores
    # print(scores)