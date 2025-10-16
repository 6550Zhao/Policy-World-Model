import math
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
def calc_l2(plan, gt):
    l2 = [0.] * 6
    for i, p in enumerate(plan):
        l2[i] += math.sqrt((p[0] - gt[i][0])**2 + (p[1] - gt[i][1])**2)
    return l2

def eval_qa(preds, refs, metric='rouge'):
    # To avoid download config from huggingface, download the metric config on github
    # and load it locally
    # evaluator = evaluate.load('rouge.py')

    # for Cider, simply use
    # evaluator = Cider()
    
    if metric == 'cider':
        predictions, references = {}, {}
        for i, p in enumerate(predictions):
            predictions[i] = p
            references[i] = refs[i]
        evaluator = Cider()
        return {'cider': evaluator.compute_score(references, predictions)[0]}
    
    evaluator = evaluate.load(metric)
    # TODO: lower batch size
    # TODO: support more metrics
    return evaluator.compute(predictions=preds, references=refs)

class textScorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))

if __name__ == '__main__':
    ref = {
        '1': ["The vehicle is positioned on a clear, well-marked urban road during what appears to be a bright, sunny day. To the left, the road is bordered by a parking structure and a gated area with a white car parked in front of it, while on the right, the road is lined with buildings and a sidewalk. Directly ahead, the road is open and traffic is sparse, with only a few cars visible in the distance. The road markings suggest a multi-lane street with no immediate turns or merges. The surrounding environment is a mix of commercial and residential buildings, with no pedestrians or significant obstructions in the immediate vicinity of the vehicle.."],
        '2': ["The vehicle is positioned at a bustling urban intersection, surrounded by the activity of construction and city life. To the left, a group of pedestrians is crossing the street, navigating between the stationary cars waiting at the traffic light. Directly in front, the intersection is marked by freshly painted white lines, suggesting recent roadwork, and a green traffic light indicates the flow of traffic is permitted."],
        '3': [
            "The vehicle is positioned at a bustling urban intersection, surrounded by the activity of construction and city life. To the left, a group of pedestrians is crossing the street, navigating between the stationary cars waiting at the traffic light. Directly in front, the intersection is marked by freshly painted white lines, suggesting recent roadwork, and a green traffic light indicates the flow of traffic is permitted."]

    }
    # ref = {
    #     '2': ['Walk down the steps and stop at the bottom. '],
    #     '1': ['It is a cat.']
    # }
    gt = {
        '1': ["The driving scene unfolds in a wide street during what appears to be a clear day with ample sunlight. In the left-front, a large, industrial building with a flat roof houses several vehicles parked alongside. A few pedestrians are visible, with one closer to the curb, suggesting proximity to pedestrian crossings or walkways. The central-front view reveals a straight, well-maintained road with clear lane markings, and no immediate signs of heavy traffic or obstructions. To the right-front, there's a partially visible construction zone denoted by a fence, with a large white truck and construction equipment."],
        '2': ["Navigating an urban road, the vehicle is positioned at an intersection. To the left-front, a pedestrian crossing is visible, complete with traffic signals. The middle-front view reveals a straightforward road, flanked by contemporary architecture and a walkway bustling with pedestrians, signaling a mix of residential and commercial areas. The right-front perspective shows another crossing filled with waiting people, suggesting it is a busy locale."],
        '3': [
            "The vehicle is positioned at a bustling urban intersection, surrounded by the activity of construction and city life. To the left, a group of pedestrians is crossing the street, navigating between the stationary cars waiting at the traffic light. Directly in front, the intersection is marked by freshly painted white lines, suggesting recent roadwork, and a green traffic light indicates the flow of traffic is permitted."]

    }
    scorer = textScorer(ref, gt)
    scorer.compute_scores()

    