import torch


def calculate_accuracy(predictions, labels):
    """
    Calculate the accuracy of the predictions
    :param predictions: The predictions
    :param labels: The labels
    :return: The accuracy
    """
    
    predictions = torch.argmax(predictions, dim=-1)
    correct = sum(predictions == labels)
    return correct / len(predictions)


def calculate_top_N_accuracy(predictions, labels, N=3):
    """
    Calculate the top N accuracy of the predictions
    :param predictions: The predictions
    :param labels: The labels
    :param N: The top N
    :return: The top N accuracy
    """
    
    top_N = torch.topk(predictions, N, dim=-1)[1]
    correct = sum([label in top_N[i] for i, label in enumerate(labels)])
    return correct / len(predictions)

if __name__ == "__main__":
    a = torch.randn(10, 5)
    b = torch.randn(10, 5)
    
    r = a @ b.t()
    
    acc = calculate_accuracy(r, torch.arange(0, r.shape[0]))
    print(acc)
    acc = calculate_top_N_accuracy(r, torch.arange(0, r.shape[0]), N=5)
    print(acc)