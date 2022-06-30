import torch



def cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, min=10e-8, max=1-10e-8)
    y_pred = y_pred.cuda()
    y_true = y_true.cuda()
    #print("pred",y_pred.max(), y_pred.min())
    #print("true",y_true.max(), y_true.min())
    s_ce_values = - y_true * torch.log(y_pred) - (1. - y_true) * torch.log(1. - y_pred)
    loss = torch.mean(s_ce_values)
    return loss

def weighted_cross_entropy(y_pred, y_true, w1):
    y_pred = torch.clamp(y_pred, min=10e-8, max=1-10e-8)
    loss_weights = 1. + (w1 - 1.) * y_true
    s_ce_values = - y_true * torch.log(y_pred) - (1. - y_true) * torch.log(1. - y_pred)
    loss = torch.mean(s_ce_values * loss_weights)
    return loss


def iou(y_pred, y_true, heatmap_threshold):
    """Measures the mean IoU of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the mean IoU of our predictions.
    """
    pred = torch.gt(y_pred, heatmap_threshold).float()
    intersection = y_true * pred
    union = torch.gt(y_true + pred, 0).float()
    iou_values = torch.sum(intersection, dim=1) / (1e-10 + torch.sum(union, dim=1))
    return torch.mean(iou_values)
    
def iou_no_mean(y_pred, y_true, heatmap_threshold):
    """Measures the mean IoU of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the mean IoU of our predictions.
    """
    pred = torch.gt(y_pred, heatmap_threshold).float()
    intersection = y_true * pred
    union = torch.gt(y_true + pred, 0).float()
    iou_values = torch.sum(intersection, dim=1) / (1e-10 + torch.sum(union, dim=1))
    return iou_values
    
def kl(y_pred, y_true):
     """Measure the KL divergence of our predictions with ground truth.

    Args:
         y_true: The ground truth bounding box locations.
         y_pred: Our heatmap predictions.

    Returns:
         A float containing the KL divergence of our predictions.
     """
     norm_true = y_true / (torch.sum(y_true, dim=1).unsqueeze(dim=1) + 1e-10)
     norm_pred = y_pred / (torch.sum(y_pred, dim=1).unsqueeze(dim=1) + 1e-10)     
     x = torch.log(1e-10 + (norm_true/(1e-10 + norm_pred)))
     return torch.mean(torch.sum((x * norm_true), dim=1))

'''
def precision(y_true, y_pred, heatmap_threshold):
    """Measures the precision of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the precision of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    tp = K.sum(y_true * pred, axis=1)
    fp = K.sum(K.cast(K.greater(pred - y_true, 0), 'float32'), axis=1)
    precision_values = tp/(tp + fp + K.epsilon())
    return K.mean(precision_values)


def recall(y_true, y_pred, heatmap_threshold):
    """Measures the recall of our predictions with ground truth.

    Args:
        y_true: The ground truth bounding box locations.
        y_pred: Our heatmap predictions.
        heatmap_threshold: Config specified theshold above which we consider
          a prediction to contain an object.

    Returns:
        A float containing the recall of our predictions.
    """
    pred = K.cast(K.greater(y_pred, heatmap_threshold), "float32")
    tp = K.sum(y_true * pred, axis=1)
    fn = K.sum(K.cast(K.greater(y_true - pred, 0), 'float32'), axis=1)
    recall_values = tp/(tp + fn + K.epsilon())
    return K.mean(recall_values)


def get_metrics(output_dim, heatmap_threshold):
    """Returns all the metrics with the corresponding thresholds.

    Args:
        output_dim: The size of the predictions.
        heatmap_threshold: The heatmap thresholds we are evaluating with.

    Returns:
        A list of metric functions that take in the ground truth and the
        predictins to evaluate the model.
    """
    metrics = []
    for metric_func in [iou, recall, precision]:
        for thresh in heatmap_threshold:
            metric = (lambda f, t: lambda gt, pred: f(gt, pred, t))(
                metric_func, thresh)
            metric.__name__ = metric_func.__name__ + '_' + str(thresh)
            metrics.append(metric)
    metrics += [kl, cc, sim]
    return metrics
'''
