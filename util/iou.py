import torch

def bbox_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes using the box corner coordinates
    :param box1: (tensor) bounding boxes, sized [N,4].
    :param box2: (tensor) bounding boxes, sized [M,4].
    :return: (tensor) IoU, sized [N,M].
    """

    N = box1.size(0)
    M = box2.size(0)

    # Calculate the left-upper and the right-lower coordinates of the intersection rectangle
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2), # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2)
    )

    # Calculate the area of intersection rectangle
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # Calculate area of both boxes
    area1 = (box1[:, 2]-box1[:, 0]) * (box1[:, 3]-box1[:, 1])  # [N,]
    area2 = (box2[:, 2]-box2[:, 0]) * (box2[:, 3]-box2[:, 1])  # [M,]

    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    # Calculate IoU
    iou = inter / (area1 + area2 - inter)

    return iou



