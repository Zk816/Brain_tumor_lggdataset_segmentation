def IoU_metric_from_dice(dice: float) -> float:
    
    assert 0 <= dice <= 1, "Dice coefficient must be in the range [0, 1]"
    iou = dice / (2 - dice)
    return iou
