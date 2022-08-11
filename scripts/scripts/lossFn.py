
import torch
import torch.nn as nn

class EASTLoss:
    def __init__(self, weight=10):
        self.weight = weight

    def __call__(self, score_gt, geo_gt, score_pred, geo_pred):
        score = self.score_loss(score_gt, score_pred)
        rbox, angle = self.geometry_loss(geo_gt, geo_pred)
        
        angle = torch.sum(angle * score_gt) / torch.sum(score_gt)
        rbox = torch.sum(rbox * score_gt) / torch.sum(score_gt)
        geo_loss = rbox + self.weight * angle
        
        return score + 1 * geo_loss

    def score_loss(self, gt, pred, smooth=1):
        intersection = torch.sum(gt * pred)
        return 1 - (2 * intersection + smooth) / (torch.sum(gt) + torch.sum(pred) + smooth)

    def geometry_loss(self, gt, pred): 
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt, 1, 1)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred, 1, 1)

        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
        w = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
        h = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
        area_intersect = w * h
        area_union = area_gt + area_pred - area_intersect
        
        loss_rbox = -torch.log( (area_intersect+1) / (area_union+1) )
        
        loss_angle = 1 - torch.cos(angle_pred - angle_gt)
        
        return loss_rbox, loss_angle