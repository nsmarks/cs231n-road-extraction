import torch


def computeIoU(output, label):
    """
    Roads labeled as 1, background as 0
    Batch of N, 1024, 1024
    """
    # flatten
    output_reshaped = output.reshape(output.shape[0], -1) # (N, 1024 ** 2)
    label_reshaped = label.reshape(label.shape[0], -1)

    road_intersection = output_reshaped * label_reshaped
    road_union = output_reshaped + label_reshaped - road_intersection

    abs_road_intersections = torch.sum(road_intersection, dim=1)
    abs_road_unions = torch.sum(road_union, dim=1)

    individualRoadIoU = (abs_road_intersections * 1.0) / abs_road_unions
    # print ('individualRoadIoU: ', individualRoadIoU)

    inverse_output = (output_reshaped - 1) * -1 # 0 goes to 1, 1 goes to 0
    inverse_label = (label_reshaped - 1) * -1

    background_intersection = inverse_output * inverse_label
    background_union = inverse_output + inverse_label - background_intersection

    abs_background_intersections = torch.sum(background_intersection, dim=1)
    abs_background_unions = torch.sum(background_union, dim=1)

    individualBackgroundIoU = (abs_background_intersections * 1.0) / abs_background_unions



    individualIoUs = (individualRoadIoU + individualBackgroundIoU) / 2
    # print ('individualIoUs: ', individualIoUs)
    return individualIoUs

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



if __name__ == '__main__':
    '''
    Test IoU
    '''
    # 4, 2, 2
    output = torch.tensor(
        [
            [
                [1, 0],
                [0, 1]
            ],            
            [
                [1, 0],
                [0, 0]
            ],
            [
                [1, 0],
                [1, 1]
            ],
            [
                [1, 0],
                [0, 0]
            ]
        ]
    )
    labels = torch.tensor(
        [
            
            [
                [1, 0],
                [1, 1]
            ],            
            [
                [1, 0],
                [0, 0]
            ],
            [
                [1, 0],
                [0, 1]
            ],
            [
                [0, 1],
                [1, 0]
            ]
        ]
    )
    IoU = computeIoU(output, labels)
    print ('iou: ', IoU)

