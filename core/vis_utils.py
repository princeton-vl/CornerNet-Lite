import cv2
import numpy as np

def draw_bboxes(image, bboxes, font_size=0.5, thresh=0.5, colors=None):
    """Draws bounding boxes on an image.

    Args:
        image: An image in OpenCV format
        bboxes: A dictionary representing bounding boxes of different object
            categories, where the keys are the names of the categories and the
            values are the bounding boxes. The bounding boxes of category should be
            stored in a 2D NumPy array, where each row is a bounding box (x1, y1,
            x2, y2, score).
        font_size: (Optional) Font size of the category names.
        thresh: (Optional) Only bounding boxes with scores above the threshold
            will be drawn.
        colors: (Optional) Color of bounding boxes for each category. If it is
            not provided, this function will use random color for each category.

    Returns:
        An image with bounding boxes.
    """

    image = image.copy()
    for cat_name in bboxes:
        keep_inds = bboxes[cat_name][:, -1] > thresh
        cat_size  = cv2.getTextSize(cat_name, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]

        if colors is None:
            color = np.random.random((3, )) * 0.6 + 0.4
            color = (color * 255).astype(np.int32).tolist()
        else:
            color = colors[cat_name]

        for bbox in bboxes[cat_name][keep_inds]:
            bbox = bbox[0:4].astype(np.int32)
            if bbox[1] - cat_size[1] - 2 < 0:
                cv2.rectangle(image,
                    (bbox[0], bbox[1] + 2),
                    (bbox[0] + cat_size[0], bbox[1] + cat_size[1] + 2),
                    color, -1
                )
                cv2.putText(image, cat_name,
                    (bbox[0], bbox[1] + cat_size[1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                )
            else:
                cv2.rectangle(image,
                    (bbox[0], bbox[1] - cat_size[1] - 2),
                    (bbox[0] + cat_size[0], bbox[1] - 2),
                    color, -1
                )
                cv2.putText(image, cat_name,
                    (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
                )
            cv2.rectangle(image,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color, 2
            )
    return image
