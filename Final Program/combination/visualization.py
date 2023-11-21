import cv2
import json as json_parser


def add_visualization(numpy_image, json):
    image_with_visualization = numpy_image.copy()

    height, width, channels = numpy_image.shape

    data = json_parser.loads(json)

    for gesture in data['gestures']:
        abs_pos_x = int(gesture['pos_x'] * width)
        abs_pos_y = int(gesture['pos_y'] * height)

        cv2.circle(
            img=image_with_visualization,
            center=(abs_pos_x, abs_pos_y),
            radius=20,
            color=(0, 255, 255),
            thickness=-1
        )

    return image_with_visualization
