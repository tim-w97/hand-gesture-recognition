import cv2
import json as json_parser


def add_visualization(numpy_image, json):
    image_with_visualization = numpy_image.copy()

    height, width, channels = numpy_image.shape

    data = json_parser.loads(json)

    for gesture in data['gestures']:
        text = gesture['name']

        abs_pos_x = gesture['pos_x'] * width
        abs_pos_y = gesture['pos_y'] * height

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 5
        text_color = (0, 255, 0)

        text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_pos_x = int(abs_pos_x - text_width / 2)
        text_pos_y = int(abs_pos_y + text_height)

        cv2.putText(
            image_with_visualization,
            text,
            (text_pos_x, text_pos_y),
            font,
            font_scale,
            text_color,
            thickness
        )

    return image_with_visualization
