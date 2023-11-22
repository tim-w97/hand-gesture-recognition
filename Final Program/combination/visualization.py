import cv2
import json as json_parser


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 5
text_color = (0, 255, 0)


def add_visualization(numpy_image, json):
    image_with_visualization = numpy_image.copy()

    screen_height, screen_width, channels = numpy_image.shape

    data = json_parser.loads(json)

    # draw gesture names on the image
    for gesture in data['gestures']:
        text = gesture['name']

        abs_pos_x = gesture['pos_x'] * screen_width
        abs_pos_y = gesture['pos_y'] * screen_height

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

    # draw fingers amount on the image
    fingers_amount = data['totalFingersAmount']
    text = f'{fingers_amount} Fingers raised'

    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_pos_x = int(screen_width - text_width)
    text_pos_y = int(screen_height - text_height / 2)

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
