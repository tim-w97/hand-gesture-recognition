import cv2
import json
import pprint


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2
thickness = 5
text_color = (0, 255, 0)


def visualize_json(numpy_image, json_str):
    prettified_json = pprint.pformat(
        json.loads(json_str),
        compact=True
    )

    custom_font_scale = 1
    custom_thickness = 2

    text_width, text_height = cv2.getTextSize(prettified_json, font, custom_font_scale, custom_thickness)[0]

    offset = 10

    text_pos_x = offset
    text_pos_y = int(text_height + offset)

    cv2.putText(
        numpy_image,
        prettified_json,
        (text_pos_x, text_pos_y),
        font,
        custom_font_scale,
        text_color,
        custom_thickness
    )

    return numpy_image


def visualize_gestures(numpy_image, gestures):
    screen_height, screen_width, channels = numpy_image.shape

    for gesture in gestures:
        text = gesture['name']

        abs_pos_x = gesture['pos_x'] * screen_width
        abs_pos_y = gesture['pos_y'] * screen_height

        text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

        text_pos_x = int(abs_pos_x - text_width / 2)
        text_pos_y = int(abs_pos_y + text_height)

        cv2.putText(
            numpy_image,
            text,
            (text_pos_x, text_pos_y),
            font,
            font_scale,
            text_color,
            thickness
        )

    return numpy_image


def visualize_fingers_amount(numpy_image, fingers_amount):
    screen_height, screen_width, channels = numpy_image.shape

    text = f'{fingers_amount} Fingers raised'

    text_width, text_height = cv2.getTextSize(text, font, font_scale, thickness)[0]

    text_pos_x = int(screen_width - text_width)
    text_pos_y = int(screen_height - text_height / 2)

    cv2.putText(
        numpy_image,
        text,
        (text_pos_x, text_pos_y),
        font,
        font_scale,
        text_color,
        thickness
    )

    return numpy_image


def add_visualization(numpy_image, json_str):
    data = json.loads(json_str)

    # draw json on the image
    image_with_visualization = visualize_json(
        numpy_image,
        json_str
    )

    # draw gesture names on the image
    image_with_visualization = visualize_gestures(
        image_with_visualization,
        gestures=data['gestures']
    )

    # draw fingers amount on the image
    image_with_visualization = visualize_fingers_amount(
        image_with_visualization,
        fingers_amount=data['totalFingersAmount']
    )

    return image_with_visualization
