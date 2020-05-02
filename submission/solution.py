#!/usr/bin/env python3

import numpy as np

from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage

expect_shape = (480, 640, 3)


class TensorflowTemplateAgent:
    def __init__(self, load_model=False, model_path=None):
        from model import TfInference
        # define observation and output shapes
        self.model = TfInference(observation_shape=(1,) + expect_shape,  # this is the shape of the image we get.
                                 action_shape=(1, 2),  # we need to output v, omega.
                                 graph_location='tf_models/')  # this is the folder where our models are stored.
        self.current_image = np.zeros(expect_shape)

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        self.current_image = jpg2rgb(camera.jpg_data)

    def compute_action(self, observation):
        action = self.model.predict(observation)
        return action.astype(float)

    def on_received_get_commands(self, context: Context):
        pwm_left, pwm_right = self.compute_action(self.current_image)
        pwm_left = float(np.clip(pwm_left, -1, +1))
        pwm_right = float(np.clip(pwm_right, -1, +1))
        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = TensorflowTemplateAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
