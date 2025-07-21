import tensorflow as tf
import numpy as np
import PIL.Image
from PIL import Image
import io
import tempfile
import os
from rembg import remove
import argparse
import json


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = self.vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def vgg_layers(self, layer_names):
        """ Creates a vgg model that returns a list of intermediate output values."""
        # Load our model. Load pretrained VGG, trained on imagenet data
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs*255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(
            inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [self.gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


class StyleTransferProcessor:
    def __init__(self, iterations=50, style_weight=1e-2, content_weight=1e4, 
                 tv_weight=30, learning_rate=0.02):
        self.iterations = iterations
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.total_variation_weight = tv_weight
        self.lr = learning_rate
        
        # Define layers
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']
        
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        
        # Initialize model
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.opt = tf.optimizers.Adam(learning_rate=self.lr, beta_1=0.99, epsilon=1e-1)

    def load_img(self, image_path_or_bytes):
        """Load image from path or bytes"""
        max_dim = 512
        
        if isinstance(image_path_or_bytes, (str, os.PathLike)):
            # Load from file path
            img = tf.io.read_file(image_path_or_bytes)
        else:
            # Load from bytes
            img = tf.constant(image_path_or_bytes)
            
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def tensor_to_image(self, tensor):
        """Convert tensor to PIL Image"""
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def clip_0_1(self, image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def style_content_loss(self, outputs, style_targets, content_targets):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2)
                               for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2)
                                 for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.num_content_layers
        
        loss = style_loss + content_loss
        return loss

    @tf.function
    def train_step(self, image, style_targets, content_targets):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs, style_targets, content_targets)
            loss += self.total_variation_weight * tf.image.total_variation(image)
        
        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))

    def transfer_style(self, content_image_data, style_image_data, remove_bg=True, progress_callback=None):
        """
        Perform neural style transfer
        
        Args:
            content_image_data: bytes or file path of content image
            style_image_data: bytes or file path of style image
            remove_bg: whether to remove background from final image
            progress_callback: callback function for progress updates
            
        Returns:
            PIL Image of the stylized result
        """
        try:
            # Load images
            content_image = self.load_img(content_image_data)
            style_image = self.load_img(style_image_data)
            
            # Get targets
            style_targets = self.extractor(style_image)['style']
            content_targets = self.extractor(content_image)['content']
            
            # Initialize optimization variable
            image = tf.Variable(content_image)
            
            # Training loop
            for step in range(self.iterations):
                self.train_step(image, style_targets, content_targets)
                
                if progress_callback and (step % 10 == 0 or step == self.iterations - 1):
                    progress = (step + 1) / self.iterations
                    progress_callback(progress, f"Processing... Step {step + 1}/{self.iterations}")
            
            # Convert to PIL Image
            result_image = self.tensor_to_image(image)
            
            # Remove background if requested
            if remove_bg:
                if progress_callback:
                    progress_callback(0.9, "Removing background...")
                
                # Convert PIL to bytes
                img_byte_arr = io.BytesIO()
                result_image.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Remove background
                output_data = remove(img_byte_arr)
                result_image = Image.open(io.BytesIO(output_data))
            
            if progress_callback:
                progress_callback(1.0, "Complete!")
            
            return result_image
            
        except Exception as e:
            raise Exception(f"Style transfer failed: {str(e)}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Neural Style Transfer Service')
    parser.add_argument('--content', required=True, help='Content image path')
    parser.add_argument('--style', required=True, help='Style image path')
    parser.add_argument('--output', required=True, help='Output image path')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    parser.add_argument('--style-weight', type=float, default=1e-2, help='Style weight')
    parser.add_argument('--content-weight', type=float, default=1e4, help='Content weight')
    parser.add_argument('--tv-weight', type=float, default=30, help='Total variation weight')
    parser.add_argument('--learning-rate', type=float, default=0.02, help='Learning rate')
    parser.add_argument('--remove-bg', action='store_true', help='Remove background from result')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = StyleTransferProcessor(
        iterations=args.iterations,
        style_weight=args.style_weight,
        content_weight=args.content_weight,
        tv_weight=args.tv_weight,
        learning_rate=args.learning_rate
    )
    
    def progress_callback(progress, message):
        print(f"Progress: {progress*100:.1f}% - {message}")
    
    try:
        # Perform style transfer
        result_image = processor.transfer_style(
            args.content, 
            args.style, 
            remove_bg=args.remove_bg,
            progress_callback=progress_callback
        )
        
        # Save result
        result_image.save(args.output)
        
        # Return JSON response
        response = {
            "status": "success",
            "output_path": args.output,
            "message": "Style transfer completed successfully"
        }
        print(json.dumps(response))
        
    except Exception as e:
        response = {
            "status": "error",
            "error": str(e)
        }
        print(json.dumps(response))


if __name__ == "__main__":
    main()
