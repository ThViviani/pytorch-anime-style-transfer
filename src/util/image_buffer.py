import random
import torch


class ImageBuffer():
    """
    This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, buffer_size):
        """
        Initializes an ImageBuffer object with the specified buffer size.

        Parameters:
        buffer_size (int): The maximum number of images that can be stored in the buffer.

        Attributes:
        buffer_size (int): The maximum number of images in the buffer.
        num_images (int): The current number of images in the buffer (initialized to zero).
        images (list): A list to store images in the buffer (initialized as an empty list).
        """
                
        self.buffer_size = buffer_size
        if self.buffer_size > 0:
            self.num_images = 0
            self.images = []
    
    def pop(self, new_images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """

        if self.buffer_size == 0:
            return new_images
        
        return_buffer = []
        for image in new_images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_images < self.buffer_size:
                self.num_images += 1
                self.images.append(image)
                return_buffer.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.buffer_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_buffer.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_buffer.append(image)
        return_buffer = torch.cat(return_buffer, 0)
        return return_buffer

