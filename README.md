# NEURAL-STYLE-TRANSFER

*COMPANY* : CODTECH IT SOLUTION

*NAME* : TALARI BHAVANA

*INTERN ID* : CT04DF313

*DOMAIN* : Artificial Intelligence

*DURATION* : 4 weeks 

*MENTOR* : NEELA SANTOSH

*DISCRIPTION* : In this project, we implemented **Neural Style Transfer (NST)** using **PyTorch**, a powerful deep learning framework. The goal of NST is to blend two images: a **content image** (such as a personal photograph) and a **style image** (like a famous painting), generating a new image that retains the structure of the content but adopts the artistic style of the second. The primary libraries used include `torch` and `torchvision`, which provide the neural network layers and pretrained models; `PIL` for image handling; `matplotlib` for visualization; and `Google Colab’s file upload utility` for loading images interactively. At the heart of the implementation is the **VGG19** model, a deep convolutional neural network pretrained on ImageNet, used as a feature extractor. Only the feature layers (`.features`) of VGG19 are used because we’re not interested in classification, but rather in capturing intermediate representations of the image at various layers. The network computes the content and style representations of the images at selected layers. Content loss is computed using **Mean Squared Error (MSE)** between the content image features and the generated image features, while style loss is computed using the **Gram matrix** of activations, which captures the correlations between different filters and reflects texture and style. The model is dynamically built using `nn.Sequential()`, and loss layers are inserted at the appropriate positions. The optimization is done using the **LBFGS optimizer**, which iteratively updates the input image (not the model weights) to minimize a weighted combination of content and style losses. The image is clamped between 0 and 1 after every iteration to ensure valid pixel values. Preprocessing is done using `torchvision.transforms`, resizing the image and converting it to a tensor. Both the content and style images are uploaded manually using `google.colab.files.upload()`, and the transformation results are visualized using `matplotlib`. The code is designed to detect and use a **GPU** if available, significantly speeding up computations. This test demonstrates how PyTorch allows for flexible model manipulation and direct control over optimization processes. The final output is a stylized image that beautifully reflects the texture and color patterns of the style image while maintaining the layout of the content image. This implementation is a powerful example of how deep learning can be used creatively and artistically, showcasing the practical application of CNNs beyond traditional classification tasks.

*OUTPUT* :
