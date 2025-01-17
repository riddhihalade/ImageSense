

Key Features:
Automated Image Captioning: Upload an image, and receive a human-like caption generated by a state-of-the-art vision-to-language model.
Simple Integration: Easy to use endpoints for quick integration into your applications.
Robust Performance: Powered by a pre-trained Vision-Encoder-Decoder model, ensuring high-quality captions for a wide range of images.

How It Works:
Upload an Image: Send a POST request with your image file to the /predict/ endpoint.
Receive a Caption: The API processes the image and returns a descriptive caption in JSON format.

Setup:
1. Set up the environment
2. To start the server, use the following command:

uvicorn main:app --reload

By default, the server will start on http://127.0.0.1:8000. 
You can access the API documentation at http://127.0.0.1:8000/docs
