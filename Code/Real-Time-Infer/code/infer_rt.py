#!/usr/bin/env python3

import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from networks.VMD_network import VMD_Network  # Your custom network for video motion detection

def main():
    # Configuration and model loading
    
    # TODO: Can change this (decreasing = decrease accuracy of model but increase speed, increasing = increase accuracy of model but decrease speed)
    # Original Value 416, try 128, 256.
    args = {'scale': 256} 
    
    
    to_pil = transforms.ToPILImage()
    img_transform = transforms.Compose([
        transforms.Resize((args['scale'], args['scale'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    net = VMD_Network().cuda()
    checkpoint = r"model.pth"
    check_point = torch.load(checkpoint)
    net.load_state_dict(check_point['model'])
    net.eval()
    
    cap = cv2.VideoCapture(0)  # 0 is usually the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    fps_font = cv2.FONT_HERSHEY_SIMPLEX
    prev_time = 0
    
    try:
        with torch.no_grad():
            previous_tensor = None
            while True:
                ret, frame = cap.read()
                
                # TODO: Can change this
                # Scale up the frame by a factor of 2
                # frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
                
                if not ret:
                    print("Error: Could not read frame.")
                    break
                
                # Calculate FPS
                new_time = time.time()
                if (new_time - prev_time) > 0:
                    fps = 1 / (new_time - prev_time)
                else:
                    fps = 0
                prev_time = new_time
                fps_text = f'FPS: {fps:.2f}'
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                input_tensor = img_transform(pil_img).unsqueeze(0).cuda()
                
                if previous_tensor is not None:
                    w, h = pil_img.size
                    output, _ = net(previous_tensor, input_tensor)
                    res = (output.data > 0).to(torch.float32).squeeze(0)
                    prediction = np.array(transforms.Resize((h, w))(to_pil(res.cpu())))
                    
                    # Convert prediction back to BGR for consistent display in OpenCV
                    prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
                    
                    # Display FPS on frame
                    cv2.putText(frame, fps_text, (10, 30), fps_font, 1, (255, 255, 255), 2)
                    
                    # Display the frames
                    cv2.imshow('Input', frame)
                    cv2.imshow('Prediction', prediction_bgr)
                
                previous_tensor = input_tensor
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
