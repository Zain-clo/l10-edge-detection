import cv2
import numpy as np
import matplotlib.pyplot as plt


def display_image(image):
    plt.figure(figsize=(8,8))
    if len(image.shape)==2:
        plt.imshow(image,cmap="gray")
    else:
        plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
    plt.show()

def interactive_edge_detection():
    image=cv2.imread("main.jpg")
    if image is None:
        print("image not found")
        return


    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #display_image(gray)

    print("""
    Select an option:
    1.Sobel Edge Detection
    2.canny edge detection
    3. Laplacian Edge detection
    4.Gaussian smoothing
    5.Exit
    """)

    while True:
        choice = input("Enter your choices(1-5):")
        if choice=="1":
            sobel_x=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=1)
            sobel_y=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=1)
            combined_sobel=cv2.bitwise_or(sobel_x.astype(np.uint8),
             sobel_y.astype(np.uint8))
            display_image(combined_sobel)

        elif choice =="2":
            print("Adjust the thresholds for Canny(default:100 and 200)")
            lower_thresh = int(input("Enter lower threshold:"))
            upper_thresh = int(input("Enter upper threshold:"))
            edges=cv2.Canny(gray,lower_thresh,upper_thresh)
            display_image(edges)

        elif choice =="3":
            laplacian=cv2.Laplacian(gray,cv2.CV_64F)
            display_image(np.abs(laplacian).astype(np.uint8))

        elif choice=="4":
            print("Adjust kernel size using an odd nober only")
            kernel_size=int(input("Enter kernel size(odd number):"))
            blurred=cv2.GaussianBlur(image,(kernel_size, kernel_size),0)
            display_image(blurred)

        elif choice=="5":
            print("Exiting...")
            break
interactive_edge_detection()
        