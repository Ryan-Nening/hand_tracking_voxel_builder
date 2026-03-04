Hand Virtual Board
A high-performance, real-time digital interface powered by MediaPipe Hand Tracking. This project provides a touchless solution for writing, drawing, and precise spatial measurement, optimized for professional and academic environments.

🚀 Engineering & Optimization
The primary goal of this project was to develop a stable, lag-free system capable of running on standard consumer hardware. Through extensive testing and iterative development, the system was optimized by:

Computational Efficiency: Utilizing MediaPipe Model Complexity 0 to maximize FPS (Frames Per Second).

Latency Reduction: Implementing a 2D Spatial Canvas approach to ensure near-zero input lag during high-speed hand movements.

Hardware Compatibility: Specifically tuned to run seamlessly on standard CPUs by managing memory allocation through NumPy-based frame processing.

✨ Key Features
Touchless Drawing: High-precision air-writing using index finger tracking.

Virtual Caliper: An integrated measurement tool that calculates the real-time distance between the thumb and index finger for digital scaling.

Holographic Eyedropper: Advanced color-sampling logic that allows the user to "pick" and replicate colors from physical objects in the camera feed.

Precision Smoothing: Built-in Linear Interpolation (LERP) logic to eliminate jitter and provide a professional, buttery-smooth drawing experience.

Smart Clear: A gesture-based "Safety Ring" timer that prevents accidental board clearing through a timed open-palm hold.

🛠️ Tech Stack
Language: Python 3.12
Libraries: OpenCV, MediaPipe, NumPy
Core Architecture: Real-time coordinate mapping and depth-proxy brush scaling.

📖 How to Run
1. Install Dependencies:
   ```bash
   pip install mediapipe opencv-python numpy
   ```
2. Run the Application:
   ```bash
   python hand_virtual_board.py
   ```

      

       
Ensure Python 3.12 is installed.

Install dependencies:
