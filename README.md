<div align="center">
  <h1>🚀 Smart Seam Carving: Content-Aware Image Resizer</h1>
  <p><b>An intelligent image surgical tool powered by AI and Dynamic Programming</b></p>
</div>

<hr>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Algorithm-Dynamic--Programming-red.svg" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Image--Processing-cyan.svg" alt="Domain">
  <img src="https://img.shields.io/badge/AI-K--Means-green.svg" alt="AI Feature">
  <img src="https://img.shields.io/badge/UI-CustomTkinter-orange.svg" alt="UI Framework">
</p>

<h2>📌 Overview</h2>
<p align="justify">
This project implements a <b>Content-Aware Image Resizing</b> tool based on the Seam Carving algorithm. Unlike traditional scaling, which distorts the image, or cropping, which removes edges, this tool identifies and removes "seams" (paths of least importance) to resize images while preserving key features like people, buildings, and objects.
</p>



<h2>✨ Key Features</h2>
<ul>
  <li><b>AI-Powered Preprocessing:</b> Uses K-Means clustering to simplify image textures and reduce noise.</li>
  <li><b>Smart Resizing Mode:</b> Automatically chooses between horizontal and vertical seam removal based on global energy impact.</li>
  <li><b>High-Performance DP:</b> Optimized Dynamic Programming implementation using NumPy for rapid pathfinding.</li>
  <li><b>Statistical Sampling:</b> Employs sub-sampling for AI training to handle 4K images without lag.</li>
  <li><b>Multi-threaded UI:</b> A responsive GUI that remains active during heavy computations.</li>
</ul>

<h2>🛠 Technical Pipeline</h2>

<h3>1️⃣ LAB Color Space Conversion</h3>
<p>
The image is converted from RGB to the <b>LAB color space</b>. We specifically target the <b>L-channel (Lightness)</b> because human perception of edges and structures is most prominent in brightness variations rather than color shifts.
</p>

<h3>2️⃣ Intelligent Quantization (K-Means)</h3>
<p>
To prevent the algorithm from getting "distracted" by fine-grained noise, we use <code>MiniBatchKMeans</code> to quantize the L-channel into 64 dominant clusters. 
<b>Optimization:</b> We train the model on a statistically sampled downscaled version of the image and apply the results to the full-resolution matrix to ensure both speed and precision.
</p>



<h3>3️⃣ Energy Mapping (Edge Detection)</h3>
<p>
Pixel importance is calculated using the <b>Laplacian operator</b> (second-order derivative). This creates an energy map where high-contrast regions (edges) are assigned high energy values, shielding them from being removed.
</p>



<h3>4️⃣ Dynamic Programming (The Seam Search)</h3>
<p>
The optimal seam is found by solving a minimization problem. We calculate a cumulative cost matrix $M$ where each cell represents the minimum energy required to reach that pixel from the top:
</p>
<p align="center">
  $M(i, j) = E(i, j) + \min(M(i-1, j-1), M(i-1, j), M(i-1, j+1))$
</p>
<p>
Once the cost matrix is complete, we perform <b>Backtracking</b> from the bottom row to identify the exact path of the seam.
</p>



<h3>5️⃣ Multithreading & GUI</h3>
<p>
To ensure a professional user experience, the processing core runs in a separate background thread. This allows the <b>CustomTkinter</b> GUI to update the progress bar and provide live visual feedback without freezing.
</p>

<h2>📦 Requirements</h2>
<pre><code>pip install numpy opencv-python scikit-learn pillow customtkinter</code></pre>

<h2>🚀 How to Run</h2>
<ol>
  <li>Clone the repository.</li>
  <li>Run <code>main_gui.py</code>.</li>
  <li>Load your image (supports paths with non-ASCII/Persian characters).</li>
  <li>Click <b>Run AI Pre-process</b> to analyze the image structure.</li>
  <li>Enter the number of pixels to remove and click <b>Start Processing</b>.</li>
</ol>

<hr>
<div align="center">
  <p>Developed with ❤️ for Algorithm and Image Processing Research</p>
</div>
