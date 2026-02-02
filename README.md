<div align="center">
  <h1> Smart Seam Carving: Content-Aware Image Resizer</h1>
  <p><b>A high-performance seam carving tool with multi-scale support and intelligent seam selection</b></p>
</div>

<hr>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Algorithm-Dynamic%20Programming-red.svg" alt="Algorithm">
  <img src="https://img.shields.io/badge/Domain-Image%20Processing-cyan.svg" alt="Domain">
  <img src="https://img.shields.io/badge/UI-PyQt6-orange.svg" alt="UI Framework">
  <img src="https://img.shields.io/badge/Optimization-Vectorized%20NumPy-green.svg" alt="Optimization">
  <img src="https://img.shields.io/badge/Feature-Multi--scale-yellow.svg" alt="Multi-scale">
</p>

<h2>📌 Overview</h2>
<p align="justify">
This project implements a <b>Content-Aware Image Resizing</b> tool based on the Seam Carving algorithm. Unlike traditional scaling, which distorts the image, or cropping, which removes edges, this tool intelligently removes "seams" (paths of least importance) to resize images while preserving essential visual content. The implementation features a modern PyQt6 GUI, multi-scale processing for speed, and a smart mode that automatically selects the optimal seam direction.
</p>

<h2>✨ Key Features</h2>
<ul>
  <li><b>Smart Mode:</b> Automatically chooses between horizontal and vertical seam removal based on global energy cost.</li>
  <li><b>Multi-scale Support:</b> Optional downscaling before processing to dramatically speed up operations on large images.</li>
  <li><b>Vectorized DP:</b> Fully vectorized Dynamic Programming implementation using NumPy for real-time performance.</li>
  <li><b>Modern PyQt6 GUI:</b> Responsive, multi-threaded interface with live preview and progress tracking.</li>
  <li><b>Live Preview:</b> Visualizes seams in red during processing for better user insight.</li>
  <li><b>Energy-based Seam Detection:</b> Uses LAB color space and Sobel operators for robust energy mapping.</li>
</ul>

<h2>🛠 Technical Pipeline</h2>

<h3>1️⃣ Energy Map Calculation</h3>
<p>
The image is converted to the <b>LAB color space</b>, and the <b>L-channel (Lightness)</b> is extracted. Gaussian blur is applied to reduce noise, followed by <b>Sobel operators</b> to compute gradients. The energy map is defined as the sum of absolute gradients:
</p>
<p align="center">
  $E = |G_x| + |G_y|$
</p>

<h3>2️⃣ Multi-scale Preprocessing (Optional)</h3>
<p>
For large images, the <b>downscale</b> method reduces the image size by a factor (default 0.5) using area interpolation (<code>cv2.INTER_AREA</code>). This speeds up seam carving significantly while maintaining visual integrity.
</p>

<h3>3️⃣ Dynamic Programming Seam Search</h3>
<p>
A cumulative cost matrix is computed using a fully vectorized DP approach:
</p>
<p align="center">
  $C(i, j) = E(i, j) + \min(C(i-1, j-1), C(i-1, j), C(i-1, j+1))$
</p>
<p>
Backtracking is then performed from the bottom row to extract the optimal seam path.
</p>

<h3>4️⃣ Smart Direction Selection</h3>
<p>
In <b>Smart Mode</b>, the algorithm computes the total energy cost for both vertical and horizontal seams (by rotating the image) and chooses the direction with the lower cost, ensuring minimal visual distortion.
</p>

<h3>5️⃣ Multi-threaded Processing with PyQt6</h3>
<p>
The GUI runs the carving process in a separate <code>QThread</code>, ensuring the interface remains responsive. The progress bar updates in real time, and the "After" panel shows a live preview of the removed seams.
</p>

<h2>📁 Project Structure</h2>
<pre>
project/
├── project_MAIN_gui.py      # PyQt6 main window and UI logic
├── seam_carving_core.py     # Core seam carving algorithm
└── README.md                # This file
</pre>

<h2>📦 Requirements</h2>
<pre><code>pip install numpy opencv-python PyQt6</code></pre>

<h2>🚀 How to Run</h2>
<ol>
  <li>Clone the repository or download the source files.</li>
  <li>Install dependencies using the command above.</li>
  <li>Run the GUI:
    <pre><code>python project_MAIN_gui.py</code></pre>
  </li>
  <li>Use the interface:
    <ul>
      <li>Click <b>Load Image</b> to select an image.</li>
      <li>Choose the number of seams to remove.</li>
      <li>Select a mode: <b>Vertical</b>, <b>Horizontal</b>, or <b>Smart</b>.</li>
      <li>Enable <b>Multi-scale</b> for faster processing on large images.</li>
      <li>Click <b>Run</b> to start seam carving.</li>
      <li>Save the result with <b>Save</b> when processing finishes.</li>
    </ul>
  </li>
</ol>

<h2>🎨 UI Overview</h2>
<p>The interface is divided into:</p>
<ul>
  <li><b>Control Panel:</b> Load image, set parameters, choose mode, enable multi-scale, run, and save.</li>
  <li><b>Progress Bar:</b> Visual feedback during processing.</li>
  <li><b>Image Panels:</b> Side-by-side display of original and processed images with live seam preview.</li>
</ul>

<h2>⚡ Performance Notes</h2>
<ul>
  <li>Multi-scale mode can reduce processing time by up to 75% on high-resolution images.</li>
  <li>The vectorized DP implementation avoids Python loops, leveraging NumPy for near-C speed.</li>
  <li>Smart mode adds a small overhead for energy comparison but yields better visual results.</li>
</ul>

<h2>📄 License</h2>
<p>This project is open-source and available for educational and research purposes.</p>

<hr>
<div align="center">
  <p>Developed with ❤️ for Advanced Image Processing and Algorithmic Research</p>
</div>
