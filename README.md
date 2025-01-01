<h2>Overview</h2>
<p>The Alzheimer's Detection Website is an intuitive platform designed to aid in the detection and classification of Alzheimer's disease using deep learning models. The site enables users to upload CT scan images for analysis, providing reliable classification into categories such as:</p>
<ul>
    <li>Non-Demented</li>
    <li>Very Mild Demented</li>
    <li>Mild Demented</li>
    <li>Moderate Demented</li>
</ul>
<p>This project serves as a tool to assist healthcare professionals and researchers in early detection and monitoring of Alzheimer's disease progression.</p>

<h2>Features</h2>
<ul>
    <li><strong>Image Upload:</strong> Users can easily upload CT scan images for analysis.</li>
    <li><strong>Classification Results:</strong> Provides a clear and accurate classification of the uploaded image.</li>
    <li><strong>User-Friendly Interface:</strong> Built with Flask and Jinja2, the website is simple and intuitive to navigate.</li>
    <li><strong>Robust Model:</strong> Powered by a deep learning model trained on an extensive dataset of CT scans.</li>
</ul>

<h2>Dataset</h2>
<p>The project uses an organized dataset with the following subfolders:</p>
<ul>
    <li>VeryMildDemented</li>
    <li>MildDemented</li>
    <li>ModerateDemented</li>
    <li>NonDemented</li>
</ul>
<p>Training and testing data are stored in separate folders for better validation and performance.</p>

<h2>Technologies Used</h2>
<ul>
    <li><strong>Backend:</strong> Flask framework with Jinja2 template engine.</li>
    <li><strong>Frontend:</strong> HTML, CSS, and JavaScript.</li>
    <li><strong>Machine Learning:</strong> TensorFlow and Scikit-learn for model development.</li>
    <li><strong>Data Handling:</strong> Pandas and NumPy for data preprocessing.</li>
</ul>

<h2>Installation</h2>
<ol>
    <li>Clone the repository:
        <pre><code>git clone &lt;repository_url&gt;</code></pre>
    </li>
    <li>Navigate to the project directory:
        <pre><code>cd alzheimers_detection-team_salvatores</code></pre>
    </li>
    <li>Install dependencies:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Run the Flask application:
        <pre><code>python app.py</code></pre>
    </li>
    <li>Open your browser and go to <code>http://127.0.0.1:5000</code>.</li>
</ol>

<h2>Usage</h2>
<ol>
    <li>Navigate to the website.</li>
    <li>Upload a CT scan image in the supported format.</li>
    <li>Click the "Analyze" button to classify the image.</li>
    <li>View the classification results on the results page.</li>
</ol>

<h2>File Structure</h2>
<ul>
    <li><strong>app.py:</strong> Main Flask application file.</li>
    <li><strong>templates/:</strong> Contains HTML files for the website.</li>
    <li><strong>static/:</strong> Includes CSS, JavaScript, and image assets.</li>
    <li><strong>models/:</strong> Pre-trained deep learning model.</li>
    <li><strong>data/:</strong> Organized dataset for training and testing.</li>
</ul>

<h2>Contributing</h2>
<p>Contributions are welcome! Feel free to open an issue or submit a pull request.</p>

<h2>License</h2>
<p>This project is licensed under the <a href="LICENSE">MIT License</a>.</p>

<h2>Acknowledgments</h2>
<p>Special thanks to Team Salvatores for their dedication to this project and to Assistant Professor Jamimul Bakas of BIT Mesra for their guidance in geo deepfake detection research, which inspired this work.</p>

<h2>Contact</h2>
<p>For inquiries, please contact <a href="mailto:your_email@example.com">your_email@example.com</a>.</p>
