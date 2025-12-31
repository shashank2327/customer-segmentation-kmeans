<h1>Customer Segmentation using K-Means Clustering</h1>

<p>
This project demonstrates <strong>customer segmentation</strong> using the
<strong>K-Means clustering</strong> algorithm. It groups customers based on their
<strong>annual income</strong> and <strong>spending behavior</strong>, helping businesses
understand customer patterns and design targeted marketing strategies.
</p>

<hr>

<h2>📌 Project Overview</h2>

<p>
Customer segmentation is an <strong>unsupervised machine learning</strong> technique used to
divide customers into meaningful groups based on similar characteristics.
In this project, K-Means clustering is applied to a retail customer dataset
to identify distinct customer segments.
</p>

<hr>

<h2>📂 Repository Structure</h2>

<pre>
customer-segmentation-kmeans/
│
├── Mall_Customers.csv           # Dataset
├── customer_segmentation.ipynb  # Jupyter Notebook (EDA + K-Means)
├── customer_app.py              # Streamlit app
├── customer_segmentation.pkl    # Saved trained model
├── requirements.txt             # Python dependencies
└── README.html                  # Project documentation
</pre>

<hr>

<h2>📊 Dataset Description</h2>

<table border="1" cellpadding="8" cellspacing="0">
    <tr>
        <th>Column Name</th>
        <th>Description</th>
    </tr>
    <tr>
        <td>CustomerID</td>
        <td>Unique customer identifier</td>
    </tr>
    <tr>
        <td>Gender</td>
        <td>Customer gender</td>
    </tr>
    <tr>
        <td>Age</td>
        <td>Customer age</td>
    </tr>
    <tr>
        <td>Annual Income (k$)</td>
        <td>Annual income in thousands</td>
    </tr>
    <tr>
        <td>Spending Score (1–100)</td>
        <td>Score based on purchasing behavior</td>
    </tr>
</table>

<hr>

<h2>🛠️ Technologies Used</h2>

<ul>
    <li>Python</li>
    <li>Pandas</li>
    <li>NumPy</li>
    <li>Matplotlib</li>
    <li>Seaborn</li>
    <li>Scikit-learn</li>
    <li>Streamlit</li>
</ul>

<hr>

<h2>🚀 How to Run the Project</h2>

<h3>Clone the Repository</h3>
<pre>
git clone https://github.com/shashank2327/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
</pre>

<h3>Install Dependencies</h3>
<pre>
pip install -r requirements.txt
</pre>

<h3>Run the Jupyter Notebook</h3>
<pre>
jupyter notebook customer_segmentation.ipynb
</pre>

<h3>Run the Streamlit App</h3>
<pre>
streamlit run customer_app.py
</pre>

<hr>

<h2>🧠 Methodology</h2>

<ol>
    <li>Load the dataset using Pandas</li>
    <li>Perform exploratory data analysis (EDA)</li>
    <li>Select relevant features for clustering</li>
    <li>Use the <strong>Elbow Method</strong> to find the optimal number of clusters</li>
    <li>Apply <strong>K-Means Clustering</strong></li>
    <li>Visualize and interpret customer segments</li>
</ol>

<hr>

<h2>📈 Results & Insights</h2>

<ul>
    <li>High income – high spending customers</li>
    <li>High income – low spending customers</li>
    <li>Low income – high spending customers</li>
    <li>Low income – low spending customers</li>
    <li>Moderate income – Moderate spending customers</li>
</ul>

<p>
These insights help businesses improve marketing strategies, customer engagement,
and product targeting.
</p>

<hr>

<h2>🤝 Contributing</h2>

<p>
Contributions are welcome! You can improve this project by adding new features,
trying different clustering algorithms, improving visualizations,
or enhancing the Streamlit UI.
</p>

<hr>

<h2>📄 License</h2>

<p>
This project is licensed under the <strong>MIT License</strong>.
</p>

<hr>
