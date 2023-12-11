from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import gradio as gr
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

app = FastAPI()

HELLO_ROUTE = "/hello"
GOODBYE_ROUTE = "/goodbye"
iframe_dimensions = "height=300px width=1000px"


# def generate_table_content():
#     # Load the dataset
#     dataset = pd.read_csv('../warsaw.csv')
#
#     # Get the last 10 rows
#     last_10_rows = dataset.tail(10)
#
#     # Replace null values in 'TMIN' and 'TMAX' with their averages
#     last_10_rows['TMIN'].fillna(last_10_rows['TMIN'].mean(), inplace=True)
#     last_10_rows['TMAX'].fillna(last_10_rows['TMAX'].mean(), inplace=True)
#
#     # Generate HTML table rows dynamically
#     table_rows = ""
#     for index, row in last_10_rows.iterrows():
#         table_rows += f'''
#           <tr>
#             <td>{row['DATE']}</td>
#             <td>{row['TMIN']}</td>
#             <td>{row['TMAX']}</td>
#           </tr>
#         '''
#
#     return table_rows
# custom_css = """
# <style>
#     body {
#         font-family: Arial, sans-serif;
#         margin: 20px;
#     }
#      body.goodbye-page {
#         background-color: #04AA6D; /* Set the background color for the /goodbye/ page */
#     }
#     nav {
#         background-color: #04AA6D;
#         overflow: hidden;
#     }
#     nav a {
#         float: left;
#         display: block;
#         color: white;
#         text-align: center;
#         padding: 14px 16px;
#         text-decoration: none;
#     }
#     nav a:hover {
#         background-color: #ddd;
#         color: black;
#     }
#     h1 {
#         padding-top: 5rem;  /* Added padding-top */
#         padding-right: 25rem;  /* Added padding-right */
#     }
#     img {
#         width: 40%;
#         max-width: 800px;
#         height: auto;
#         margin: 20px 0;
#     }
#     #customers {
#         font-family: Arial, Helvetica, sans-serif;
#         border-collapse: collapse;
#         width: 100%;
#     }
#
#     #customers td, #customers th {
#         border: 1px solid #ddd;
#         padding: 8px;
#     }
#
#     #customers tr:nth-child(even){background-color: #f2f2f2;}
#
#     #customers tr:hover {background-color: #ddd;}
#
#     #customers th {
#         padding-top: 12px;
#         padding-bottom: 12px;
#         text-align: left;
#         background-color: #04AA6D;
#         color: white;
#     }
#
#     #header-container {
#         display: flex;
#         justify-content: space-between;
#         padding-left: 10rem;
#         padding-right: 10rem;  /* Added padding-right */
#     }
#
#     #button-container {
#         text-align: center;
#         margin-top: 20px;  /* Added margin-top */
#     }
#
#     .button-text {
#         font-size: 1.2em;  /* Adjust font size */
#         position: relative;
#     }
#
#     .button-text::before,
#     .button-text::after {
#         content: "\2190";
#         font-size: 1.5em;
#         padding: 0 5px;
#         position: absolute;
#         top: 50%;
#         transform: translateY(-50%);
#     }
#
#     .button-text::before {
#         left: -20px;
#     }
#
#     .button-text::after {
#         right: -20px;
#     }
#
#     button {
#         background-color: #04AA6D;
#         color: white;
#         padding: 10px 20px;
#         border: none;
#         border-radius: 5px;
#         cursor: pointer;
#     }
#
#     button:hover {
#         background-color: #45a049;
#     }
# </style>
# """
#
# index_html = f'''
# {custom_css}
# <body>
#     <nav class="navbar">
#         <a href="/" target="content">Home</a>
#     </nav>
#     <div>
#         <div id="header-container">
#             <h1>Welcome to the Main Page!</h1>
#             <img src="https://img.freepik.com/premium-vector/clouds-sun-cartoon-white-background-vector_566661-19137.jpg" alt="Cute Kitten">
#         </div>
#         <div id="button-container">
#             <p class="button-text">To go to the prediction page, please click the button below</p>
#         </div>
#         <div>
#             <h2>Temperature Data Table</h2>
#             <table id="customers">
#               <tr>
#                 <th>Date</th>
#                 <th>Min Temperature (TMIN)</th>
#                 <th>Max Temperature (TMAX)</th>
#               </tr>
#               {generate_table_content()}
#             </table>
#         </div>
#     </div>
# </body>
# '''
#
#
# index_html = f'''
# {custom_css}
# <body>
#     <nav>
#         <a href="/" target="content">Home</a>
#
#     </nav>
#     <div>
#         <div id="header-container">
#             <h1>Welcome to the Main Page!</h1>
#             <img src="https://img.freepik.com/premium-vector/clouds-sun-cartoon-white-background-vector_566661-19137.jpg" alt="Cute Kitten">
#         </div>
#         <div id="button-container">
#             <p>To go to the prediction page, please click the button below</p>
#             <button onclick="window.location.href='{GOODBYE_ROUTE}'">For Prediction Page</button>
#         </div>
#         <div>
#             <h2>Temperature Data Table</h2>
#             <table id="customers">
#               <tr>
#                 <th>Date</th>
#                 <th>Min Temperature (TMIN)</th>
#                 <th>Max Temperature (TMAX)</th>
#               </tr>
#               {generate_table_content()}
#             </table>
#         </div>
#     </div>
# </body>
# '''
#
#
#
# @app.get("/", response_class=HTMLResponse)
# def index():
#     return index_html



dataset = pd.read_csv('../warsaw.csv')


dataset['DATE'] = pd.to_datetime(dataset['DATE'])
dataset['Year'] = dataset['DATE'].dt.year
dataset['Month'] = dataset['DATE'].dt.month
dataset['Day'] = dataset['DATE'].dt.day


features = ['Year', 'Month', 'Day', 'PRCP', 'SNWD']
target = 'TAVG'

X = dataset[features]
y = dataset[target]

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['float64', 'int64']).columns

# Exclude non-numeric columns from imputation
numeric_features = list(set(features) - set(non_numeric_cols))

# Define transformers for numeric and non-numeric features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', 'passthrough', non_numeric_cols)
    ])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

# Create and train the kNN model with preprocessing pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=4))
])

model.fit(X_train, y_train)

with open('knn_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Define a prediction function for Gradio
def predict_temperature(year, month, day, prcp, snwd):

    year, month, day = float(year), float(month), float(day)


    prcp = float(prcp) if prcp != "" else None
    snwd = float(snwd) if snwd != "" else None


    latitude, longitude = 52.166, 20.967

    input_data = pd.DataFrame([[latitude, longitude, 110.3, year, month, day, prcp, snwd]],
                              columns=['LATITUDE', 'LONGITUDE', 'ELEVATION', 'Year', 'Month', 'Day', 'PRCP', 'SNWD'])

    prediction = model.predict(input_data)
    return prediction[0]


hello_app = gr.Interface(lambda x: "Hello, " + x + "!", "textbox", "textbox")

goodbye_app = goodbye_interface = gr.Interface(
    fn=predict_temperature,
    inputs=[
        gr.Number(label="Year"),
        gr.Number(label="Month"),
        gr.Number(label="Day"),
        gr.Number(label="Precipitation (PRCP)"),
        gr.Number(label="Snow Depth (SNWD)"),
    ],
    outputs=gr.Textbox(),
    live=True  # Set to True for live updates without restarting the server
)
app = gr.mount_gradio_app(app, hello_app, path=HELLO_ROUTE)
app = gr.mount_gradio_app(app, goodbye_app, path=GOODBYE_ROUTE)



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)

