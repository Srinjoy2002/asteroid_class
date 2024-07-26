from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
with open('your_model.pkl', 'rb') as file:
    clf1 = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message = ""

    if request.method == 'POST':
        # Get form data
        data = []
        data.append(float(request.form['minimum_orbit_intersection']))
        data.append(float(request.form['absolute_magnitude']))
        data.append(float(request.form['est_dia_in_km_min']))
        data.append(float(request.form['asc_node_longitude']))
        data.append(float(request.form['orbit_uncertainty']))
        data.append(float(request.form['perihelion_time']))
        data.append(float(request.form['perihelion_arg']))
        data.append(float(request.form['jupiter_tisserand_invariant']))
        data.append(float(request.form['mean_motion']))
        data.append(float(request.form['eccentricity']))
        data.append(float(request.form['perihelion_distance']))
        data.append(float(request.form['orbital_period']))
        data.append(float(request.form['miss_dist_astronomical']))
        data.append(float(request.form['inclination']))
        data.append(float(request.form['mean_anomaly']))
        data.append(float(request.form['aphelion_dist']))
        data.append(float(request.form['semi_major_axis']))
        data.append(float(request.form['epoch_osculation']))
        data.append(float(request.form['relative_velocity_km_per_sec']))

        # Convert data to numpy array and reshape for prediction
        data_array = np.array(data).reshape(1, -1)

        # Make prediction
        prediction = clf1.predict(data_array)[0]

        # Custom message based on prediction
        if prediction == 1:
            message = "This is a near-Earth object and is potentially hazardous."
        else:
            message = "This is not a near-Earth object and is not potentially hazardous."

    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
