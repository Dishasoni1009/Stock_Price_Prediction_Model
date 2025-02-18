// document.getElementById("predictionForm").addEventListener("submit", async function (e) {
//     e.preventDefault();

//     // Fetch user inputs
//     const symbol = document.getElementById("symbol").value;
//     const interval = document.getElementById("interval").value;
//     const limit = document.getElementById("limit").value;

//     // Display loading state
//     document.getElementById("results").textContent = "Loading...";

//     try {
//         // Send a POST request to your backend API
//         const response = await fetch("http://127.0.0.1:5000/predict", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ symbol, interval, limit }),
//         });

//         // Parse and display results
//         const data = await response.json();
//         if (response.ok) {
//             document.getElementById("results").textContent = `Predicted Close Price: ${data.prediction}`;
//         } else {
//             document.getElementById("results").textContent = `Error: ${data.error}`;
//         }
//     } catch (err) {
//         console.error(err);
//         document.getElementById("results").textContent = "An error occurred. Please try again.";
//     }
// });













































// // document.getElementById('predictionForm').addEventListener('submit', async function (event) {
// //     event.preventDefault();

// //     // Gather form data
// //     const formData = new FormData(event.target);
// //     const data = Object.fromEntries(formData);

// //     // Convert all inputs to floats
// //     for (let key in data) {
// //         data[key] = parseFloat(data[key]);
// //     }

// //     // Send POST request to the backend
// //     try {
// //         const response = await fetch('/predict', {
// //             method: 'POST',
// //             headers: {
// //                 'Content-Type': 'application/json',
// //             },
// //             body: JSON.stringify(data),
// //         });

// //         if (!response.ok) {
// //             throw new Error('Prediction request failed');
// //         }

// //         const result = await response.json();
// //         document.getElementById('result').innerHTML = `Predicted Value: ${result.prediction}`;
// //     } catch (error) {
// //         console.error('Error:', error);
// //         document.getElementById('result').innerHTML = 'An error occurred while predicting. Please try again.';
// //     }
// // });

// ................................................................
// document.getElementById("predictionForm").addEventListener("submit", async function (e) {
//     e.preventDefault();

//     // Collect form inputs
//     const symbol = document.getElementById("symbol").value;
//     const interval = document.getElementById("interval").value;
//     const limit = document.getElementById("limit").value;

//     try {
//         const response = await fetch("/predict", {
//             method: "POST",
//             headers: { "Content-Type": "application/json" },
//             body: JSON.stringify({ symbol, interval, limit })
//         });

//         const data = await response.json();

//         // Display results
//         if (data.prediction) {
//             document.getElementById("results").innerText = `Predicted Close Price: ${data.prediction.toFixed(2)}`;
//         } else {
//             document.getElementById("results").innerText = `Error: ${data.error}`;
//         }
//     } catch (error) {
//         console.error("Error fetching prediction:", error);
//         document.getElementById("results").innerText = "Something went wrong!";
//     }
// });


document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const symbol = document.getElementById('symbol').value;
    const interval = document.getElementById('interval').value;
    const limit = document.getElementById('limit').value;

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                interval: interval,
                limit: limit
            })
        });

    // try {
    //     const response = await fetch('http://127.0.0.1:5000/predict', {
    //         method: 'POST',
    //         headers: { 'Content-Type': 'application/json' },
    //         body: JSON.stringify({
    //             symbol: symbol,
    //             interval: interval,
    //             limit: limit
    //         })
    //     });

        const data = await response.json();
        if (response.ok) {
            document.getElementById('results').innerText = `Predicted Prices: ${data.predictions.join(', ')}`;
        } else {
            document.getElementById('results').innerText = `Error: ${data.error}`;
        }
    } catch (error) {
        document.getElementById('results').innerText = `Error: ${error.message}`;
    }
});
