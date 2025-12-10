// Change this if your FastAPI is hosted elsewhere
const API_BASE_URL = "http://127.0.0.1:8000";

const apiStatusEl = document.getElementById("api-status");
const apiUrlLabel = document.getElementById("api-url-label");
const form = document.getElementById("predict-form");
const submitBtn = document.getElementById("submit-btn");
const resultContent = document.getElementById("result-content");

apiUrlLabel.textContent = `API: ${API_BASE_URL}`;

// --- Check API health on load ---
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (!res.ok) throw new Error("Health check failed");
    await res.json();

    apiStatusEl.textContent = "API online";
    apiStatusEl.className = "status status--ok";
  } catch (err) {
    apiStatusEl.textContent = "API unreachable";
    apiStatusEl.className = "status status--error";
  }
}

checkHealth();

// --- Helper to format prices ---
function formatPrice(value) {
  return `$${value.toFixed(2)}`;
}

// --- Handle form submit ---
form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    latitude: parseFloat(document.getElementById("latitude").value),
    longitude: parseFloat(document.getElementById("longitude").value),
    borough: document.getElementById("borough").value,
    room_type: document.getElementById("room_type").value,
    minimum_nights: parseInt(
      document.getElementById("minimum_nights").value,
      10
    ),
    guests: parseInt(document.getElementById("guests").value, 10),
    description: document.getElementById("description").value || "",
  };

  if (
    Number.isNaN(payload.latitude) ||
    Number.isNaN(payload.longitude) ||
    !payload.borough ||
    !payload.room_type
  ) {
    alert("Please fill in latitude, longitude, borough, and room type.");
    return;
  }

  submitBtn.disabled = true;
  submitBtn.textContent = "Predicting...";

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(
        `Server error (${res.status}): ${text || res.statusText}`
      );
    }

    const data = await res.json();
    renderResult(data);
  } catch (err) {
    console.error(err);
    resultContent.innerHTML = `
      <p style="color:#fca5a5;">
        Something went wrong while calling the API.<br/>
        <small>${err.message}</small>
      </p>
    `;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Predict Price";
  }
});

// --- Render prediction response ---
function renderResult(data) {
  const {
    final_price,
    random_forest,
    linear_regression,
    neural_network,
    confidence_low,
    confidence_high,
  } = data;

  resultContent.innerHTML = `
    <div class="result-main-label">Estimated nightly price</div>
    <div class="result-main">${formatPrice(final_price)}</div>

    <div class="result-grid">
      <div class="result-card">
        <h3>Random Forest</h3>
        <p>${formatPrice(random_forest)}</p>
      </div>
      <div class="result-card">
        <h3>Linear Regression</h3>
        <p>${formatPrice(linear_regression)}</p>
      </div>
      <div class="result-card">
        <h3>Neural Network</h3>
        <p>${formatPrice(neural_network)}</p>
      </div>
      <div class="result-card">
        <h3>Confidence Range</h3>
        <p>${formatPrice(confidence_low)} – ${formatPrice(
    confidence_high
  )}</p>
      </div>
    </div>
  `;
}
