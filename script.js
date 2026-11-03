// ---------- Config ----------
// On Azure Static Web Apps, "/api/*" is automatically routed to your linked Azure Functions.
// For local testing against a different backend, change this to e.g. "http://localhost:7071/api/classify".
const API_URL = "https://ticket-classifier-api-sakthi-b2g6e5gdgycucqdu.centralindia-01.azurewebsites.net/api/classify";
const MIN_LENGTH = 10;

// ---------- Elements ----------
const form = document.getElementById("ticket-form");
const textarea = document.getElementById("complaint");
const charCount = document.getElementById("char-count");
const errorEl = document.getElementById("error");
const submitBtn = document.getElementById("submit-btn");
const btnText = submitBtn.querySelector(".btn-text");
const spinner = submitBtn.querySelector(".spinner");
const clearBtn = document.getElementById("clear-btn");
const resultCard = document.getElementById("result");
const resultBadge = document.getElementById("result-badge");
const confidenceWrap = document.getElementById("confidence-wrap");
const confidenceText = document.getElementById("confidence-text");
const confidenceBar = document.getElementById("confidence-bar");

// Maps model labels to badge colour classes
const CATEGORY_CLASS = {
  "Bank Account services": "cat-bank",
  "Credit card or prepaid card": "cat-card",
  "Theft/Dispute Reporting": "cat-theft",
  "Mortgage/Loan": "cat-mortgage",
  "Others": "cat-other",
};

// ---------- Helpers ----------
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  spinner.hidden = !isLoading;
  btnText.textContent = isLoading ? "Classifying..." : "Classify Ticket";
}

function showError(message) {
  errorEl.textContent = message;
}

function showResult({ category, confidence }) {
  resultBadge.textContent = category;
  resultBadge.className = "badge " + (CATEGORY_CLASS[category] || "");

  if (typeof confidence === "number") {
    const pct = Math.round(confidence * 100);
    confidenceText.textContent = pct + "%";
    confidenceWrap.hidden = false;
    // reset then animate
    confidenceBar.style.width = "0";
    requestAnimationFrame(() => (confidenceBar.style.width = pct + "%"));
  } else {
    confidenceWrap.hidden = true;
  }

  resultCard.hidden = false;
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ---------- Events ----------
textarea.addEventListener("input", () => {
  charCount.textContent = `${textarea.value.length} / ${textarea.maxLength}`;
  showError("");
});

clearBtn.addEventListener("click", () => {
  textarea.value = "";
  charCount.textContent = `0 / ${textarea.maxLength}`;
  showError("");
  resultCard.hidden = true;
  textarea.focus();
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textarea.value = chip.dataset.text;
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  });
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = textarea.value.trim();

  if (text.length < MIN_LENGTH) {
    showError(`Please enter at least ${MIN_LENGTH} characters.`);
    return;
  }

  showError("");
  setLoading(true);
  resultCard.hidden = true;

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "text/plain" },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      let detail = "";
      try { detail = (await response.json()).error || ""; } catch (_) {}
      throw new Error(detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    // Expected shape: { "category": "Mortgage/Loan", "confidence": 0.93 }  (confidence optional)
    if (!data.category) throw new Error("Unexpected response from server.");
    showResult(data);
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
});
