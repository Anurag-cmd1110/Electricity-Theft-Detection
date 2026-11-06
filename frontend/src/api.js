const API_BASE_URL = "http://127.0.0.1:8000";

async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = "An unexpected error occurred.";
    try {
      const errorJson = await response.json();
      errorDetail = errorJson.detail || errorDetail;
    } catch {
      // Fallback
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function predictMeter(payload) {
  return request("/meters/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function batchPredictCsv(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/meters/predict/batch`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorJson = await response.json().catch(() => ({}));
    throw new Error(errorJson.detail || "Batch CSV processing failed.");
  }
  return response.json();
}

export async function uploadBillPdf(file) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/meters/predict/pdf`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorJson = await response.json().catch(() => ({}));
    throw new Error(errorJson.detail || "Failed to process Bill PDF.");
  }
  return response.json();
}

export async function fetchMeterHistory(limit = 50) {
  return request(`/meters/history?limit=${limit}`);
}

export async function fetchInvestigations() {
  return request("/meters/investigations");
}

export async function updateInvestigationStatus(meterId, status, notes = "") {
  return request(`/meters/investigations/${meterId}/status`, {
    method: "PATCH",
    body: JSON.stringify({ status, notes }),
  });
}

export async function fetchMeterRegistry(search = "", status = "all") {
  return request(`/meters/registry?search=${encodeURIComponent(search)}&status=${status}`);
}

export async function registerNewMeter(payload) {
  return request("/meters/registry", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function fetchMeterDetails(meterId) {
  return request(`/meters/registry/${meterId}/details`);
}

export async function syncMetersFromHistory() {
  return request("/meters/registry/sync-from-history", { method: "POST" });
}

export async function fetchFeederBalance() {
  return request("/meters/analytics/feeder-balance");
}

export async function updateConsumerName(meterId, consumerName) {
  return request(`/meters/registry/${meterId}/consumer`, {
    method: "PATCH",
    body: JSON.stringify({ consumer_name: consumerName }),
  });
}

export async function deleteSingleRecord(readingId) {
  return request(`/meters/history/record/${readingId}`, {
    method: "DELETE",
  });
}

export async function deleteMeterHistory(meterId) {
  return request(`/meters/history/meter/${meterId}`, {
    method: "DELETE",
  });
}

export async function clearAllHistory() {
  return request("/meters/history", {
    method: "DELETE",
  });
}