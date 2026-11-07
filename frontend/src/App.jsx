import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  LineChart,
  Line,
} from "recharts";
import {
  predictMeter,
  batchPredictCsv,
  uploadBillPdf,
  fetchMeterHistory,
  fetchInvestigations,
  updateInvestigationStatus,
  fetchMeterRegistry,
  registerNewMeter,
  fetchMeterDetails,
  syncMetersFromHistory,
  fetchFeederBalance,
  deleteSingleRecord,
  clearAllHistory,
  updateConsumerName,
} from "./api";
import "./App.css";

const INITIAL_FORM = {
  meter_id: "",
  previous_month_kwh: "",
  current_month_kwh: "",
  avg_daily_kwh: "",
  usage_change_pct: "",
  night_usage_ratio: "0.30",
  weekend_usage_ratio: "0.95",
  zero_reading_days: "0",
};

const INITIAL_METER_FORM = {
  meter_id: "",
  consumer_name: "",
  sanctioned_load_kw: "",
  meter_type: "Three Phase Smart",
  area_zone: "",
};

const NAV_ITEMS = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "registry", label: "Meter Registry", icon: "▤" },
  { id: "analysis", label: "Meter Analysis", icon: "⌁" },
  { id: "anomalies", label: "Anomaly Detection", icon: "⚠" },
];

const COLORS = {
  normal: "#10b981",
  suspicious: "#f43f5e",
  unknown: "#94a3b8",
};

function getRiskScore(record) {
  const raw =
    record?.risk_score ??
    record?.riskScore ??
    record?.theft_probability ??
    record?.probability;
  const number = Number(raw);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(100, number));
}

function getPrediction(record) {
  const score = getRiskScore(record);
  const value = String(
    record?.prediction ?? record?.result ?? record?.status ?? record?.label ?? ""
  ).toLowerCase();

  if (
    score >= 50 ||
    record?.is_suspicious === true ||
    value.includes("theft") ||
    value.includes("suspicious") ||
    value.includes("anomal") ||
    value.includes("high risk")
  ) {
    return "Suspicious";
  }

  if (
    value.includes("normal") ||
    value.includes("no theft") ||
    value.includes("legitimate") ||
    value.includes("low risk")
  ) {
    return "Normal";
  }

  return score >= 50 ? "Suspicious" : "Normal";
}

function getMeterId(record) {
  return String(record?.meter_id ?? record?.meterId ?? "Unknown");
}

function formatDate(value) {
  const dateStr = typeof value === "object" ? value?.created_at : value;
  if (!dateStr) return "—";
  const date = new Date(dateStr);
  return Number.isNaN(date.getTime())
    ? "—"
    : date.toLocaleString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
}

function formatNumber(value, decimals = 1) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return number.toLocaleString("en-IN", { maximumFractionDigits: decimals });
}

function getUsageChange(record) {
  const explicit = Number(record?.usage_change_pct);
  if (Number.isFinite(explicit)) return explicit;
  const prev = Number(record?.previous_month_kwh) || 0;
  const curr = Number(record?.current_month_kwh) || 0;
  if (prev <= 0) return 0;
  return ((curr - prev) / prev) * 100;
}

function getRiskLabel(score) {
  if (score >= 70) return "High";
  if (score >= 40) return "Medium";
  return "Low";
}

function formatAuditBadge(status) {
  switch (status) {
    case "resolved_theft":
      return <span className="severity-tag severity-high">⚠ Confirmed Theft</span>;
    case "under_investigation":
      return <span className="severity-tag severity-medium">🔍 In Progress</span>;
    case "false_positive":
      return <span className="severity-tag severity-low">✓ False Positive</span>;
    case "pending":
      return (
        <span
          className="severity-tag"
          style={{
            background: "rgba(148, 163, 184, 0.15)",
            color: "#cbd5e1",
            border: "1px solid rgba(148, 163, 184, 0.25)",
          }}
        >
          ⏳ Pending Review
        </span>
      );
    default:
      return <span style={{ color: "var(--text-dark)", fontSize: "11px" }}>—</span>;
  }
}

function StatCard({ title, value, description, icon, tone = "default" }) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <div className="stat-card-top">
        <span className="stat-label">{title}</span>
        <span className="stat-icon" aria-hidden="true">{icon}</span>
      </div>
      <div className="stat-value">{value}</div>
      {description && <div className="stat-description">{description}</div>}
    </article>
  );
}

function StatusBadge({ status }) {
  const norm = String(status || "Unknown").toLowerCase();
  let className = "status-unknown";
  if (norm === "normal" || norm === "active") className = "status-normal";
  if (norm === "suspicious" || norm === "flagged") className = "status-suspicious";
  return <span className={`status-badge ${className}`}>{status}</span>;
}

function RiskBadge({ score }) {
  const label = getRiskLabel(score);
  const className = label === "High" ? "risk-high" : label === "Medium" ? "risk-medium" : "risk-low";
  return (
    <span className={`risk-badge ${className}`}>
      {label} · {formatNumber(score, 1)}%
    </span>
  );
}

function EmptyState({ title, description }) {
  return (
    <div className="empty-state">
      <div className="empty-state-icon" aria-hidden="true">◇</div>
      <strong>{title}</strong>
      <p>{description}</p>
    </div>
  );
}

function App() {
  const [activePage, setActivePage] = useState("dashboard");
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [history, setHistory] = useState([]);
  const [investigations, setInvestigations] = useState([]);
  const [metersList, setMetersList] = useState([]);
  const [feederBalance, setFeederBalance] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [formError, setFormError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [batchLoading, setBatchLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);

  // Staged selection for investigation status before clicking submit
  const [pendingStatuses, setPendingStatuses] = useState({});
  const [savingStatusId, setSavingStatusId] = useState(null);

  const fileInputRef = useRef(null);
  const pdfInputRef = useRef(null);

  // Filters
  const [selectedMeter, setSelectedMeter] = useState("all");
  const [selectedPeriod, setSelectedPeriod] = useState("all");
  const [registrySearch, setRegistrySearch] = useState("");

  // Inline consumer edit
  const [editingMeterId, setEditingMeterId] = useState(null);
  const [editingNameValue, setEditingNameValue] = useState("");

  // Forms
  const [form, setForm] = useState(INITIAL_FORM);
  const [meterForm, setMeterForm] = useState(INITIAL_METER_FORM);
  const [latestPrediction, setLatestPrediction] = useState(null);

  // Drilldown Modal
  const [activeMeterModal, setActiveMeterModal] = useState(null);

  const loadAllData = useCallback(async () => {
    setHistoryLoading(true);
    setApiError("");
    try {
      const [histData, invData, regData, feederData] = await Promise.all([
        fetchMeterHistory(50),
        fetchInvestigations().catch(() => ({ investigations: [] })),
        fetchMeterRegistry(registrySearch).catch(() => ({ meters: [] })),
        fetchFeederBalance().catch(() => ({ feeders: [] })),
      ]);
      setHistory(histData?.records || []);
      setInvestigations(invData?.investigations || []);
      setMetersList(regData?.meters || []);
      setFeederBalance(feederData?.feeders || []);
    } catch (error) {
      setApiError(error.message || "Failed to communicate with GridGuard backend.");
    } finally {
      setHistoryLoading(false);
    }
  }, [registrySearch]);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  const meterOptions = useMemo(() => {
    return [...new Set(history.map(getMeterId))]
      .filter((id) => id && id !== "Unknown")
      .sort((a, b) => a.localeCompare(b));
  }, [history]);

  const filteredHistory = useMemo(() => {
    const now = new Date();
    return history.filter((record) => {
      const meterId = getMeterId(record);
      if (selectedMeter !== "all" && meterId !== selectedMeter) return false;
      if (selectedPeriod === "all") return true;

      const dateStr = record?.created_at;
      if (!dateStr) return false;
      const date = new Date(dateStr);
      const age = now.getTime() - date.getTime();

      if (selectedPeriod === "today") return date.toDateString() === now.toDateString();
      if (selectedPeriod === "7days") return age >= 0 && age <= 7 * 86400000;
      if (selectedPeriod === "30days") return age >= 0 && age <= 30 * 86400000;
      return true;
    });
  }, [history, selectedMeter, selectedPeriod]);

  // Unified single source of truth: ensures top cards match substation table counts
  const summary = useMemo(() => {
    const totalConnectedMeters = feederBalance.reduce((acc, f) => acc + (f.meters || 0), 0);
    const totalSuspiciousMeters = feederBalance.reduce((acc, f) => acc + (f.suspicious_meters || 0), 0);
    const totalNormalMeters = Math.max(0, totalConnectedMeters - totalSuspiciousMeters);

    const avgRisk = filteredHistory.length
      ? filteredHistory.reduce((acc, r) => acc + getRiskScore(r), 0) / filteredHistory.length
      : 0;

    return {
      total: totalConnectedMeters > 0 ? totalConnectedMeters : filteredHistory.length,
      normal: totalNormalMeters,
      suspicious: totalSuspiciousMeters,
      averageRisk: avgRisk,
    };
  }, [filteredHistory, feederBalance]);

  const riskDistribution = useMemo(() => {
    return [
      { name: "Normal", value: summary.normal, fill: COLORS.normal },
      { name: "Suspicious", value: summary.suspicious, fill: COLORS.suspicious },
    ].filter((item) => item.value > 0);
  }, [summary]);

  const consumptionData = useMemo(() => {
    return [...filteredHistory]
      .slice(0, 8)
      .reverse()
      .map((r, i) => ({
        name: getMeterId(r) === "Unknown" ? `Rec ${i + 1}` : getMeterId(r),
        previous: Number(r.previous_month_kwh) || 0,
        current: Number(r.current_month_kwh) || 0,
      }));
  }, [filteredHistory]);

  const riskTrendData = useMemo(() => {
    return [...filteredHistory].reverse().map((r, i) => {
      const d = r.created_at ? new Date(r.created_at) : null;
      const datePart = d
        ? d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" })
        : `Rec ${i + 1}`;
      const timePart = d
        ? d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
        : "";

      return {
        pointKey: `${getMeterId(r)}-${r.id || r._id || i}-${timePart}`,
        displayDate: datePart,
        time: timePart,
        meter: getMeterId(r),
        risk: getRiskScore(r),
      };
    });
  }, [filteredHistory]);

  const handleSelectStatus = (meterId, newStatus) => {
    setPendingStatuses((prev) => ({ ...prev, [meterId]: newStatus }));
  };

  const handleSubmitInvestigation = async (meterId, currentStatus) => {
    const statusToSave = pendingStatuses[meterId] || currentStatus || "pending";
    setSavingStatusId(meterId);
    try {
      await updateInvestigationStatus(meterId, statusToSave);

      // If resolved, remove it from Open Investigations. If under investigation, keep it.
      setInvestigations((prev) => {
        if (statusToSave === "resolved_theft" || statusToSave === "false_positive") {
          return prev.filter((inv) => inv.meter_id !== meterId);
        }
        return prev.map((inv) =>
          inv.meter_id === meterId ? { ...inv, status: statusToSave, updated_at: new Date().toISOString() } : inv
        );
      });

      // Update audit status in telemetry logs
      setHistory((prev) =>
        prev.map((r) => (getMeterId(r) === meterId ? { ...r, audit_status: statusToSave } : r))
      );

      setPendingStatuses((prev) => {
        const copy = { ...prev };
        delete copy[meterId];
        return copy;
      });

      const msg =
        statusToSave === "resolved_theft" || statusToSave === "false_positive"
          ? `Investigation for ${meterId} marked as "${statusToSave}" and removed from open queue.`
          : `Investigation for ${meterId} marked as "${statusToSave}" (active).`;
      setSuccessMessage(msg);
    } catch (err) {
      alert("Failed to update status: " + err.message);
    } finally {
      setSavingStatusId(null);
    }
  };

  const handleOpenMeterModal = async (meterId) => {
    try {
      const data = await fetchMeterDetails(meterId);
      setActiveMeterModal(data);
    } catch (err) {
      alert("Error loading meter details: " + err.message);
    }
  };

  const handleSyncMeters = async () => {
    try {
      await syncMetersFromHistory();
      const [reg, fData] = await Promise.all([
        fetchMeterRegistry(registrySearch),
        fetchFeederBalance().catch(() => ({ feeders: [] })),
      ]);
      setMetersList(reg.meters || []);
      setFeederBalance(fData.feeders || []);
      setSuccessMessage("All meters synchronized and distributed across substations.");
    } catch (err) {
      alert("Sync failed: " + err.message);
    }
  };

  const handleRegisterMeter = async (e) => {
    e.preventDefault();
    try {
      await registerNewMeter({
        meter_id: meterForm.meter_id.trim(),
        consumer_name: meterForm.consumer_name.trim(),
        sanctioned_load_kw: Number(meterForm.sanctioned_load_kw),
        meter_type: meterForm.meter_type,
        area_zone: meterForm.area_zone.trim(),
      });
      setSuccessMessage(`Meter ${meterForm.meter_id} registered successfully.`);
      setMeterForm(INITIAL_METER_FORM);
      await loadAllData();
    } catch (err) {
      setFormError(err.message || "Registration failed.");
    }
  };

  const handleStartEditConsumer = (meter) => {
    setEditingMeterId(meter.meter_id);
    setEditingNameValue(meter.consumer_name || "");
  };

  const handleCancelEditConsumer = () => {
    setEditingMeterId(null);
    setEditingNameValue("");
  };

  const handleSaveConsumerName = async (meterId) => {
    if (!editingNameValue.trim()) {
      alert("Consumer name cannot be empty.");
      return;
    }
    try {
      await updateConsumerName(meterId, editingNameValue.trim());
      setMetersList((prev) =>
        prev.map((m) =>
          m.meter_id === meterId ? { ...m, consumer_name: editingNameValue.trim() } : m
        )
      );
      const fData = await fetchFeederBalance().catch(() => ({ feeders: [] }));
      setFeederBalance(fData.feeders || []);
      setEditingMeterId(null);
      setEditingNameValue("");
    } catch (err) {
      alert("Failed to update consumer name: " + err.message);
    }
  };

  const handleDeleteSingle = async (readingId, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this telemetry record?")) return;
    try {
      await deleteSingleRecord(readingId);
      await loadAllData();
    } catch (err) {
      alert("Failed to delete record: " + err.message);
    }
  };

  const handleClearAllHistory = async () => {
    if (!window.confirm("Are you sure you want to permanently clear all telemetry readings and investigations?")) return;
    try {
      await clearAllHistory();
      await loadAllData();
      setSuccessMessage("All meter history cleared successfully.");
    } catch (err) {
      alert("Failed to clear history: " + err.message);
    }
  };

  const handleBatchFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setBatchLoading(true);
    setFormError("");
    setSuccessMessage("");

    try {
      const res = await batchPredictCsv(file);
      setSuccessMessage(res.message);
      await loadAllData();
      if (res.results && res.results.length > 0) {
        setLatestPrediction(res.results[0]);
      }
    } catch (err) {
      setFormError(err.message || "Failed to process batch CSV.");
    } finally {
      setBatchLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handlePdfUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setPdfLoading(true);
    setFormError("");
    setSuccessMessage("");

    try {
      const res = await uploadBillPdf(file);
      setSuccessMessage(res.message);
      setLatestPrediction(res.evaluation);
      await loadAllData();
    } catch (err) {
      setFormError(err.message || "Failed to parse bill PDF.");
    } finally {
      setPdfLoading(false);
      if (pdfInputRef.current) pdfInputRef.current.value = "";
    }
  };

  const exportFeederBalanceCSV = () => {
    if (feederBalance.length === 0) return;
    const headers = [
      "Feeder Substation Zone",
      "Audited Meters Count",
      "Connected Meters",
      "Supplied Energy (kWh)",
      "Billed Consumption (kWh)",
      "Estimated Stolen Energy (kWh)",
      "Non-Technical Loss (%)",
      "Revenue at Risk (INR)",
      "Suspicious Meters Flagged",
    ];

    const rows = feederBalance.map((f) => [
      `"${f.feeder_zone}"`,
      f.meters,
      `"${(f.connected_meters || []).map((m) => `${m.meter_id} (${m.consumer_name})`).join("; ")}"`,
      f.supplied_kwh,
      f.billed_kwh,
      f.stolen_kwh,
      `${f.non_technical_loss_pct}%`,
      f.revenue_at_risk_inr,
      f.suspicious_meters,
    ]);

    const csvContent = "\uFEFF" + [headers.join(","), ...rows.map((e) => e.join(","))].join("\r\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gridguard-feeder-balance-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleFormChange = (event) => {
    const { name, value } = event.target;

    setForm((prev) => {
      const updated = { ...prev, [name]: value };

      if (name === "previous_month_kwh" || name === "current_month_kwh") {
        const prevKwh = Number(name === "previous_month_kwh" ? value : updated.previous_month_kwh) || 0;
        const currKwh = Number(name === "current_month_kwh" ? value : updated.current_month_kwh) || 0;

        if (currKwh > 0) {
          updated.avg_daily_kwh = (currKwh / 30).toFixed(2);
        }

        if (prevKwh > 0) {
          const change = (((currKwh - prevKwh) / prevKwh) * 100).toFixed(1);
          updated.usage_change_pct = change;
        }

        if (!updated.night_usage_ratio) updated.night_usage_ratio = "0.30";
        if (!updated.weekend_usage_ratio) updated.weekend_usage_ratio = "0.95";
        if (updated.zero_reading_days === "") updated.zero_reading_days = "0";
      }

      return updated;
    });

    setFormError("");
    setSuccessMessage("");
  };

  const handleAnalyze = async (event) => {
    event.preventDefault();
    setSubmitting(true);
    setFormError("");
    setSuccessMessage("");

    const prev = Number(form.previous_month_kwh) || 0;
    const curr = Number(form.current_month_kwh) || 0;
    const computedDaily = curr > 0 ? Number((curr / 30).toFixed(2)) : 0;
    const computedChange = prev > 0 ? Number((((curr - prev) / prev) * 100).toFixed(1)) : 0;

    const payload = {
      meter_id: form.meter_id.trim(),
      previous_month_kwh: prev,
      current_month_kwh: curr,
      avg_daily_kwh: form.avg_daily_kwh !== "" ? Number(form.avg_daily_kwh) : computedDaily,
      usage_change_pct: form.usage_change_pct !== "" ? Number(form.usage_change_pct) : computedChange,
      night_usage_ratio: form.night_usage_ratio !== "" ? Number(form.night_usage_ratio) : 0.30,
      weekend_usage_ratio: form.weekend_usage_ratio !== "" ? Number(form.weekend_usage_ratio) : 0.95,
      zero_reading_days: form.zero_reading_days !== "" ? Number(form.zero_reading_days) : 0,
    };

    try {
      const result = await predictMeter(payload);
      setLatestPrediction({ ...payload, ...result });
      setSuccessMessage("Meter analysis completed.");
      setForm(INITIAL_FORM);
      await loadAllData();
      setActivePage("analysis");
    } catch (err) {
      setFormError(err.message || "Failed to analyze meter.");
    } finally {
      setSubmitting(false);
    }
  };

  const exportToCSV = () => {
    if (filteredHistory.length === 0) return;
    const headers = [
      "Meter ID",
      "Previous Month (kWh)",
      "Current Month (kWh)",
      "Usage Change (%)",
      "Risk Score (%)",
      "Severity",
      "Prediction",
      "Audit Status",
      "Flag Reasons",
      "Date",
    ];

    const rows = filteredHistory.map((r) => [
      `"${getMeterId(r)}"`,
      r.previous_month_kwh ?? "",
      r.current_month_kwh ?? "",
      getUsageChange(r).toFixed(2),
      getRiskScore(r).toFixed(2),
      `"${getRiskLabel(getRiskScore(r))}"`,
      `"${getPrediction(r)}"`,
      `"${r.audit_status || "none"}"`,
      `"${(r.reasons || []).join("; ")}"`,
      `"${r.created_at ?? ""}"`,
    ]);

    const csvContent = "\uFEFF" + [headers.join(","), ...rows.map((e) => e.join(","))].join("\r\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `gridguard-audit-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      {mobileMenuOpen && (
        <div
          className="modal-backdrop"
          style={{ zIndex: 90 }}
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      <aside className={`sidebar ${mobileMenuOpen ? "mobile-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark">G</div>
          <div className="brand-copy">
            <strong>GridGuard</strong>
            <span>Electricity Intelligence</span>
          </div>
        </div>

        <div className="sidebar-section-label">WORKSPACE</div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={`nav-item ${activePage === item.id ? "active" : ""}`}
              onClick={() => {
                setActivePage(item.id);
                setMobileMenuOpen(false);
              }}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.label}</span>
              {item.id === "anomalies" && investigations.length > 0 && (
                <span className="nav-count">{investigations.length}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="backend-status">
            <span className={`backend-status-dot ${apiError ? "offline" : "online"}`} />
            <div>
              <strong>{apiError ? "Connection issue" : "GridGuard Core API"}</strong>
              <span>http://127.0.0.1:8000</span>
            </div>
          </div>
          <div className="sidebar-version">Demonstration Prototype</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu-toggle"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle Navigation"
            >
              ☰
            </button>
            <div className="breadcrumb">
              GridGuard <span>/</span> {NAV_ITEMS.find((i) => i.id === activePage)?.label}
            </div>
          </div>
          <button type="button" className="topbar-refresh" onClick={loadAllData} disabled={historyLoading}>
            {historyLoading ? "Refreshing…" : "↻ Refresh data"}
          </button>
        </header>

        <div className="page-content">
          <div className="prototype-notice-banner">
            <strong>Prototype Notice:</strong> Models are trained on synthetic baseline datasets. Predictions identify statistical consumption anomalies and are intended for inspection prioritization, not legal determinations.
          </div>

          {/* ---------------- DASHBOARD PAGE ---------------- */}
          {activePage === "dashboard" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">OVERVIEW</p>
                  <h1>Dashboard</h1>
                  <p className="page-subtitle">Track load variances, theft anomalies, and aggregate system metrics.</p>
                </div>
                <button type="button" className="primary-button" onClick={() => setActivePage("analysis")}>
                  + Analyze meter
                </button>
              </div>

              <div className="dashboard-filters">
                <label className="filter-control">
                  <span>Meter</span>
                  <select value={selectedMeter} onChange={(e) => setSelectedMeter(e.target.value)}>
                    <option value="all">All meters</option>
                    {meterOptions.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                </label>
                <label className="filter-control">
                  <span>Period</span>
                  <select value={selectedPeriod} onChange={(e) => setSelectedPeriod(e.target.value)}>
                    <option value="all">All records</option>
                    <option value="today">Today</option>
                    <option value="7days">Last 7 days</option>
                    <option value="30days">Last 30 days</option>
                  </select>
                </label>
              </div>

              <div className="stats-grid">
                <StatCard title="Total Connected Meters" value={summary.total} description="Across distribution substations" icon="▤" />
                <StatCard title="Normal meters" value={summary.normal} description="Healthy patterns" icon="✓" tone="success" />
                <StatCard title="Suspicious meters" value={summary.suspicious} description="Flagged for review" icon="⚠" tone="danger" />
                <StatCard title="Average risk" value={`${formatNumber(summary.averageRisk)}%`} description="System aggregate" icon="◉" tone="warning" />
              </div>

              <div className="charts-grid">
                <section className="panel chart-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Risk distribution</h2>
                      <p>Normal vs suspicious classification</p>
                    </div>
                  </div>
                  {riskDistribution.length === 0 ? (
                    <EmptyState title="No chart data" description="Run predictions to populate distribution." />
                  ) : (
                    <div className="chart-area">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={riskDistribution}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={62}
                            outerRadius={92}
                            paddingAngle={4}
                          >
                            {riskDistribution.map((entry) => (
                              <Cell key={entry.name} fill={entry.fill} />
                            ))}
                          </Pie>
                          <Tooltip />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </section>

                <section className="panel chart-panel">
                  <div className="panel-heading">
                    <div>
                      <h2>Consumption comparison</h2>
                      <p>Previous vs current month (kWh)</p>
                    </div>
                  </div>
                  {consumptionData.length === 0 ? (
                    <EmptyState title="No consumption data" description="Meter records will appear here." />
                  ) : (
                    <div className="chart-area">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={consumptionData} margin={{ top: 10, right: 10, left: -15, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255, 255, 255, 0.07)" />
                          <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8" }} stroke="rgba(255, 255, 255, 0.15)" />
                          <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} stroke="rgba(255, 255, 255, 0.15)" />
                          <Tooltip
                            cursor={{ fill: "rgba(255, 255, 255, 0.05)" }}
                            formatter={(value, name) => [
                              `${value} kWh`,
                              name === "current" ? "Current month" : "Previous month",
                            ]}
                            labelFormatter={(label) => `Meter: ${label}`}
                          />
                          <Legend />
                          <Bar dataKey="current" name="Current month" fill="#818cf8" radius={[4, 4, 0, 0]} />
                          <Bar dataKey="previous" name="Previous month" fill="#475569" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </section>

                <section className="panel chart-panel chart-panel-wide">
                  <div className="panel-heading">
                    <div>
                      <h2>Risk score trend</h2>
                      <p>Chronological anomaly probabilities</p>
                    </div>
                  </div>
                  {riskTrendData.length === 0 ? (
                    <EmptyState title="No trend available" description="Predictions will populate this timeline." />
                  ) : (
                    <div className="chart-area">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={riskTrendData} margin={{ top: 10, right: 18, left: -10, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255, 255, 255, 0.07)" />
                          <XAxis
                            dataKey="pointKey"
                            tick={{ fontSize: 10, fill: "#94a3b8" }}
                            tickFormatter={(value, idx) => riskTrendData[idx]?.displayDate || value}
                            stroke="rgba(255, 255, 255, 0.15)"
                          />
                          <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: "#94a3b8" }} tickFormatter={(v) => `${v}%`} stroke="rgba(255, 255, 255, 0.15)" />
                          <Tooltip
                            cursor={{ stroke: "rgba(129, 140, 248, 0.3)", strokeWidth: 1.5 }}
                            formatter={(val) => [`${formatNumber(val)}%`, "Risk score"]}
                            labelFormatter={(label, payload) => {
                              const item = payload?.[0]?.payload;
                              return item ? `${item.displayDate} ${item.time ? `(${item.time})` : ""} · Meter: ${item.meter}` : label;
                            }}
                          />
                          <Line
                            type="monotone"
                            dataKey="risk"
                            stroke="#8b5cf6"
                            strokeWidth={2.5}
                            dot={{ r: 4, fill: "#a78bfa", stroke: "#0f172a", strokeWidth: 1.5 }}
                            activeDot={{ r: 7, fill: "#c4b5fd" }}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </section>
              </div>

              {/* Feeder Substation Energy Accounting */}
              <section className="panel" style={{ marginBottom: "28px" }}>
                <div className="panel-heading">
                  <div>
                    <h2>Distribution Feeder Energy Balance & Non-Technical Loss (NTL)</h2>
                    <p>Substation-level energy balancing showing connected meters per feeder line (Tariff: ₹6.50/kWh)</p>
                  </div>
                  <button
                    type="button"
                    className="refresh-button"
                    onClick={exportFeederBalanceCSV}
                    disabled={feederBalance.length === 0}
                    title="Export Distribution Feeder Energy Balance to CSV"
                  >
                    ↓ Export Feeder Audit CSV
                  </button>
                </div>

                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feeder Substation Zone</th>
                        <th>Audited Meters</th>
                        <th>Connected Meters (Consumer Name)</th>
                        <th>Supplied (DT Out)</th>
                        <th>Billed Consumption</th>
                        <th>Non-Technical Loss</th>
                        <th>Unrecovered Revenue</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {feederBalance.length === 0 ? (
                        <tr><td colSpan="8" style={{ textAlign: "center", padding: "18px" }}>No feeder data aggregated yet.</td></tr>
                      ) : (
                        feederBalance.map((f, i) => (
                          <tr key={i}>
                            <td className="meter-id-cell">{f.feeder_zone}</td>
                            <td>{f.meters}</td>
                            <td>
                              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", maxWidth: "340px" }}>
                                {(f.connected_meters || []).map((cm) => (
                                  <span
                                    key={cm.meter_id}
                                    className="link-cell"
                                    onClick={() => handleOpenMeterModal(cm.meter_id)}
                                    style={{
                                      display: "inline-flex",
                                      alignItems: "center",
                                      gap: "4px",
                                      padding: "3px 7px",
                                      borderRadius: "4px",
                                      fontSize: "11px",
                                      background: cm.is_suspicious ? "rgba(244, 63, 94, 0.15)" : "rgba(6, 182, 212, 0.12)",
                                      color: cm.is_suspicious ? "var(--accent-rose)" : "var(--accent-cyan)",
                                      border: `1px solid ${cm.is_suspicious ? "rgba(244, 63, 94, 0.3)" : "rgba(6, 182, 212, 0.25)"}`,
                                    }}
                                    title={`Click to view audit history for ${cm.consumer_name}`}
                                  >
                                    <strong>{cm.meter_id}</strong>
                                    <small style={{ opacity: 0.85 }}>({cm.consumer_name.replace(`Consumer (${cm.meter_id})`, "Auto")})</small>
                                    {cm.is_suspicious && " ⚠"}
                                  </span>
                                ))}
                              </div>
                            </td>
                            <td>{formatNumber(f.supplied_kwh)} kWh</td>
                            <td>{formatNumber(f.billed_kwh)} kWh</td>
                            <td>
                              <span className={f.non_technical_loss_pct > 15 ? "change-negative" : "change-positive"}>
                                {f.non_technical_loss_pct}% loss
                              </span>
                            </td>
                            <td style={{ color: "var(--accent-amber)", fontWeight: "700" }}>
                              ₹{formatNumber(f.revenue_at_risk_inr)}
                            </td>
                            <td>
                              {f.suspicious_meters > 0 ? (
                                <span className="severity-tag severity-high">{f.suspicious_meters} Suspicious</span>
                              ) : (
                                <span className="severity-tag severity-low">Balanced</span>
                              )}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="panel history-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Recent meter analyses</h2>
                    <p>{filteredHistory.length} total records displayed</p>
                  </div>
                  <div style={{ display: "flex", gap: "10px" }}>
                    <button
                      type="button"
                      className="refresh-button"
                      style={{ color: "var(--accent-rose)", borderColor: "rgba(244, 63, 94, 0.3)" }}
                      onClick={handleClearAllHistory}
                      disabled={filteredHistory.length === 0}
                    >
                      🗑 Clear All
                    </button>
                    <button type="button" className="refresh-button" onClick={exportToCSV} disabled={filteredHistory.length === 0}>
                      ↓ Export CSV
                    </button>
                  </div>
                </div>

                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Meter ID</th>
                        <th>Previous (kWh)</th>
                        <th>Current (kWh)</th>
                        <th>Variance</th>
                        <th>Risk score</th>
                        <th>Prediction</th>
                        <th>Audit Status</th>
                        <th>Date</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredHistory.map((r, i) => (
                        <tr key={r.id || i}>
                          <td className="meter-id-cell link-cell" onClick={() => handleOpenMeterModal(getMeterId(r))}>
                            {getMeterId(r)} ↗
                          </td>
                          <td>{formatNumber(r.previous_month_kwh)}</td>
                          <td>{formatNumber(r.current_month_kwh)}</td>
                          <td>
                            <span className={getUsageChange(r) < 0 ? "change-negative" : "change-positive"}>
                              {getUsageChange(r) > 0 ? "+" : ""}{formatNumber(getUsageChange(r))}%
                            </span>
                          </td>
                          <td><RiskBadge score={getRiskScore(r)} /></td>
                          <td><StatusBadge status={getPrediction(r)} /></td>
                          <td>{formatAuditBadge(r.audit_status)}</td>
                          <td>{formatDate(r)}</td>
                          <td>
                            <button
                              type="button"
                              className="modal-close-btn"
                              style={{ color: "var(--accent-rose)", padding: "4px 8px" }}
                              onClick={(e) => handleDeleteSingle(r.id, e)}
                              title="Delete this reading"
                            >
                              ✕
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {/* ---------------- METER REGISTRY PAGE ---------------- */}
          {activePage === "registry" && (
            <div className="registry-layout">
              <section className="panel form-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Register new meter</h2>
                    <p>Add meter connection to database registry.</p>
                  </div>
                </div>

                <form onSubmit={handleRegisterMeter} className="meter-form">
                  <div className="form-field form-field-full">
                    <label>Meter ID</label>
                    <input
                      name="meter_id"
                      value={meterForm.meter_id}
                      onChange={(e) => setMeterForm({ ...meterForm, meter_id: e.target.value })}
                      placeholder="e.g. MTR-2045"
                      required
                    />
                  </div>
                  <div className="form-field form-field-full">
                    <label>Consumer Name</label>
                    <input
                      name="consumer_name"
                      value={meterForm.consumer_name}
                      onChange={(e) => setMeterForm({ ...meterForm, consumer_name: e.target.value })}
                      placeholder="e.g. Priya Sharma"
                      required
                    />
                  </div>
                  <div className="form-field">
                    <label>Sanctioned Load (kW)</label>
                    <input
                      name="sanctioned_load_kw"
                      type="number"
                      step="0.5"
                      min="0.5"
                      value={meterForm.sanctioned_load_kw}
                      onChange={(e) => setMeterForm({ ...meterForm, sanctioned_load_kw: e.target.value })}
                      placeholder="e.g. 5.0"
                      required
                    />
                  </div>
                  <div className="form-field">
                    <label>Meter Hardware Type</label>
                    <select
                      value={meterForm.meter_type}
                      onChange={(e) => setMeterForm({ ...meterForm, meter_type: e.target.value })}
                    >
                      <option value="Single Phase Smart">Single Phase Smart</option>
                      <option value="Three Phase Smart">Three Phase Smart</option>
                      <option value="Three Phase Industrial">Three Phase Industrial</option>
                    </select>
                  </div>
                  <div className="form-field form-field-full">
                    <label>Area / Distribution Substation (Leave empty for auto-assign)</label>
                    <input
                      name="area_zone"
                      value={meterForm.area_zone}
                      onChange={(e) => setMeterForm({ ...meterForm, area_zone: e.target.value })}
                      placeholder="e.g. Sector 4 Feeder (Substation A)"
                    />
                  </div>

                  {formError && <div className="alert alert-error form-field-full">{formError}</div>}
                  {successMessage && <div className="alert alert-success form-field-full">{successMessage}</div>}

                  <div className="form-actions form-field-full">
                    <button type="submit" className="primary-button">
                      + Register Meter
                    </button>
                    <button type="button" className="refresh-button" onClick={handleSyncMeters} title="Import all meters found in existing prediction history across substations">
                      ↻ Sync from History
                    </button>
                  </div>
                </form>
              </section>

              <section className="panel directory-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Meter Directory ({metersList.length})</h2>
                    <p>Select any meter to view full audit logs. Click consumer name to edit.</p>
                  </div>
                  <input
                    type="text"
                    className="search-input"
                    placeholder="Search ID, name or zone…"
                    value={registrySearch}
                    onChange={(e) => setRegistrySearch(e.target.value)}
                  />
                </div>

                {metersList.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon">▤</div>
                    <strong>No meters in directory</strong>
                    <p>Register a meter on the left, or click <strong>"Sync from History"</strong> to auto-import existing meters.</p>
                  </div>
                ) : (
                  <div className="table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Meter ID</th>
                          <th>Consumer</th>
                          <th>Load</th>
                          <th>Feeder Substation</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {metersList.map((m) => (
                          <tr key={m.id}>
                            <td className="meter-id-cell link-cell" onClick={() => handleOpenMeterModal(m.meter_id)}>
                              {m.meter_id} ↗
                            </td>
                            <td>
                              {editingMeterId === m.meter_id ? (
                                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                                  <input
                                    type="text"
                                    value={editingNameValue}
                                    onChange={(e) => setEditingNameValue(e.target.value)}
                                    autoFocus
                                    style={{
                                      background: "#090e1a",
                                      color: "#f1f5f9",
                                      border: "1px solid var(--accent-indigo)",
                                      borderRadius: "4px",
                                      padding: "4px 8px",
                                      fontSize: "12px",
                                      width: "160px",
                                    }}
                                    onKeyDown={(e) => {
                                      if (e.key === "Enter") handleSaveConsumerName(m.meter_id);
                                      if (e.key === "Escape") handleCancelEditConsumer();
                                    }}
                                  />
                                  <button
                                    type="button"
                                    onClick={() => handleSaveConsumerName(m.meter_id)}
                                    style={{
                                      background: "rgba(16, 185, 129, 0.2)",
                                      color: "var(--accent-green)",
                                      border: "1px solid rgba(16, 185, 129, 0.4)",
                                      borderRadius: "4px",
                                      cursor: "pointer",
                                      padding: "3px 6px",
                                      fontSize: "12px",
                                    }}
                                    title="Save"
                                  >
                                    ✓
                                  </button>
                                  <button
                                    type="button"
                                    onClick={handleCancelEditConsumer}
                                    style={{
                                      background: "rgba(244, 63, 94, 0.2)",
                                      color: "var(--accent-rose)",
                                      border: "1px solid rgba(244, 63, 94, 0.4)",
                                      borderRadius: "4px",
                                      cursor: "pointer",
                                      padding: "3px 6px",
                                      fontSize: "12px",
                                    }}
                                    title="Cancel"
                                  >
                                    ✕
                                  </button>
                                </div>
                              ) : (
                                <div
                                  className="consumer-name-wrap"
                                  onClick={() => handleStartEditConsumer(m)}
                                  title="Click to rename consumer"
                                >
                                  <span className="consumer-title">{m.consumer_name}</span>
                                  <span className="consumer-edit-icon">✎</span>
                                </div>
                              )}
                            </td>
                            <td>{m.sanctioned_load_kw} kW</td>
                            <td>{m.area_zone}</td>
                            <td><StatusBadge status={m.status || "Active"} /></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </div>
          )}

          {/* ---------------- METER ANALYSIS PAGE ---------------- */}
          {activePage === "analysis" && (
            <div className="analysis-layout">
              <section className="panel form-panel">
                <div className="panel-heading">
                  <div>
                    <h2>Meter Telemetry Input</h2>
                    <p>Enter billing units, upload bulk CSV, or parse digital bill PDF.</p>
                  </div>
                  <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                    <input
                      type="file"
                      ref={fileInputRef}
                      style={{ display: "none" }}
                      accept=".csv"
                      onChange={handleBatchFileUpload}
                    />
                    <button
                      type="button"
                      className="refresh-button"
                      onClick={() => fileInputRef.current?.click()}
                      disabled={batchLoading || pdfLoading}
                      title="Upload CSV with multiple meter readings"
                    >
                      {batchLoading ? "Uploading…" : "📁 Upload Batch CSV"}
                    </button>

                    <input
                      type="file"
                      ref={pdfInputRef}
                      style={{ display: "none" }}
                      accept=".pdf"
                      onChange={handlePdfUpload}
                    />
                    <button
                      type="button"
                      className="refresh-button"
                      style={{ borderColor: "rgba(99, 102, 241, 0.4)", color: "var(--accent-indigo)" }}
                      onClick={() => pdfInputRef.current?.click()}
                      disabled={pdfLoading || batchLoading}
                      title="Upload Electricity Bill Invoice PDF"
                    >
                      {pdfLoading ? "Reading PDF…" : "📄 Upload Bill PDF"}
                    </button>
                  </div>
                </div>

                <form onSubmit={handleAnalyze} className="meter-form">
                  <div className="form-field form-field-full">
                    <label>Meter ID</label>
                    <input
                      name="meter_id"
                      value={form.meter_id}
                      onChange={handleFormChange}
                      placeholder="e.g. MTR-0042"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label>Previous Month Consumption (kWh)</label>
                    <input
                      name="previous_month_kwh"
                      type="number"
                      min="0"
                      step="any"
                      value={form.previous_month_kwh}
                      onChange={handleFormChange}
                      placeholder="e.g. 200"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label>Current Month Consumption (kWh)</label>
                    <input
                      name="current_month_kwh"
                      type="number"
                      min="0"
                      step="any"
                      value={form.current_month_kwh}
                      onChange={handleFormChange}
                      placeholder="e.g. 100"
                      required
                    />
                  </div>

                  <div className="form-field">
                    <label>
                      Avg Daily Consumption (kWh){" "}
                      <small style={{ color: "var(--accent-cyan)" }}>[Auto]</small>
                    </label>
                    <input
                      name="avg_daily_kwh"
                      type="number"
                      step="any"
                      value={form.avg_daily_kwh}
                      onChange={handleFormChange}
                      placeholder="Auto-calculated (kWh / 30)"
                      readOnly
                      style={{ opacity: 0.85, cursor: "not-allowed" }}
                    />
                  </div>

                  <div className="form-field">
                    <label>
                      Usage Change (%){" "}
                      <small style={{ color: "var(--accent-cyan)" }}>[Auto]</small>
                    </label>
                    <input
                      name="usage_change_pct"
                      type="number"
                      step="any"
                      value={form.usage_change_pct}
                      onChange={handleFormChange}
                      placeholder="Auto-calculated variance"
                      readOnly
                      style={{ opacity: 0.85, cursor: "not-allowed" }}
                    />
                  </div>

                  <div className="form-field">
                    <label>Night Usage Ratio (0 to 1)</label>
                    <input
                      name="night_usage_ratio"
                      type="number"
                      min="0"
                      max="1.5"
                      step="any"
                      value={form.night_usage_ratio}
                      onChange={handleFormChange}
                      placeholder="0.30"
                    />
                  </div>

                  <div className="form-field">
                    <label>Weekend Usage Ratio (0 to 1)</label>
                    <input
                      name="weekend_usage_ratio"
                      type="number"
                      min="0"
                      max="2.0"
                      step="any"
                      value={form.weekend_usage_ratio}
                      onChange={handleFormChange}
                      placeholder="0.95"
                    />
                  </div>

                  <div className="form-field form-field-full">
                    <label>Zero Reading Days</label>
                    <input
                      name="zero_reading_days"
                      type="number"
                      min="0"
                      step="1"
                      value={form.zero_reading_days}
                      onChange={handleFormChange}
                      placeholder="0"
                    />
                  </div>

                  {formError && <div className="alert alert-error form-field-full">{formError}</div>}
                  {successMessage && <div className="alert alert-success form-field-full">{successMessage}</div>}

                  <div className="form-actions form-field-full">
                    <button type="submit" className="primary-button" disabled={submitting}>
                      {submitting ? "Analyzing…" : "Run ML Anomaly Analysis"}
                    </button>
                  </div>
                </form>
              </section>

              {/* Diagnostic Verdict Card */}
              <section className="panel result-panel diagnostic-card">
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">ML INFERENCE</span>
                    <h2>Diagnostic Verdict</h2>
                    <p>Explainable AI anomaly & telemetry breakdown</p>
                  </div>
                  {latestPrediction && (
                    <StatusBadge status={getPrediction(latestPrediction)} />
                  )}
                </div>

                {!latestPrediction ? (
                  <EmptyState
                    title="No active evaluation"
                    description="Enter telemetry readings, upload a batch CSV, or parse a bill PDF."
                  />
                ) : (
                  <div className="verdict-body">
                    <div className="verdict-hero">
                      <div className="verdict-score-group">
                        <span className="verdict-label">ANOMALY PROBABILITY</span>
                        <div className="verdict-metric">
                          <span className="score-number">
                            {formatNumber(getRiskScore(latestPrediction))}
                          </span>
                          <span className="score-unit">%</span>
                        </div>
                        <div className="verdict-meta">
                          <span>Severity Level:</span>
                          <span
                            className={`severity-tag severity-${(
                              latestPrediction.severity || getRiskLabel(getRiskScore(latestPrediction))
                            ).toLowerCase()}`}
                          >
                            {latestPrediction.severity || getRiskLabel(getRiskScore(latestPrediction))}
                          </span>
                        </div>
                      </div>

                      <div className="meter-id-badge-large">
                        <span className="badge-title">TARGET METER</span>
                        <strong>{latestPrediction.meter_id}</strong>
                        <small>{formatDate(latestPrediction.created_at || new Date())}</small>
                      </div>
                    </div>

                    <div className="gauge-track">
                      <div
                        className={`gauge-fill severity-bar-${(
                          latestPrediction.severity || getRiskLabel(getRiskScore(latestPrediction))
                        ).toLowerCase()}`}
                        style={{ width: `${getRiskScore(latestPrediction)}%` }}
                      />
                    </div>

                    <div className="verdict-metrics-grid">
                      <div className="metric-chip">
                        <span>Previous Month</span>
                        <strong>{formatNumber(latestPrediction.previous_month_kwh)} kWh</strong>
                      </div>
                      <div className="metric-chip">
                        <span>Current Month</span>
                        <strong>{formatNumber(latestPrediction.current_month_kwh)} kWh</strong>
                      </div>
                      <div className="metric-chip">
                        <span>Load Variance</span>
                        <strong
                          className={
                            getUsageChange(latestPrediction) < 0
                              ? "change-negative"
                              : "change-positive"
                          }
                        >
                          {getUsageChange(latestPrediction) > 0 ? "+" : ""}
                          {formatNumber(getUsageChange(latestPrediction))}%
                        </strong>
                      </div>
                      <div className="metric-chip">
                        <span>Zero Days</span>
                        <strong>{latestPrediction.zero_reading_days ?? 0} Days</strong>
                      </div>
                    </div>

                    {latestPrediction.reasons && latestPrediction.reasons.length > 0 && (
                      <div className="diagnostic-reasons">
                        <div className="reasons-header">
                          <span className="warning-dot" />
                          <strong>Identified Anomaly Indicators</strong>
                        </div>
                        <ul>
                          {latestPrediction.reasons.map((r, idx) => (
                            <li key={idx}>{r}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    <div className="verdict-disclaimer">
                      <span className="info-icon">ℹ</span>
                      <p>
                        {latestPrediction.warning ||
                          "Trained on synthetic demonstration data. This score indicates statistical variance and is intended to prioritize field inspection."}
                      </p>
                    </div>
                  </div>
                )}
              </section>
            </div>
          )}

          {/* ---------------- ANOMALY WORKFLOW PAGE ---------------- */}
          {activePage === "anomalies" && (
            <>
              <div className="page-heading">
                <div>
                  <p className="eyebrow">RISK MONITORING</p>
                  <h1>Anomaly Detection & Workflow</h1>
                  <p className="page-subtitle">Assign field audit stages and log resolution statuses directly in MongoDB.</p>
                </div>
              </div>

              <div className="stats-grid">
                <StatCard
                  title="Flagged records"
                  value={summary.suspicious}
                  description="Suspicious predictions in active view"
                  icon="⚠"
                  tone="danger"
                />
                <StatCard
                  title="Active Investigations"
                  value={investigations.length}
                  description="Meters requiring site audit"
                  icon="!"
                  tone="warning"
                />
                <StatCard
                  title="Average risk"
                  value={`${formatNumber(summary.averageRisk)}%`}
                  description="Across matching records"
                  icon="◉"
                />
              </div>

              <section className="panel anomaly-panel" style={{ marginBottom: "28px" }}>
                <div className="panel-heading">
                  <div>
                    <h2>Open Investigations ({investigations.length})</h2>
                    <p>Assign field audit stages and update case statuses</p>
                  </div>
                </div>

                {investigations.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-icon">✓</div>
                    <strong>No pending investigations</strong>
                    <p>All flagged meter anomalies have been resolved or none are present.</p>
                  </div>
                ) : (
                  <div className="table-container">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Meter ID</th>
                          <th>Severity</th>
                          <th>Anomaly Score</th>
                          <th>Triggers & Root Causes</th>
                          <th>Audit Status</th>
                          <th>Last Modified</th>
                          <th style={{ textAlign: "center" }}>Action</th>
                        </tr>
                      </thead>
                      <tbody>
                        {investigations.map((inv) => {
                          const displayReasons =
                            inv.reasons && inv.reasons.length > 0
                              ? inv.reasons
                              : ["Significant drop in consumption detected", "Suppressed baseline usage"];

                          const selectedVal = pendingStatuses[inv.meter_id] ?? inv.status ?? "pending";
                          const hasUnsavedChange =
                            pendingStatuses[inv.meter_id] !== undefined &&
                            pendingStatuses[inv.meter_id] !== inv.status;

                          return (
                            <tr key={inv.id}>
                              <td className="meter-id-cell link-cell" onClick={() => handleOpenMeterModal(inv.meter_id)}>
                                {inv.meter_id} ↗
                              </td>
                              <td>
                                <span className={`severity-tag severity-${(inv.severity || "medium").toLowerCase()}`}>
                                  {inv.severity || "Medium"}
                                </span>
                              </td>
                              <td style={{ fontWeight: "700", color: "#f8fafc" }}>
                                {formatNumber(inv.risk_score)}%
                              </td>
                              <td>
                                <div className="triggers-cell">
                                  {displayReasons.map((res, i) => (
                                    <span key={i} className="trigger-badge">
                                      {res}
                                    </span>
                                  ))}
                                </div>
                              </td>
                              <td>
                                <select
                                  className="status-selector"
                                  value={selectedVal}
                                  onChange={(e) => handleSelectStatus(inv.meter_id, e.target.value)}
                                >
                                  <option value="pending">⏳ Pending Review</option>
                                  <option value="under_investigation">🔍 Under Investigation</option>
                                  <option value="resolved_theft">⚠ Confirmed Theft</option>
                                  <option value="false_positive">✓ False Positive</option>
                                </select>
                              </td>
                              <td>{formatDate(inv.updated_at)}</td>
                              <td style={{ textAlign: "center" }}>
                                <button
                                  type="button"
                                  className="primary-button"
                                  style={{
                                    padding: "5px 12px",
                                    fontSize: "11.5px",
                                    background: hasUnsavedChange ? "#6366f1" : "rgba(99, 102, 241, 0.2)",
                                    border: hasUnsavedChange ? "1px solid #818cf8" : "1px solid rgba(99, 102, 241, 0.4)",
                                    cursor: savingStatusId === inv.meter_id ? "wait" : "pointer",
                                  }}
                                  disabled={savingStatusId === inv.meter_id}
                                  onClick={() => handleSubmitInvestigation(inv.meter_id, inv.status)}
                                >
                                  {savingStatusId === inv.meter_id ? "Saving…" : hasUnsavedChange ? "Submit *" : "Submit"}
                                </button>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>

              <section className="panel history-panel">
                <div className="panel-heading">
                  <div>
                    <h2>All Flagged Meter Records ({filteredHistory.length})</h2>
                    <p>Complete historical telemetry logs</p>
                  </div>
                  <div style={{ display: "flex", gap: "10px" }}>
                    <button
                      type="button"
                      className="refresh-button"
                      style={{ color: "var(--accent-rose)", borderColor: "rgba(244, 63, 94, 0.3)" }}
                      onClick={handleClearAllHistory}
                      disabled={filteredHistory.length === 0}
                    >
                      🗑 Clear All
                    </button>
                    <button type="button" className="refresh-button" onClick={exportToCSV} disabled={filteredHistory.length === 0}>
                      ↓ Export CSV
                    </button>
                  </div>
                </div>

                <div className="table-container">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Meter ID</th>
                        <th>Previous (kWh)</th>
                        <th>Current (kWh)</th>
                        <th>Variance</th>
                        <th>Risk score</th>
                        <th>Prediction</th>
                        <th>Audit Status</th>
                        <th>Date</th>
                        <th>Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredHistory.map((r, i) => (
                        <tr key={r.id || i}>
                          <td className="meter-id-cell link-cell" onClick={() => handleOpenMeterModal(getMeterId(r))}>
                            {getMeterId(r)} ↗
                          </td>
                          <td>{formatNumber(r.previous_month_kwh)}</td>
                          <td>{formatNumber(r.current_month_kwh)}</td>
                          <td>
                            <span className={getUsageChange(r) < 0 ? "change-negative" : "change-positive"}>
                              {getUsageChange(r) > 0 ? "+" : ""}{formatNumber(getUsageChange(r))}%
                            </span>
                          </td>
                          <td><RiskBadge score={getRiskScore(r)} /></td>
                          <td><StatusBadge status={getPrediction(r)} /></td>
                          <td>{formatAuditBadge(r.audit_status)}</td>
                          <td>{formatDate(r)}</td>
                          <td>
                            <button
                              type="button"
                              className="modal-close-btn"
                              style={{ color: "var(--accent-rose)", padding: "4px 8px" }}
                              onClick={(e) => handleDeleteSingle(r.id, e)}
                              title="Delete this reading"
                            >
                              ✕
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {/* ---------------- DRILLDOWN MODAL ---------------- */}
          {activeMeterModal && (
            <div className="modal-backdrop" onClick={() => setActiveMeterModal(null)}>
              <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-header">
                  <div>
                    <h2>Meter Profile: {activeMeterModal.profile.meter_id}</h2>
                    <p>
                      Consumer: <strong>{activeMeterModal.profile.consumer_name}</strong> · Load: {activeMeterModal.profile.sanctioned_load_kw} kW · Feeder: {activeMeterModal.profile.area_zone}
                    </p>
                  </div>
                  <button className="modal-close-btn" onClick={() => setActiveMeterModal(null)}>✕</button>
                </div>

                <div className="modal-body">
                  <div style={{ marginBottom: "16px", display: "flex", gap: "10px", alignItems: "center" }}>
                    <span style={{ fontSize: "12px", color: "var(--text-muted)" }}>Current Audit Status:</span>
                    {formatAuditBadge(activeMeterModal.profile.current_audit_status)}
                  </div>

                  <h3 style={{ marginBottom: "14px", fontSize: "14px", color: "var(--text-muted)" }}>Audit & Telemetry History</h3>
                  {activeMeterModal.readings.length === 0 ? (
                    <EmptyState title="No telemetry logged" description="This meter has not reported telemetry readings yet." />
                  ) : (
                    <div className="table-container">
                      <table className="data-table">
                        <thead>
                          <tr>
                            <th>Reading Date</th>
                            <th>Prev (kWh)</th>
                            <th>Curr (kWh)</th>
                            <th>Variance</th>
                            <th>Risk Score</th>
                            <th>Prediction</th>
                            <th>Audit Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activeMeterModal.readings.map((r) => (
                            <tr key={r.id}>
                              <td>{formatDate(r.created_at)}</td>
                              <td>{formatNumber(r.previous_month_kwh)}</td>
                              <td>{formatNumber(r.current_month_kwh)}</td>
                              <td>{formatNumber(getUsageChange(r))}%</td>
                              <td><RiskBadge score={getRiskScore(r)} /></td>
                              <td><StatusBadge status={getPrediction(r)} /></td>
                              <td>{formatAuditBadge(r.audit_status)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;