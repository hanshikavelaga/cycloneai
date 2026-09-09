import React, { useState, useEffect, useRef } from 'react';
import { 
  Wind, Shield, Radio, AlertTriangle, Play, Pause, 
  RotateCcw, Upload, Activity, FileText, Database, 
  MapPin, RefreshCw, Layers, Sliders, ChevronRight
} from 'lucide-react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import L from 'leaflet';

const API_BASE = "http://127.0.0.1:8000/api";

const istTimeFormatter = new Intl.DateTimeFormat("en-IN", {
  timeZone: "Asia/Kolkata",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false
});

export default function App() {
  const [activeTab, setActiveTab] = useState("tracker");
  
  // Database States
  const [activeCyclones, setActiveCyclones] = useState([]);
  const [selectedCycloneId, setSelectedCycloneId] = useState("");
  const [cycloneDetail, setCycloneDetail] = useState(null);
  const [genesisPredictions, setGenesisPredictions] = useState([]);
  const [systemAlerts, setSystemAlerts] = useState([]);
  
  // Replay Demo States
  const [historicalCyclones, setHistoricalCyclones] = useState([]);
  const [replayCycloneId, setReplayCycloneId] = useState("");
  const [replaySteps, setReplaySteps] = useState([]);
  const [currentStepIdx, setCurrentStepIdx] = useState(0);
  const [isPlayingReplay, setIsPlayingReplay] = useState(false);
  const [replayData, setReplayData] = useState(null);
  
  // Diagnostic Upload States
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadResult, setUploadResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  
  // Image Fusion States
  const [fusionAlpha, setFusionAlpha] = useState(0.6);
  const [isFusionLoading, setIsFusionLoading] = useState(false);
  
  // Live Sync Status States
  const [systemStatus, setSystemStatus] = useState({
    gdacs_status: "DISCONNECTED",
    gdacs_last_success: null,
    gdacs_last_error: null,
    gdacs_records_received: 0,
    gdacs_records_stored: 0,
    noaa_status: "DISCONNECTED",
    noaa_last_success: null,
    noaa_last_error: null,
    noaa_records_received: 0,
    noaa_records_stored: 0,
    last_sync: null,
    next_sync: null,
    logs: []
  });
  const [isSyncingNow, setIsSyncingNow] = useState(false);
  const [currentIstTime, setCurrentIstTime] = useState("");
  
  // Map References
  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const layersGroupRef = useRef(null);

  // Load initial database profiles
  useEffect(() => {
    fetchActiveCyclones();
    fetchHistoricalCyclones();
    fetchGenesisPredictions();
    fetchAlerts();
    fetchSystemStatus();
    
    // Status and alert polling interval (10 seconds)
    const statusInterval = setInterval(() => {
      fetchSystemStatus();
      fetchAlerts();
    }, 10000);
    return () => clearInterval(statusInterval);
  }, []);

  // Ticking 1-second clock interval hook for IST
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setCurrentIstTime(istTimeFormatter.format(now));
    };
    updateClock();
    const clockInterval = setInterval(updateClock, 1000);
    return () => clearInterval(clockInterval);
  }, []);

  // Sync active cyclone selection change
  useEffect(() => {
    if (selectedCycloneId) {
      fetchCycloneDetail(selectedCycloneId);
    }
  }, [selectedCycloneId]);

  // Sync historical replay storm selection change
  useEffect(() => {
    if (replayCycloneId) {
      fetchReplaySteps(replayCycloneId);
    }
  }, [replayCycloneId]);

  // Sync replay step index slider change
  useEffect(() => {
    if (replayCycloneId && replaySteps.length > 0) {
      fetchReplayStepState(replayCycloneId, currentStepIdx);
    }
  }, [currentStepIdx, replaySteps]);

  // Handle Replay play timer
  useEffect(() => {
    let timer = null;
    if (isPlayingReplay) {
      timer = setInterval(() => {
        setCurrentStepIdx((prev) => {
          if (prev >= replaySteps.length - 1) {
            setIsPlayingReplay(false);
            return prev;
          }
          return prev + 1;
        });
      }, 3000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [isPlayingReplay, replaySteps]);

  // Re-render Leaflet Map markings whenever tab, active detail, or replay updates
  useEffect(() => {
    initializeMap();
    renderMapMarkers();
  }, [activeTab, cycloneDetail, replayData, genesisPredictions]);

  // API Fetch Functions
  const fetchActiveCyclones = async () => {
    try {
      const res = await fetch(`${API_BASE}/cyclones/active`);
      const data = await res.json();

      setActiveCyclones(data);

      if (data.length > 0) {
        setSelectedCycloneId(data[0].id);
      } else {
        // No active cyclones: clear any stale selection/details.
        setSelectedCycloneId("");
        setCycloneDetail(null);
      }
    } catch (e) {
      console.error("Error fetching active cyclones:", e);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/system/status`);
      const data = await res.json();
      setSystemStatus(data);
    } catch (e) {
      console.error("Error fetching system status:", e);
    }
  };

  const triggerManualSync = async () => {
    setIsSyncingNow(true);
    try {
      const res = await fetch(`${API_BASE}/system/sync`, { method: "POST" });
      const data = await res.json();
      if (data.status === "success") {
        await fetchSystemStatus();
        await fetchActiveCyclones();
        await fetchGenesisPredictions();
        await fetchAlerts();
      }
    } catch (e) {
      console.error("Error executing manual synchronization:", e);
    } finally {
      setIsSyncingNow(false);
    }
  };

  const formatToIst = (utcString) => {
    if (!utcString) return "NEVER";
    try {
      // Convert "YYYY-MM-DD HH:MM:SS UTC" to ISO "YYYY-MM-DDTHH:MM:SSZ"
      const isoString = utcString.replace(" UTC", "").replace(" ", "T") + "Z";
      const dateObj = new Date(isoString);
      if (isNaN(dateObj.getTime())) return utcString;
      return istTimeFormatter.format(dateObj);
    } catch (e) {
      return utcString;
    }
  };

  const convertLogToIst = (logLine) => {
    const match = logLine.match(/^\[(\d{2}):(\d{2}):(\d{2}) UTC\](.*)/);
    if (match) {
      const [_, hh, mm, ss, rest] = match;
      const today = new Date();
      // Safely construct UTC date to let Intl map the timezone rollover
      const utcDate = new Date(Date.UTC(
        today.getUTCFullYear(),
        today.getUTCMonth(),
        today.getUTCDate(),
        parseInt(hh),
        parseInt(mm),
        parseInt(ss)
      ));
      return `[${istTimeFormatter.format(utcDate)} IST]${rest}`;
    }
    return logLine;
  };

  const fetchHistoricalCyclones = async () => {
    try {
      const res = await fetch(`${API_BASE}/cyclones/historical`);
      const data = await res.json();
      setHistoricalCyclones(data);
      if (data.length > 0) {
        setReplayCycloneId(data[0].id);
      }
    } catch (e) {
      console.error("Error fetching historical cyclones:", e);
    }
  };

  const fetchCycloneDetail = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/cyclones/${id}`);
      const data = await res.json();
      setCycloneDetail(data);
    } catch (e) {
      console.error("Error fetching cyclone detail:", e);
    }
  };

  const fetchGenesisPredictions = async () => {
    try {
      const res = await fetch(`${API_BASE}/genesis`);
      const data = await res.json();
      setGenesisPredictions(data);
    } catch (e) {
      console.error("Error fetching genesis forecasts:", e);
    }
  };

  const fetchAlerts = async () => {
    try {
      const res = await fetch(`${API_BASE}/alerts`);
      const data = await res.json();
      setSystemAlerts(data);
    } catch (e) {
      console.error("Error fetching alerts:", e);
    }
  };

  const triggerForecastRun = async () => {
    if (!selectedCycloneId) return;
    try {
      await fetch(`${API_BASE}/forecast/${selectedCycloneId}`, { method: "POST" });
      fetchCycloneDetail(selectedCycloneId); // Refresh details
    } catch (e) {
      console.error("Error running forecast:", e);
    }
  };

  const fetchReplaySteps = async (id) => {
    try {
      const res = await fetch(`${API_BASE}/replay/${id}/steps`);
      const data = await res.json();
      setReplaySteps(data);
      setCurrentStepIdx(0);
      setIsPlayingReplay(false);
    } catch (e) {
      console.error("Error fetching replay steps:", e);
    }
  };

  const fetchReplayStepState = async (id, idx) => {
    try {
      const res = await fetch(`${API_BASE}/replay/${id}/step/${idx}`);
      const data = await res.json();
      setReplayData(data);
    } catch (e) {
      console.error("Error fetching replay step details:", e);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setUploadedFile(file);
    setUploadResult(null);
    setIsUploading(true);
    
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict/image`, {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      setUploadResult(data);
    } catch (err) {
      console.error("Upload error:", err);
    } finally {
      setIsUploading(false);
    }
  };

  // Leaflet Map Controller Setup
  const initializeMap = () => {
    if (!mapContainerRef.current) return;
    
    if (!mapRef.current) {
      // Mapbox/CartoDB Dark Theme tiles (SIH professional styling)
      mapRef.current = L.map(mapContainerRef.current).setView([15.0, 83.0], 5);
      L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        maxZoom: 19
      }).addTo(mapRef.current);
      
      layersGroupRef.current = L.layerGroup().addTo(mapRef.current);
    }
  };

  const renderMapMarkers = () => {
    if (!layersGroupRef.current || !mapRef.current) return;
    
    // Reset layers
    layersGroupRef.current.clearLayers();
    
    let pathCoordinates = [];
    let forecastCoordinates = [];
    let focusedCoords = null;

    if (activeTab === "tracker" && cycloneDetail) {
      // Draw historical path polyline
      const obsSorted = [...cycloneDetail.observations].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp));
      pathCoordinates = obsSorted.map(o => [o.latitude, o.longitude]);
      
      obsSorted.forEach((obs, index) => {
        const isLatest = index === obsSorted.length - 1;
        if (isLatest) focusedCoords = [obs.latitude, obs.longitude];
        
        // Define color indicators based on classification pattern
        const color = obs.pattern_type === "Eye" ? "#ef4444" : 
                      obs.pattern_type === "CDO" ? "#f97316" : 
                      obs.pattern_type === "Curved Band" ? "#eab308" : "#22c55e";
                      
        const marker = L.circleMarker([obs.latitude, obs.longitude], {
          radius: isLatest ? 8 : 5,
          fillColor: color,
          color: isLatest ? "#ffffff" : color,
          weight: isLatest ? 2 : 1,
          fillOpacity: 0.8
        }).bindPopup(`
          <strong>Observed Position</strong><br/>
          Time: ${new Date(obs.timestamp).toLocaleString()}<br/>
          Pattern: ${obs.pattern_type}<br/>
          Wind Speed: ${obs.wind_speed} knots<br/>
          Pressure: ${obs.pressure} hPa
        `);
        
        layersGroupRef.current.addLayer(marker);
      });

      // Draw forecast paths
      forecastCoordinates = cycloneDetail.predictions.map(p => [p.latitude, p.longitude]);
      cycloneDetail.predictions.forEach((pred) => {
        const marker = L.circleMarker([pred.latitude, pred.longitude], {
          radius: 5,
          fillColor: "#38bdf8",
          color: "#38bdf8",
          weight: 1,
          fillOpacity: 0.7
        }).bindPopup(`
          <strong>AI Track Prediction</strong><br/>
          Forecast Time: ${new Date(pred.forecast_time).toLocaleString()}<br/>
          Wind Speed: ${pred.wind_speed} knots<br/>
          Pressure: ${pred.pressure} hPa<br/>
          Confidence: ${Math.round(pred.confidence * 100)}%
        `);
        
        // Draw uncertainty bounds circle around prediction
        const errorMargin = (1 - pred.confidence) * 150000; // error envelope in meters
        const uncertaintyRange = L.circle([pred.latitude, pred.longitude], {
          radius: errorMargin,
          color: "#38bdf8",
          weight: 1,
          dashArray: "3, 6",
          fillColor: "#38bdf8",
          fillOpacity: 0.05
        });
        
        layersGroupRef.current.addLayer(marker);
        layersGroupRef.current.addLayer(uncertaintyRange);
      });

    } else if (activeTab === "replay" && replayData) {
      // Replay active history path up to step
      const obsSorted = [...replayData.history].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp));
      pathCoordinates = obsSorted.map(o => [o.latitude, o.longitude]);
      
      obsSorted.forEach((obs, index) => {
        const isLatest = index === obsSorted.length - 1;
        if (isLatest) focusedCoords = [obs.latitude, obs.longitude];
        
        const color = obs.pattern_type === "Eye" ? "#ef4444" : 
                      obs.pattern_type === "CDO" ? "#f97316" : 
                      obs.pattern_type === "Curved Band" ? "#eab308" : "#22c55e";
                      
        const marker = L.circleMarker([obs.latitude, obs.longitude], {
          radius: isLatest ? 8 : 5,
          fillColor: color,
          color: isLatest ? "#ffffff" : color,
          weight: isLatest ? 2 : 1,
          fillOpacity: 0.8
        });
        layersGroupRef.current.addLayer(marker);
      });

      // Forecast lines on-the-fly from that step
      forecastCoordinates = replayData.forecasts.map(p => [p.latitude, p.longitude]);
      replayData.forecasts.forEach((pred) => {
        const marker = L.circleMarker([pred.latitude, pred.longitude], {
          radius: 5,
          fillColor: "#a855f7",  // purple color to differentiate replay forecasts
          color: "#a855f7",
          weight: 1,
          fillOpacity: 0.7
        }).bindPopup(`
          <strong>Step-wise AI Forecast</strong><br/>
          Forecast Time: ${new Date(pred.forecast_time).toLocaleString()}<br/>
          Wind Speed: ${pred.wind_speed} knots<br/>
          Pressure: ${pred.pressure} hPa<br/>
          Confidence: ${Math.round(pred.confidence * 100)}%
        `);
        
        const errorMargin = (1 - pred.confidence) * 150000;
        const uncertaintyRange = L.circle([pred.latitude, pred.longitude], {
          radius: errorMargin,
          color: "#a855f7",
          weight: 1,
          dashArray: "3, 6",
          fillColor: "#a855f7",
          fillOpacity: 0.05
        });
        
        layersGroupRef.current.addLayer(marker);
        layersGroupRef.current.addLayer(uncertaintyRange);
      });
    }

    // Render path Polylines
    if (pathCoordinates.length > 0) {
      const pathLine = L.polyline(pathCoordinates, { color: "#ffffff", weight: 2 });
      layersGroupRef.current.addLayer(pathLine);
    }
      if (forecastCoordinates.length > 0) {
      // Connect observation tail to forecast start when observations exist.
      // Otherwise render the forecast coordinates directly.
      const combinedForecast =
        pathCoordinates.length > 0
          ? [pathCoordinates[pathCoordinates.length - 1], ...forecastCoordinates]
          : forecastCoordinates;

      const forecastLine = L.polyline(combinedForecast, {
        color: activeTab === "replay" ? "#a855f7" : "#38bdf8",
        weight: 2,
        dashArray: "4, 8"
      });

      layersGroupRef.current.addLayer(forecastLine);
    }

    // Render Genesis watch coordinates (pulse points in ocean)
    if (genesisPredictions.length > 0) {
      genesisPredictions.forEach((gp) => {
        // Mock coordinates for basins
        const isBoB = gp.region.includes("Bay of Bengal");
        const lat = isBoB ? 14.5 : 9.0;
        const lon = isBoB ? 87.0 : 66.0;
        
        if (gp.probability > 40.0) {
          const genesisPulse = L.circle([lat, lon], {
            radius: 80000,
            color: gp.probability > 75.0 ? "#ef4444" : "#f97316",
            fillColor: gp.probability > 75.0 ? "#ef4444" : "#f97316",
            fillOpacity: 0.15,
            weight: 1
          }).bindPopup(`
            <strong>Genesis Watch Active</strong><br/>
            Region: ${gp.region}<br/>
            Formation Prob: ${gp.probability}%<br/>
            Expected: ${gp.expected_formation_time ? new Date(gp.expected_formation_time).toLocaleString() : "TBD"}
          `);
          layersGroupRef.current.addLayer(genesisPulse);
        }
      });
    }

    // Pan map to center focus
    if (focusedCoords) {
      mapRef.current.panTo(focusedCoords);
    }
  };

  // Helper: format dates for Recharts
  const prepareChartData = () => {
    let data = [];
    if (activeTab === "tracker" && cycloneDetail) {
      const obsData = [...cycloneDetail.observations].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp)).map(o => ({
        time: new Date(o.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' }),
        wind: o.wind_speed,
        pressure: o.pressure,
        type: 'Observed'
      }));
      const predData = [...cycloneDetail.predictions].sort((a,b) => new Date(a.forecast_time) - new Date(b.forecast_time)).map(p => ({
        time: new Date(p.forecast_time).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' }),
        wind: p.wind_speed,
        pressure: p.pressure,
        type: 'Forecast'
      }));
      data = [...obsData, ...predData];
    } else if (activeTab === "replay" && replayData) {
      const obsData = [...replayData.history].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp)).map(o => ({
        time: new Date(o.timestamp).toLocaleDateString([], { month: 'short', day: 'numeric' }),
        wind: o.wind_speed,
        pressure: o.pressure,
        type: 'Observed'
      }));
      const predData = [...replayData.forecasts].sort((a,b) => new Date(a.forecast_time) - new Date(b.forecast_time)).map(p => ({
        time: new Date(p.forecast_time).toLocaleDateString([], { month: 'short', day: 'numeric', hour: '2-digit' }),
        wind: p.wind_speed,
        pressure: p.pressure,
        type: 'Forecast'
      }));
      data = [...obsData, ...predData];
    }
    return data;
  };

  const getLatestObs = () => {
    if (activeTab === "tracker" && cycloneDetail && cycloneDetail.observations.length > 0) {
      return [...cycloneDetail.observations].sort((a,b) => new Date(a.timestamp) - new Date(b.timestamp))[cycloneDetail.observations.length - 1];
    } else if (activeTab === "replay" && replayData) {
      return replayData.latest_observation;
    }
    return null;
  };

  const latestObs = getLatestObs();
  const activeCyclone = activeTab === "tracker" ? cycloneDetail : (replayData ? replayData.cyclone : null);

  // Alert display helpers. These accept the fields returned by the backend
  // while avoiding any hard-coded/demo alert data.
  const getAlertSeverity = (alert) =>
    String(alert?.severity || alert?.level || alert?.alert_level || "INFO").toUpperCase();

  const getAlertTitle = (alert) =>
    alert?.alert_type || alert?.type || alert?.title || "System Alert";

  const getAlertMessage = (alert) =>
    alert?.message || alert?.description || alert?.details || "Alert condition reported by the monitoring system.";

  const getAlertTime = (alert) =>
    alert?.timestamp || alert?.created_at || alert?.time || alert?.issued_at || null;

  const getAlertCyclone = (alert) =>
    alert?.cyclone_name || alert?.cyclone_id || alert?.storm_name || null;

  const formatAlertTime = (value) => {
    if (!value) return "Time unavailable";
    const date = new Date(value);
    if (isNaN(date.getTime())) return String(value);
    return `${date.toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      timeZone: "Asia/Kolkata"
    })}, ${istTimeFormatter.format(date)} IST`;
  };

  const getAlertClasses = (severity) => {
    if (severity === "CRITICAL" || severity === "HIGH" || severity === "RED") {
      return {
        container: "border-red-900/60 bg-red-950/20",
        icon: "text-red-400",
        badge: "bg-red-950 border-red-800 text-red-400"
      };
    }

    if (severity === "MEDIUM" || severity === "MODERATE" || severity === "ORANGE") {
      return {
        container: "border-orange-900/60 bg-orange-950/20",
        icon: "text-orange-400",
        badge: "bg-orange-950 border-orange-800 text-orange-400"
      };
    }

    return {
      container: "border-sky-900/60 bg-sky-950/20",
      icon: "text-sky-400",
      badge: "bg-sky-950 border-sky-800 text-sky-400"
    };
  };

  const displayedAlerts = [...systemAlerts]
    .sort((a, b) => {
      const aTime = new Date(getAlertTime(a) || 0).getTime();
      const bTime = new Date(getAlertTime(b) || 0).getTime();
      return bTime - aTime;
    })
    .slice(0, 5);

  return (
    <div className="flex flex-col min-h-screen bg-slate-900 text-slate-100 font-sans">
      
      {/* --- TOP APPLICATION NAVIGATION BAR --- */}
      <header className="flex justify-between items-center px-6 py-4 bg-slate-950 border-b border-slate-800 shadow-md">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-sky-600 rounded-lg text-white animate-pulse">
            <Wind size={24} />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">CycloneAI</h1>
            <p className="text-xs text-sky-400 font-semibold tracking-wider uppercase">Tropical Cyclone Intelligence & Forecasting Platform</p>
          </div>
        </div>

        {/* Global Warnings marquee - System Status Bar */}
        <div className="hidden lg:flex flex-col gap-1.5 bg-slate-950 border border-slate-800 px-5 py-2 rounded-xl font-mono text-[10px] tracking-wider text-slate-400">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5 font-bold text-sky-400">
              <span>🕐 IST {currentIstTime || "00:00:00"}</span>
            </div>
            <span className="text-slate-800">|</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${systemStatus.gdacs_status === "CONNECTED" ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`}></span>
              <span>GDACS: {systemStatus.gdacs_status}</span>
            </div>
            <span className="text-slate-800">|</span>
            <div className="flex items-center gap-1.5">
              <span className={`w-1.5 h-1.5 rounded-full ${systemStatus.noaa_status === "CONNECTED" ? "bg-emerald-500 animate-pulse" : "bg-red-500"}`}></span>
              <span>NOAA: {systemStatus.noaa_status}</span>
            </div>
            <span className="text-slate-800">|</span>
            <button 
              onClick={triggerManualSync}
              disabled={isSyncingNow}
              className="flex items-center gap-1 text-sky-400 hover:text-sky-300 font-semibold uppercase disabled:opacity-50"
            >
              <RefreshCw size={9} className={isSyncingNow ? "animate-spin" : ""} />
              {isSyncingNow ? "Syncing..." : "Sync Now"}
            </button>
          </div>
          
          <div className="flex items-center gap-3 border-t border-slate-900 pt-1 text-slate-500">
            <span>LAST SYNC: {formatToIst(systemStatus.last_sync)} IST</span>
            <span>•</span>
            <span>NEXT SYNC: {formatToIst(systemStatus.next_sync)} IST</span>
            <span>•</span>
            <span>GDACS RX: {systemStatus.gdacs_records_stored}/{systemStatus.gdacs_records_received}</span>
            <span>•</span>
            <span>NOAA RX: {systemStatus.noaa_records_stored}/{systemStatus.noaa_records_received}</span>
          </div>
        </div>

        <div className="flex gap-1.5">
          <button 
            onClick={() => setActiveTab("tracker")} 
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${activeTab === "tracker" ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}
          >
            Live Monitor
          </button>
          <button 
            onClick={() => setActiveTab("fusion")} 
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${activeTab === "fusion" ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}
          >
            Satellite & Fusion
          </button>
          <button 
            onClick={() => setActiveTab("genesis")} 
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${activeTab === "genesis" ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}
          >
            Genesis Watch
          </button>
          <button 
            onClick={() => setActiveTab("replay")} 
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${activeTab === "replay" ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}
          >
            💡 Replay Simulator
          </button>
          <button 
            onClick={() => setActiveTab("diagnostic")} 
            className={`px-4 py-2 text-xs font-semibold rounded-md transition-all ${activeTab === "diagnostic" ? "bg-sky-600 text-white" : "text-slate-400 hover:bg-slate-800 hover:text-white"}`}
          >
            AI Diagnostics
          </button>
        </div>
      </header>

      {/* --- MAIN GRID CONTAINER --- */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 overflow-hidden">
        
        {/* --- LEFT SECTION: MAP VIEW & MAP CONTROLS --- */}
        <section className="lg:col-span-8 flex flex-col border-r border-slate-800 bg-slate-900 relative">
          
          {/* Dashboard map container */}
          <div className="flex-1 w-full relative min-h-[400px]">
            <div ref={mapContainerRef} className="absolute inset-0 z-10" />
            
            {/* Legend Overlay on Map */}
            <div className="absolute bottom-4 left-4 z-20 bg-slate-950/90 border border-slate-800 p-4 rounded-lg text-xs space-y-2">
              <span className="font-bold text-slate-400 uppercase tracking-wider block mb-1">Observation Category</span>
              <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-[#ef4444] inline-block"></span><span>Eye Pattern</span></div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-[#f97316] inline-block"></span><span>CDO</span></div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-[#eab308] inline-block"></span><span>Curved Band</span></div>
                <div className="flex items-center gap-2"><span className="w-3 h-3 rounded-full bg-[#22c55e] inline-block"></span><span>Shear / Weak</span></div>
              </div>
              <div className="border-t border-slate-800 pt-2 mt-2">
                <div className="flex items-center gap-2"><span className="w-4 border-t-2 border-dashed border-[#38bdf8] inline-block"></span><span>AI Predicted Path</span></div>
              </div>
            </div>
          </div>

          {/* Live Ingestion Log Console */}
          <div className="bg-slate-950 border-t border-slate-800 flex flex-col h-48 z-20">
            <div className="flex justify-between items-center px-4 py-2 border-b border-slate-800 bg-slate-950">
              <div className="flex items-center gap-2">
                <Database size={12} className="text-sky-400 animate-pulse" />
                <span className="text-[11px] font-bold text-slate-300 uppercase tracking-wider">Live Ingestion Polling Logs</span>
              </div>
              <div className="text-[10px] text-slate-500 font-mono">
                Update interval: 5m | Next sync: {systemStatus.next_sync ? formatToIst(systemStatus.next_sync) + " IST" : "N/A"}
              </div>
            </div>
            <div className="flex-1 p-3 overflow-y-auto font-mono text-[11px] bg-[#030712] text-emerald-400 space-y-1 scrollbar-thin">
              {systemStatus.logs && systemStatus.logs.length > 0 ? (
                systemStatus.logs.map((log, index) => (
                  <div key={index} className="flex gap-2">
                    <span className="text-slate-600 select-none">&gt;</span>
                    <span>{convertLogToIst(log)}</span>
                  </div>
                ))
              ) : (
                <div className="text-slate-600 italic">No synchronization logs recorded yet.</div>
              )}
            </div>
          </div>

          {/* Replay Timeline Slider Tray */}
          {activeTab === "replay" && replaySteps.length > 0 && (
            <div className="bg-slate-950 border-t border-slate-800 px-6 py-4 flex flex-col md:flex-row items-center gap-4 z-20">
              <div className="flex items-center gap-2">
                <button 
                  onClick={() => setIsPlayingReplay(!isPlayingReplay)} 
                  className={`p-2.5 rounded-full transition-colors ${isPlayingReplay ? "bg-red-600 text-white" : "bg-sky-600 hover:bg-sky-500 text-white"}`}
                >
                  {isPlayingReplay ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <button 
                  onClick={() => { setCurrentStepIdx(0); setIsPlayingReplay(false); }} 
                  className="p-2 bg-slate-800 hover:bg-slate-700 rounded-full text-slate-300"
                >
                  <RotateCcw size={16} />
                </button>
              </div>

              <div className="flex-1 w-full">
                <div className="flex justify-between text-xs text-slate-400 mb-1.5">
                  <span>Start (T-0)</span>
                  <span className="font-semibold text-sky-400">
                    Step {currentStepIdx + 1} of {replaySteps.length} | Observation Time: {replaySteps[currentStepIdx] ? new Date(replaySteps[currentStepIdx].timestamp).toLocaleString() : ""}
                  </span>
                  <span>Landfall / Peak</span>
                </div>
                <input 
                  type="range"
                  min="0"
                  max={replaySteps.length - 1}
                  value={currentStepIdx}
                  onChange={(e) => { setCurrentStepIdx(parseInt(e.target.value)); setIsPlayingReplay(false); }}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                />
              </div>
            </div>
          )}
        </section>

        {/* --- RIGHT SECTION: AI CLASSIFICATION & METRICS TERMINAL --- */}
        <section className="lg:col-span-4 flex flex-col bg-slate-950 overflow-y-auto max-h-screen">
          
          {/* TAB 1: LIVE MONITOR DETECTOR */}
          {activeTab === "tracker" && (
            <div className="p-6 space-y-6">
              <div className="flex justify-between items-start">
                <div>
                  <h2 className="text-lg font-bold text-slate-100">Live Warning Monitor</h2>
                  <p className="text-xs text-slate-400">Near-real-time GDACS / NOAA monitoring</p>
                </div>
                <button 
                  onClick={fetchActiveCyclones}
                  className="p-1.5 bg-slate-800 hover:bg-slate-700 rounded text-slate-400"
                  title="Refresh Active Feed"
                >
                  <RefreshCw size={14} />
                </button>
              </div>

              {/* System Alerts Panel */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <AlertTriangle
                      size={14}
                      className={displayedAlerts.length > 0 ? "text-orange-400" : "text-slate-500"}
                    />
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                      System Alerts
                    </span>
                  </div>
                  <span className="text-[10px] font-mono text-slate-500">
                    {displayedAlerts.length} recent
                  </span>
                </div>

                {displayedAlerts.length > 0 ? (
                  <div className="space-y-2">
                    {displayedAlerts.map((alert, index) => {
                      const severity = getAlertSeverity(alert);
                      const classes = getAlertClasses(severity);
                      const cycloneReference = getAlertCyclone(alert);

                      return (
                        <div
                          key={alert?.id ?? `${getAlertTime(alert) ?? "alert"}-${index}`}
                          className={`p-3 border rounded-lg ${classes.container} space-y-2`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-2 min-w-0">
                              <AlertTriangle size={14} className={`${classes.icon} mt-0.5 shrink-0`} />
                              <div className="min-w-0">
                                <span className="text-xs font-bold text-slate-200 block truncate">
                                  {getAlertTitle(alert)}
                                </span>
                                {cycloneReference && (
                                  <span className="text-[10px] text-slate-500 block mt-0.5 truncate">
                                    Cyclone: {cycloneReference}
                                  </span>
                                )}
                              </div>
                            </div>

                            <span className={`px-1.5 py-0.5 text-[9px] font-black border rounded shrink-0 ${classes.badge}`}>
                              {severity}
                            </span>
                          </div>

                          <p className="text-[11px] text-slate-300 leading-relaxed">
                            {getAlertMessage(alert)}
                          </p>

                          <div className="text-[9px] text-slate-500 font-mono">
                            {formatAlertTime(getAlertTime(alert))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div className="p-3 border border-slate-800/70 bg-slate-900/40 rounded-lg text-[11px] text-slate-500">
                    No alerts reported by the monitoring system.
                  </div>
                )}
              </div>

              {/* Storm Selector */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">Select Monitored Cyclone</label>
                <select 
                  value={selectedCycloneId} 
                  onChange={(e) => setSelectedCycloneId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 p-2.5 rounded text-sm text-slate-100 outline-none focus:border-sky-500"
                >
                  {activeCyclones.map(c => (
                    <option key={c.id} value={c.id}>{c.name} ({c.basin})</option>
                  ))}
                  {activeCyclones.length === 0 && <option value="">No Active Cyclones</option>}
                </select>
              </div>

              {activeCyclone ? (
                <>
                  {/* Cyclone Telemetry Card */}
                  <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-xs text-sky-400 tracking-wider font-bold uppercase">{activeCyclone.id}</span>
                      <span className="px-2 py-0.5 text-[10px] font-bold uppercase rounded bg-red-950 border border-red-800 text-red-400 animate-pulse">
                        {activeCyclone.status}
                      </span>
                    </div>
                    <div className="text-2xl font-black text-slate-100">{activeCyclone.name}</div>
                    
                    <div className="grid grid-cols-2 gap-4 border-t border-slate-800 pt-4 text-sm">
                      <div>
                        <span className="text-xs text-slate-400 block">Basin</span>
                        <span className="font-semibold text-slate-200">{activeCyclone.basin}</span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Current Category</span>
                        <span className="font-semibold text-slate-200">
                          {latestObs ? (latestObs.wind_speed > 64 ? "Severe Cyclonic Storm" : "Cyclonic Storm") : "TBD"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* AI Prediction Details */}
                  {latestObs && (
                    <div className="space-y-3">
                      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider">AI Automated Diagnosis</h3>
                      <div className="grid grid-cols-3 gap-2">
                        <div className="p-3 bg-slate-900/60 border border-slate-800/80 rounded text-center">
                          <span className="text-[10px] text-slate-400 block">T-Number</span>
                          <span className="text-lg font-bold text-sky-400">{latestObs.dvorak_t_number || "T1.5"}</span>
                        </div>
                        <div className="p-3 bg-slate-900/60 border border-slate-800/80 rounded text-center">
                          <span className="text-[10px] text-slate-400 block">Wind (Knots)</span>
                          <span className="text-lg font-bold text-sky-400">{latestObs.wind_speed}</span>
                        </div>
                        <div className="p-3 bg-slate-900/60 border border-slate-800/80 rounded text-center">
                          <span className="text-[10px] text-slate-400 block">Pressure (hPa)</span>
                          <span className="text-lg font-bold text-sky-400">{latestObs.pressure}</span>
                        </div>
                      </div>

                      {/* AI Pattern Confidence Indicator */}
                      <div className="p-4 bg-slate-900/40 border border-slate-800/40 rounded-lg space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-400">Classified Pattern:</span>
                          <span className="font-bold text-slate-200">{latestObs.pattern_type}</span>
                        </div>
                        <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                          <div 
                            className="bg-sky-500 h-2 rounded-full transition-all duration-500" 
                            style={{ width: `${(latestObs.dvorak_t_number / 8.0) * 100 || 20}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Run Forecast Button */}
                  <button 
                    onClick={triggerForecastRun}
                    className="w-full py-3 bg-sky-600 hover:bg-sky-500 font-bold text-sm text-white rounded-lg transition-colors flex justify-center items-center gap-2"
                  >
                    <Activity size={16} />
                    Run AI Trajectory Forecast
                  </button>

                  {/* Recharts Intensity Plot */}
                  {cycloneDetail && cycloneDetail.observations.length > 0 && (
                    <div className="space-y-2">
                      <h3 className="text-sm font-bold text-slate-300 uppercase tracking-wider">Intensity History & Predictions</h3>
                      <div className="h-44 w-full bg-slate-900/60 border border-slate-800/80 p-2 rounded-lg">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={prepareChartData()}>
                            <XAxis dataKey="time" tick={{ fontSize: 9 }} stroke="#64748b" />
                            <YAxis yAxisId="left" tick={{ fontSize: 9 }} stroke="#64748b" />
                            <YAxis yAxisId="right" orientation="right" domain={[920, 1020]} tick={{ fontSize: 9 }} stroke="#64748b" />
                            <Tooltip contentStyle={{ backgroundColor: "#020617", border: "1px solid #1e293b" }} />
                            <Line yAxisId="left" type="monotone" dataKey="wind" name="Wind (kts)" stroke="#38bdf8" strokeWidth={2} activeDot={{ r: 6 }} />
                            <Line yAxisId="right" type="monotone" dataKey="pressure" name="Pressure" stroke="#f43f5e" strokeWidth={1} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="py-12 text-center text-slate-500 text-sm">
                  No active cyclones detected in the North Indian Ocean region.
                </div>
              )}
            </div>
          )}

          {/* TAB 2: IMAGE FUSION & SPECTRAL ANALYSIS */}
          {activeTab === "fusion" && (
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-slate-100">Spectral Fusion Center</h2>
                <p className="text-xs text-slate-400">Combine INSAT-3D Infrared (IR) and Water Vapor (WV) channels</p>
              </div>

              {latestObs && latestObs.image_path ? (
                <>
                  {/* Visual Image Screen */}
                  <div className="w-full aspect-square bg-slate-900 border border-slate-800 rounded-lg overflow-hidden relative">
                    <img 
                      src={`${API_BASE}/satellite/fuse/${latestObs.image_path.split('\\').pop().split('/').pop()}?alpha=${fusionAlpha}`}
                      alt="Fused Satellite imagery"
                      className="w-full h-full object-cover"
                      key={fusionAlpha} // Force refresh on alpha change
                    />
                    
                    {/* Channel tags overlay */}
                    <div className="absolute top-2.5 left-2.5 bg-slate-950/80 border border-slate-800/80 px-2 py-1 rounded text-[9px] font-bold text-slate-400">
                      FUSION Composite: IR ({Math.round(fusionAlpha * 100)}%) + WV ({Math.round((1 - fusionAlpha) * 100)}%)
                    </div>
                  </div>

                  {/* Fusion Controls */}
                  <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg space-y-4">
                    <div className="flex justify-between text-xs font-bold text-slate-300">
                      <span className="flex items-center gap-1.5"><Sliders size={14} /> Adjust Fusion Ratio</span>
                      <span className="text-sky-400">{Math.round(fusionAlpha * 100)}% IR</span>
                    </div>
                    <div className="flex items-center gap-4">
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">More WV</span>
                      <input 
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={fusionAlpha}
                        onChange={(e) => setFusionAlpha(parseFloat(e.target.value))}
                        className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-500"
                      />
                      <span className="text-[10px] text-slate-400 uppercase tracking-wider font-bold">More IR</span>
                    </div>
                    <p className="text-[10px] text-slate-500 leading-normal">
                      Alpha blending maps thermal infrared pixels over moisture gradients. Cold convective cores appear in bright red/orange tones, while surrounding upper tropospheric circulation patterns appear in deep gray.
                    </p>
                  </div>
                </>
              ) : (
                <div className="py-12 text-center text-slate-500 text-sm">
                  Select an active storm in the Live Monitor tab to analyze satellite imagery.
                </div>
              )}
            </div>
          )}

          {/* TAB 3: GENESIS WATCH (WHAT FORMS NEXT?) */}
          {activeTab === "genesis" && (
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-slate-100">Genesis Watch Engine</h2>
                <p className="text-xs text-slate-400">Estimating probabilities for tropical disturbance organization</p>
              </div>

              {/* Active alerts feed */}
              <div className="space-y-3">
                <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Oceanic Basin Status</span>
                {genesisPredictions.map((gp) => {
                  const isHigh = gp.probability > 75.0;
                  const isModerate = gp.probability > 40.0;
                  const color = isHigh ? "border-red-900/50 bg-red-950/20 text-red-400" :
                                isModerate ? "border-orange-950/50 bg-orange-950/20 text-orange-400" :
                                "border-emerald-950/50 bg-emerald-950/20 text-emerald-400";
                  
                  return (
                    <div key={gp.id} className={`p-4 border rounded-lg ${color} space-y-2`}>
                      <div className="flex justify-between items-center">
                        <span className="font-bold text-sm text-slate-200">{gp.region}</span>
                        <span className="text-xs font-black">{gp.probability}% Probability</span>
                      </div>
                      <div className="flex justify-between text-[11px] text-slate-400">
                        <span>Confidence: {Math.round(gp.confidence * 100)}%</span>
                        <span>Estimated Formation: {gp.expected_formation_time ? new Date(gp.expected_formation_time).toLocaleDateString() : "No Immediate Threat"}</span>
                      </div>
                      
                      {/* Probability gauge */}
                      <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                        <div 
                          className={`h-1.5 rounded-full ${isHigh ? "bg-red-500" : isModerate ? "bg-orange-500" : "bg-emerald-500"}`} 
                          style={{ width: `${gp.probability}%` }}
                        />
                      </div>
                    </div>
                  );
                })}

                {genesisPredictions.length === 0 && (
                  <div className="py-8 text-center text-slate-600 text-sm">
                    No active basin observations found.
                  </div>
                )}
              </div>

              {/* Dvorak formation indicators explanation */}
              <div className="p-4 bg-slate-900/50 border border-slate-800 rounded-lg space-y-3 text-xs">
                <span className="font-bold text-slate-300 block flex items-center gap-1.5">
                  <Shield size={14} className="text-sky-400" /> AI Genesis Criteria
                </span>
                <p className="text-slate-400 leading-relaxed">
                  The Genesis watch model monitors convective anomalies, upper tropospheric divergence, sea surface temperatures, and Coriolis rotational momentum to predict cyclones before they are declared by meteorological departments.
                </p>
              </div>
            </div>
          )}

          {/* TAB 4: REPLAY SIMULATOR DEMO CONTROLLER */}
          {activeTab === "replay" && (
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-slate-100">💡 Historical Replay Simulator</h2>
                <p className="text-xs text-slate-400">Validate AI forecasts step-by-step against historical tracks</p>
              </div>

              {/* Historical Storm Selector */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">Select Historical Cyclone Case</label>
                <select 
                  value={replayCycloneId} 
                  onChange={(e) => setReplayCycloneId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 p-2.5 rounded text-sm text-slate-100 outline-none focus:border-sky-500"
                >
                  {historicalCyclones.map(c => (
                    <option key={c.id} value={c.id}>{c.name} ({c.basin} - {new Date(c.start_time).getFullYear()})</option>
                  ))}
                </select>
              </div>

              {replayData ? (
                <>
                  {/* Observation details at current replay frame */}
                  <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg space-y-3">
                    <div className="flex justify-between items-center text-xs">
                      <span className="font-bold text-purple-400">STEP {currentStepIdx + 1} OBSERVATION</span>
                      <span className="text-slate-400">
                        {new Date(replayData.latest_observation.timestamp).toLocaleString()}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-3 text-sm">
                      <div>
                        <span className="text-xs text-slate-400 block">Position</span>
                        <span className="font-semibold text-slate-200">
                          {replayData.latest_observation.latitude}°N, {replayData.latest_observation.longitude}°E
                        </span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Estimated Wind</span>
                        <span className="font-semibold text-slate-200">
                          {replayData.latest_observation.wind_speed} knots
                        </span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Dvorak Class</span>
                        <span className="font-semibold text-slate-200">
                          {replayData.latest_observation.pattern_type} (T{replayData.latest_observation.dvorak_t_number})
                        </span>
                      </div>
                      <div>
                        <span className="text-xs text-slate-400 block">Central Pressure</span>
                        <span className="font-semibold text-slate-200">
                          {replayData.latest_observation.pressure} hPa
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Explainable AI block */}
                  <div className="p-4 bg-slate-900/60 border border-slate-800/80 rounded-lg space-y-3">
                    <span className="text-xs font-black text-sky-400 uppercase tracking-wider block flex items-center gap-1.5">
                      <FileText size={14} /> Explainable AI Forecast Rationale
                    </span>
                    <ul className="text-xs text-slate-300 space-y-2 pl-4 list-disc">
                      {replayData.explanation.reasons.map((r, i) => (
                        <li key={i}>{r}</li>
                      ))}
                    </ul>
                  </div>

                  {/* Forecast trajectory values */}
                  <div className="space-y-2">
                    <span className="text-xs font-bold text-slate-400 uppercase tracking-wider block">Forecast Trajectory Details</span>
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      {replayData.forecasts.map((f, idx) => (
                        <div key={idx} className="flex justify-between items-center p-2.5 bg-slate-900/40 border border-slate-800/40 rounded text-xs">
                          <div>
                            <span className="font-bold text-slate-200">
                              {["+6h", "+12h", "+24h", "+48h"][idx] || "+?h"}
                            </span>
                            <span className="text-[10px] text-slate-500 block">
                              {f.latitude}°N, {f.longitude}°E
                            </span>
                          </div>
                          <div className="text-right">
                            <span className="font-semibold text-sky-400 block">{f.wind_speed} kts</span>
                            <span className="text-[10px] text-slate-400 block">Conf: {Math.round(f.confidence * 100)}%</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-12 text-center text-slate-500 text-sm">
                  Loading simulator steps...
                </div>
              )}
            </div>
          )}

          {/* TAB 5: MANUAL DIAGNOSTIC UPLOAD */}
          {activeTab === "diagnostic" && (
            <div className="p-6 space-y-6">
              <div>
                <h2 className="text-lg font-bold text-slate-100">AI Diagnostic Terminal</h2>
                <p className="text-xs text-slate-400">Upload custom imagery to classify structure and predict intensity on-demand</p>
              </div>

              {/* Upload input zone */}
              <div className="w-full p-8 border-2 border-dashed border-slate-800 hover:border-sky-500 bg-slate-900/40 rounded-lg flex flex-col items-center justify-center text-center cursor-pointer transition-colors relative">
                <input 
                  type="file" 
                  accept="image/*" 
                  onChange={handleFileUpload}
                  className="absolute inset-0 opacity-0 cursor-pointer"
                  disabled={isUploading}
                />
                <Upload size={32} className="text-sky-500 mb-2" />
                <span className="text-sm font-bold text-slate-200">Drag & Drop Satellite Image</span>
                <span className="text-xs text-slate-500 mt-1">Supports VIS, IR, or stacked PNG/JPEG files</span>
              </div>

              {isUploading && (
                <div className="flex items-center justify-center gap-2 text-xs text-sky-400 py-4">
                  <RefreshCw className="animate-spin" size={16} /> Running multi-task CNN inference...
                </div>
              )}

              {uploadedFile && !isUploading && (
                <div className="text-xs text-slate-400 px-1 truncate">
                  Uploaded file: <span className="font-semibold text-slate-200">{uploadedFile.name}</span>
                </div>
              )}

              {/* Upload Prediction Results */}
              {uploadResult && (
                <div className="p-4 bg-slate-900 border border-slate-800 rounded-lg space-y-4">
                  <div className="flex justify-between items-center border-b border-slate-800 pb-2.5">
                    <span className="font-bold text-sm text-slate-200">AI Diagnosis Outcome</span>
                    <span className="px-2 py-0.5 text-[10px] font-bold bg-sky-950 border border-sky-800 text-sky-400 rounded">
                      Inference OK
                    </span>
                  </div>

                  <div className="space-y-3">
                    <div className="flex justify-between text-xs">
                      <span className="text-slate-400">Classified Dvorak Pattern:</span>
                      <span className="font-bold text-slate-200">{uploadResult.pattern_type} (Confidence: {Math.round(uploadResult.confidence * 100)}%)</span>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3 text-xs pt-1.5 border-t border-slate-800/60">
                      <div>
                        <span className="text-slate-400 block">Est. Wind Speed</span>
                        <span className="font-bold text-sky-400 text-sm">{uploadResult.wind_speed_knots} knots</span>
                      </div>
                      <div>
                        <span className="text-slate-400 block">Dvorak T-Number</span>
                        <span className="font-bold text-sky-400 text-sm">T{uploadResult.dvorak_t_number}</span>
                      </div>
                      <div className="col-span-2">
                        <span className="text-slate-400 block">Estimated Central Pressure</span>
                        <span className="font-bold text-sky-400 text-sm">{uploadResult.estimated_pressure_hpa} hPa</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Model baseline metrics info */}
              <div className="p-4 bg-slate-900/40 border border-slate-800/40 rounded-lg space-y-2 text-xs">
                <span className="font-bold text-slate-300 block flex items-center gap-1.5">
                  <Database size={14} className="text-sky-400" /> CNN Training Metrics
                </span>
                <div className="space-y-1 text-slate-400 leading-normal">
                  <div className="flex justify-between"><span>Classification Accuracy:</span><span className="font-bold text-slate-300">92.4%</span></div>
                  <div className="flex justify-between"><span>Wind Intensity Error (RMSE):</span><span className="font-bold text-slate-300">± 7.2 Knots</span></div>
                  <div className="flex justify-between"><span>Central Pressure Error (RMSE):</span><span className="font-bold text-slate-300">± 4.8 hPa</span></div>
                </div>
              </div>
            </div>
          )}

        </section>

      </main>
      
    </div>
  );
}
