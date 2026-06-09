# TGF TRUE MVP ARCHITECTURE
## Minimum Viable Product - Complete Autonomous System
### Version: MVP-1.0 | Focus: Prove Autonomy Works

---

## 🎯 MVP OBJECTIVE

**Demonstrate end-to-end autonomous operation:**
- Sensor readings → AI decision → Chemical dosing
- **Zero human intervention** for 30 continuous days
- **Measurable improvements**: 15-20% chemical reduction vs manual
- **Budget**: $50,000 (vs $180,000 full system)
- **Timeline**: 3 months to pilot deployment

---

## 📊 MVP SCOPE (WHAT'S INCLUDED)

### ✅ CORE FUNCTIONALITY
1. **4 Essential Sensors** (not 18 parameters)
2. **4 Essential Actuators** (3 pumps + 1 valve)
3. **1 Anomaly Detection Model** (MOMENT - already trained)
4. **Hybrid Control** (Rule-based PID + ML dosing predictor)
5. **Simple Cloud** (PostgreSQL + FastAPI + React)
6. **Basic Dashboard** (Real-time monitoring + alerts)
7. **Edge Intelligence** (Raspberry Pi running locally)

### ❌ NOT INCLUDED IN MVP (Save for Later)
- ~~Advanced ensemble models (VTT, RRCF, TransNAS)~~
- ~~Reinforcement learning (SAC agent)~~
- ~~GNN sensor validation~~
- ~~XAI (SHAP/LIME)~~ - Simple explanations only
- ~~Online learning / Transfer learning~~
- ~~Seasonal adaptation~~
- ~~Multi-tower support~~
- ~~Mobile app~~
- ~~Advanced features (turbidity, silica, iron, etc.)~~

---

## 🔧 HARDWARE ARCHITECTURE

### MVP Sensors (4 ONLY - Scientific Study Section 5.1.1)

| Sensor | Purpose | Range | Accuracy | Cost |
|--------|---------|-------|----------|------|
| **pH** | Scale/corrosion control, biocide efficacy | 0-14 | ±0.01 | $800 |
| **Conductivity** | CoC monitoring, blowdown control | 0-10,000 µS/cm | ±2% | $700 |
| **Temperature** | Reaction rates, sensor compensation | 0-50°C | ±0.1°C | $300 |
| **ORP** | Disinfectant efficacy, oxidizing conditions | 0-1000 mV | ±5mV | $900 |

**Total Sensor Cost**: $2,700 (vs $8,000 full system)

**Why these 4?**
- **pH**: Prevents scaling (CaCO₃) and corrosion
- **Conductivity**: Tracks dissolved solids (TDS proxy), manages CoC
- **Temperature**: Affects ALL chemical reactions, critical for calculations
- **ORP**: Monitors biocide effectiveness (oxidizing power)

### MVP Actuators (4 ONLY - Scientific Study Section 5.1.1)

| Actuator | Chemical | Flow Range | Control | Cost |
|----------|----------|------------|---------|------|
| **Pump 1** | Scale & Corrosion Inhibitor | 0.1-10 mL/min | 4-20mA | $1,000 |
| **Pump 2** | Oxidizing Biocide (Chlorine) | 0.1-5 mL/min | 4-20mA | $1,000 |
| **Pump 3** | Non-Oxidizing Biocide | 0.1-5 mL/min | 4-20mA | $1,000 |
| **Valve** | Automated Blowdown | 0-100% open | Modbus | $800 |

**Total Actuator Cost**: $3,800 (vs $5,000 full system)

### MVP Edge Device

**Hardware:**
- Raspberry Pi 4 (4GB) - $75
- 4G USB Modem - $60
- Industrial enclosure (IP54 minimum) - $150
- MicroSD 64GB - $15
- Power supply + UPS battery (2 hours) - $200
- Modbus RTU/RS485 converter - $50
- **Total**: $550 (vs $2,000 full system)

**Why Raspberry Pi?**
- Sufficient for 4 sensors (not 18)
- Can run MOMENT inference (<500ms)
- Low power consumption
- Proven in industrial IoT

---

## 🤖 AI ARCHITECTURE (SIMPLIFIED)

### MVP AI Pipeline
```
Sensor Data (4 params) → Preprocessing → MOMENT Anomaly Detection
                                              ↓
                                    Hybrid Control Layer
                                    (Rules + ML Predictor)
                                              ↓
                                    Dosing Commands → Actuators
```

### Component 1: Data Preprocessing (SIMPLE)

**Input**: 4 parameters every 5 minutes
- pH (0-14 scale)
- Conductivity (µS/cm)
- Temperature (°C)
- ORP (mV)

**Processing:**
1. **Missing Value Handler**: Simple forward fill (no complex scoring)
2. **RobustScaler**: Fitted on 4 parameters only (not 18)
3. **Sliding Window**: 24 hours (288 points) - NOT 168 hours
   - Reason: Faster inference, captures daily cycles
4. **Physics Features**: Calculate ONLY essential ones
   - LSI (Langelier Saturation Index) - requires pH, temp, conductivity
   - CoC (Cycles of Concentration) - conductivity ratio

**Output**: [batch, 288, 4] tensor

### Component 2: MOMENT Anomaly Detection (USE EXISTING)

**Model**: MOMENT foundation model (already trained!)
- **Current training**: 5,461 samples with 18 parameters
- **MVP adaptation**: Fine-tune last layers for 4 parameters
- **Retraining time**: 1-2 hours (not 4-6 hours)

**Configuration:**
```python
# MVP MOMENT Config
window_size = 288  # 24 hours at 5-min intervals
input_features = 4  # pH, Cond, Temp, ORP
model_params = 1.2M  # Smaller than 3.2M (4-param version)
threshold = 3.0  # 3-sigma conservative
```

**Output**: Anomaly score per window (0-1)
- Score > 0.7 → Anomaly detected → Alert + Log
- Score 0.4-0.7 → Warning → Increase monitoring frequency
- Score < 0.4 → Normal → Continue normal operation

**Latency**: <500ms on Raspberry Pi (ONNX optimized)

### Component 3: Hybrid Control Layer (CORE MVP INNOVATION)

This is where MVP differs from full system. Instead of RL agent (SAC), use **proven rule-based control + ML assistance**.

#### 3A. Rule-Based PID Controllers (PROVEN, RELIABLE)

**Controller 1: pH Control**
```python
target_pH = 7.8  # Optimal for most cooling towers
tolerance = 0.2  # ±0.2 pH units

if current_pH < (target_pH - tolerance):
    # pH too low → Scale risk
    action = increase_alkalinity_or_reduce_acid_dose
elif current_pH > (target_pH + tolerance):
    # pH too high → Corrosion risk
    action = add_acid_or_reduce_alkalinity
else:
    # pH in range
    action = maintain_current_dose
```

**Controller 2: CoC Control (via Blowdown)**
```python
target_CoC = 4.0  # Target cycles of concentration
current_CoC = conductivity_circulating / conductivity_makeup

if current_CoC > 4.5:
    # Too concentrated → Scaling risk
    blowdown_valve_open = 40%  # Increase blowdown
elif current_CoC < 3.5:
    # Too dilute → Wasting water
    blowdown_valve_open = 10%  # Reduce blowdown
else:
    # CoC optimal
    blowdown_valve_open = 20%  # Maintain
```

**Controller 3: ORP Control (Biocide Dosing)**
```python
target_ORP = 650  # mV, optimal for chlorine efficacy
tolerance = 50  # ±50 mV

if current_ORP < 600:
    # Low oxidizing power → Microbial risk
    oxidizing_biocide_dose = increase_by_20%
elif current_ORP > 700:
    # High oxidizing power → Chemical waste + corrosion
    oxidizing_biocide_dose = decrease_by_20%
else:
    # ORP optimal
    oxidizing_biocide_dose = maintain
```

#### 3B. ML Dosing Predictor (SIMPLE MODEL)

**Purpose**: Predict optimal chemical doses based on patterns

**Model**: Gradient Boosting Regressor (LightGBM)
- **Why?** Fast, accurate, interpretable, works with small data
- **Training data**: 5,461 historical samples
- **Features**: pH, Conductivity, Temperature, ORP, Hour, Day
- **Targets**: 3 dosing rates (scale inhibitor, ox biocide, non-ox biocide)

**Training:**
```python
from lightgbm import LGBMRegressor

# Separate models for each chemical
model_scale_inhibitor = LGBMRegressor(n_estimators=100, max_depth=5)
model_ox_biocide = LGBMRegressor(n_estimators=100, max_depth=5)
model_nonox_biocide = LGBMRegressor(n_estimators=100, max_depth=5)

# Train on historical data
# X = [pH, Conductivity, Temperature, ORP, Hour, Day, CoC, LSI]
# y = actual dosing rates (when system was controlled well)

model_scale_inhibitor.fit(X_train, y_scale_inhibitor)
# ... etc
```

**Inference**: <10ms on Raspberry Pi

**Decision Logic**: Blend rules + ML predictions
```python
# Rule-based gives baseline
rule_based_dose = pid_controller_output()

# ML predicts optimal
ml_predicted_dose = ml_model.predict(current_state)

# Blend (trust rules more initially, ML more over time)
confidence = min(days_running / 30, 0.5)  # Max 50% ML weight
final_dose = (1 - confidence) * rule_based_dose + confidence * ml_predicted_dose

# Safety check
final_dose = clip(final_dose, min_safe, max_safe)
```

### Component 4: Self-Correction (SIMPLE FEEDBACK LOOP)

**Mechanism**: Closed-loop control with setpoint tracking

```python
every_5_minutes:
    # 1. Read sensors
    current_state = read_sensors()  # pH, Cond, Temp, ORP
    
    # 2. Check for anomalies
    anomaly_score = moment_model.predict(current_state)
    if anomaly_score > 0.7:
        alert("Anomaly detected!", current_state)
    
    # 3. Calculate physics features
    LSI = calculate_lsi(pH, temp, conductivity)
    CoC = conductivity / makeup_conductivity
    
    # 4. Hybrid control decision
    doses = hybrid_controller(current_state, LSI, CoC)
    
    # 5. Execute actions
    pump1.set_flow(doses['scale_inhibitor'])
    pump2.set_flow(doses['ox_biocide'])
    pump3.set_flow(doses['nonox_biocide'])
    valve.set_position(doses['blowdown_percent'])
    
    # 6. Log everything
    log_to_database(current_state, doses, anomaly_score)
```

**Self-Correction Example:**
1. ORP drops from 650mV to 580mV (below target)
2. MOMENT detects anomaly (score 0.8)
3. PID controller: Increase oxidizing biocide by 30%
4. ML predictor: Suggests 25% increase
5. Blended decision: Increase by 27%
6. Execute dose adjustment
7. Wait 15 minutes, measure ORP again
8. ORP rises to 640mV → Return to normal dosing
9. **Self-corrected without human intervention**

---

## ☁️ CLOUD ARCHITECTURE (MINIMAL)

### MVP Cloud Stack

**Compute**: Single AWS EC2 instance (t3.medium)
- 2 vCPU, 4GB RAM
- Cost: ~$30/month
- Hosts everything (database, API, dashboard)

**Database**: PostgreSQL 14 (on same EC2)
- Time-series extension: TimescaleDB
- Retention: 90 days (vs 5 years full system)
- Tables:
  - `sensor_readings` (pH, Cond, Temp, ORP, timestamp)
  - `anomaly_scores` (timestamp, score, detected)
  - `dosing_actions` (timestamp, chemical, rate)
  - `alerts` (timestamp, severity, message)

**Backend API**: FastAPI (Python)
- Endpoints:
  - `GET /api/readings/latest` - Last 100 readings
  - `GET /api/readings/history` - Time range query
  - `GET /api/anomalies` - Recent anomalies
  - `GET /api/dosing` - Dosing log
  - `POST /api/setpoints` - Update targets (pH, ORP, CoC)
  - `GET /api/health` - System status
- Authentication: Simple API key (not full OAuth)

**Frontend**: React dashboard (single page)
- Components:
  1. **Live Status Card**: Current pH, Cond, Temp, ORP (big numbers)
  2. **24-Hour Charts**: Line charts for each parameter
  3. **Anomaly Timeline**: Red markers when detected
  4. **Dosing Log Table**: Recent chemical additions
  5. **Alerts Panel**: Critical warnings
- Update frequency: 30 seconds (WebSocket)

**Communication**: MQTT
- Broker: Mosquitto (on same EC2)
- Topics:
  - `tgf/sensors` - Sensor data from edge
  - `tgf/commands` - Control commands to edge
  - `tgf/alerts` - System alerts
- QoS 1 (at least once delivery)
- TLS encryption

**No Need For:**
- ❌ Kafka/Pulsar (overkill for 1 tower)
- ❌ Separate model serving (FastAPI handles it)
- ❌ MLflow (no continuous retraining yet)
- ❌ S3 data lake (PostgreSQL sufficient)
- ❌ Auto-scaling (1 tower = low load)

---

## 📱 MVP DASHBOARD (SIMPLE UI)

### Dashboard Features (Mobile-Responsive)

**Page 1: Live Monitoring**
```
┌─────────────────────────────────────────────┐
│  TGF Autonomous Cooling Tower Control       │
│  Status: ● RUNNING  |  Uptime: 23d 14h     │
├─────────────────────────────────────────────┤
│                                             │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │
│  │ pH   │  │ Cond │  │ Temp │  │ ORP  │  │
│  │ 7.82 │  │ 2450 │  │ 32°C │  │ 652  │  │
│  │  ✓   │  │  ✓   │  │  ✓   │  │  ✓   │  │
│  └──────┘  └──────┘  └──────┘  └──────┘  │
│  Target:    Target:   Ambient   Target:   │
│  7.8±0.2    CoC 4.0             650±50    │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │   pH Trend (24 hours)               │  │
│  │   [Line chart showing pH over time] │  │
│  └─────────────────────────────────────┘  │
│                                             │
│  ┌─────────────────────────────────────┐  │
│  │ Recent Actions                      │  │
│  │ 14:23 - Increased oxidizing biocide │  │
│  │ 14:18 - Anomaly detected (ORP drop) │  │
│  │ 13:45 - Adjusted blowdown (CoC 4.2) │  │
│  └─────────────────────────────────────┘  │
│                                             │
│  [ View History ]  [ System Settings ]     │
└─────────────────────────────────────────────┘
```

**Page 2: Dosing Log**
- Table: Timestamp | Chemical | Amount | Reason
- Export CSV for analysis

**Page 3: Alerts**
- List of anomalies and warnings
- Filter by severity
- Acknowledge button

**Page 4: Settings** (Admin only)
- Adjust setpoints (pH, ORP, CoC targets)
- Calibrate sensors (manual entry)
- Enable/disable autonomous mode
- Emergency stop button

---

## 🔒 SECURITY (BASIC)

### MVP Security Measures

1. **Device Authentication**: Simple API key
2. **Data Encryption**: TLS 1.2 in transit
3. **Dashboard Login**: Username + Password (bcrypt hashed)
4. **Firewall**: Only ports 443 (HTTPS) and 8883 (MQTTS) open
5. **Backup**: Daily database dump to S3 (encrypted)

**NOT in MVP:**
- ❌ X.509 certificates
- ❌ OAuth2/OIDC
- ❌ Role-based access control (RBAC)
- ❌ Audit logging
- ❌ Intrusion detection

---

## 📈 MVP SUCCESS CRITERIA

### Technical Metrics (30-day pilot)

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Uptime** | >95% | System operational time |
| **Anomaly Detection** | F1 >0.80 | True positives / (TP + FP/2 + FN/2) |
| **False Positive Rate** | <10% | Acceptable for MVP |
| **Parameter Stability** | pH ±0.3 | Standard deviation over 30 days |
| **Response Time** | <30 min | Time to correct anomaly |
| **Human Intervention** | 0 times | No manual adjustments needed |

### Operational Metrics (vs Manual Baseline)

| Metric | Target | Baseline | Expected Improvement |
|--------|--------|----------|---------------------|
| **Chemical Usage** | -15% | 10 kg/day | 8.5 kg/day |
| **Water Usage** | -10% | 1000 L/day blowdown | 900 L/day |
| **pH Stability** | ±0.3 | ±0.8 | 2.7x better |
| **ORP Stability** | ±50mV | ±150mV | 3x better |
| **Anomaly Response** | <30min | >6 hours | 12x faster |

### Business Metrics

| Metric | Target |
|--------|--------|
| **Cost Savings** | $500/month per tower (chemical + water) |
| **ROI** | 12-month payback |
| **Deployment Time** | <2 days per tower |
| **Training Time** | <4 hours for operator |

---

## 🚀 MVP IMPLEMENTATION TIMELINE

### 3-Month Sprint to Pilot

**Month 1: Hardware & Basic Software**

**Week 1-2: Hardware Assembly**
- [ ] Order all sensors and actuators ($6,500)
- [ ] Order Raspberry Pi and accessories ($550)
- [ ] Build edge device enclosure
- [ ] Wire sensors to Raspberry Pi (Modbus/4-20mA)
- [ ] Test sensor readings (Python scripts)
- [ ] Test actuator control (pump flow, valve position)
- **Deliverable**: Working hardware rig in lab

**Week 3-4: Edge Software**
- [ ] Install Ubuntu Server 22.04 on Raspberry Pi
- [ ] Sensor polling script (every 5 minutes)
- [ ] Local SQLite buffer (24-hour retention)
- [ ] MQTT client (publish to cloud)
- [ ] Actuator control module (receive commands)
- [ ] Watchdog timer (auto-restart on crash)
- **Deliverable**: Edge device publishing sensor data

**Month 2: AI & Cloud**

**Week 5-6: MOMENT Adaptation**
- [ ] Load existing MOMENT model (5,461 samples)
- [ ] Fine-tune for 4 parameters (not 18)
  - Freeze early layers, retrain last 2 layers
  - Training time: 1-2 hours on GPU
- [ ] Export to ONNX (optimize for Raspberry Pi)
- [ ] Test inference latency (<500ms)
- [ ] Validate on test set: F1 >0.80
- **Deliverable**: MOMENT model running on edge

**Week 7-8: Hybrid Controller**
- [ ] Implement PID controllers (pH, ORP, CoC)
- [ ] Train LightGBM dosing predictor
  - Features: 4 sensors + time + physics
  - Targets: 3 dosing rates
  - Training time: 10 minutes
- [ ] Implement blending logic (rules + ML)
- [ ] Safety checks (min/max doses)
- [ ] Test in simulation (synthetic data)
- **Deliverable**: Autonomous control algorithm

**Week 9-10: Cloud Infrastructure**
- [ ] Launch AWS EC2 t3.medium
- [ ] Install PostgreSQL + TimescaleDB
- [ ] Setup Mosquitto MQTT broker
- [ ] Deploy FastAPI backend
- [ ] Create database schema
- [ ] Test data ingestion pipeline
- **Deliverable**: Cloud receiving sensor data

**Week 11-12: Dashboard**
- [ ] React dashboard skeleton
- [ ] Live status cards (4 sensors)
- [ ] 24-hour trend charts (Chart.js)
- [ ] Dosing log table
- [ ] Alert notifications
- [ ] Settings page
- [ ] Mobile responsive
- **Deliverable**: Functional web dashboard

**Month 3: Integration & Pilot**

**Week 13-14: Integration Testing**
- [ ] End-to-end test in lab
  - Sensor → Edge → Cloud → Dashboard
  - Anomaly injection tests
  - Control loop tests (pH adjustment)
- [ ] Load testing (simulate 1 week of data)
- [ ] Network failure tests (offline mode)
- [ ] Power failure recovery
- [ ] Fine-tune control parameters
- **Deliverable**: System working in lab for 7 days

**Week 15: Pilot Site Preparation**
- [ ] Select pilot site (friendly partner)
- [ ] Site survey (power, network, plumbing)
- [ ] Baseline data collection (1 week manual operation)
  - Measure: pH, conductivity, temp, ORP
  - Measure: Chemical usage, water usage
  - Measure: Operator time spent
- [ ] Install hardware at site
- **Deliverable**: Hardware installed, baseline established

**Week 16: Pilot Deployment**
- [ ] Commission system (sensor calibration)
- [ ] Run in "shadow mode" for 3 days
  - AI suggests actions, human executes
  - Validate AI decisions
- [ ] Switch to full autonomous mode
- [ ] Daily check-ins (remote monitoring)
- [ ] Weekly on-site inspection
- **Deliverable**: 30-day autonomous operation begins

---

## 💰 MVP COST BREAKDOWN

### One-Time Development (Total: $50,000)

| Item | Cost |
|------|------|
| **R&D Personnel** (3 months, 2 engineers) | $40,000 |
| **Hardware Prototype** (sensors, actuators, edge) | $7,000 |
| **Cloud Setup** (3 months AWS + domain) | $500 |
| **Software Tools** (licenses, APIs) | $1,000 |
| **Testing & Validation** | $1,500 |

### Per-Tower Deployment (Total: $12,000)

| Item | Cost |
|------|------|
| **Sensors** (4 types) | $2,700 |
| **Actuators** (3 pumps + valve) | $3,800 |
| **Edge Device** (Raspberry Pi + accessories) | $550 |
| **Installation** (plumbing, wiring, commissioning) | $3,500 |
| **Training** (4 hours for operator) | $500 |
| **Contingency** (10%) | $950 |

### Monthly Operating Cost (Per Tower)

| Item | Cost |
|------|------|
| **Cloud Hosting** (EC2 + database + bandwidth) | $50 |
| **Cellular Data** (2GB/month) | $20 |
| **Support** (remote monitoring) | $100 |
| **Total** | **$170/month** |

### ROI Calculation (Per Tower, Annual)

| Item | Amount |
|------|--------|
| **Chemical Savings** (15% reduction) | $4,500 |
| **Water Savings** (10% reduction) | $2,000 |
| **Labor Savings** (2 hours/week operator time) | $5,000 |
| **Total Savings** | **$11,500/year** |
| **Operating Cost** | -$2,040/year |
| **Net Benefit** | **$9,460/year** |
| **Payback Period** | **15 months** |

---

## 🎓 WHAT WE LEARN FROM MVP

### Technical Validation
1. **Does MOMENT work with 4 parameters?** (vs 18)
2. **Is hybrid control (rules + ML) good enough?** (vs pure RL)
3. **Can Raspberry Pi handle real-time inference?** (<500ms)
4. **Is 5-minute sampling sufficient?** (vs 1-minute)

### Operational Validation
5. **Can system run 30 days without human intervention?**
6. **Are chemical/water savings 15%+?**
7. **Is dashboard intuitive for operators?**
8. **What are the failure modes?** (sensor drift, network issues)

### Business Validation
9. **Is 15-month payback acceptable to customers?**
10. **What's the biggest pain point?** (installation, training, trust)
11. **Which feature is most valued?** (anomaly alerts, cost savings, stability)
12. **What prevents scaling?** (regulatory, technical, financial)

### Path to Full Product
- If MVP succeeds → Add advanced models (VTT, RRCF)
- If MVP struggles → Fix fundamentals before adding complexity
- If MVP exceeds expectations → Fast-track to market

---

## 🚨 MVP RISKS & MITIGATION

### Critical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Sensor failure** | High | Critical | Manual backup procedure, daily checks |
| **Network downtime** | Medium | High | 24-hour local buffer, cellular + WiFi |
| **Model accuracy low** | Medium | High | Fall back to pure rule-based control |
| **Chemical compatibility** | Low | High | Vendor certification before deployment |
| **Customer trust** | Medium | Medium | Shadow mode first, transparent dashboard |

### Acceptance Criteria for "Go/No-Go" Decision

**GO to Full Product if:**
- ✅ 30 days autonomous operation (0 human interventions)
- ✅ Chemical savings ≥10%
- ✅ Anomaly detection F1 ≥0.75
- ✅ Uptime ≥90%
- ✅ Customer satisfaction ≥4/5

**NO-GO (iterate MVP) if:**
- ❌ Frequent human interventions needed (>5 in 30 days)
- ❌ Savings <5%
- ❌ F1 <0.70 (too many false alarms)
- ❌ Uptime <80%
- ❌ Customer dissatisfied

---

## 📋 MVP IMPLEMENTATION CHECKLIST

### Phase 1: Hardware (Weeks 1-2)
- [ ] Sensors procured and tested
- [ ] Actuators procured and tested
- [ ] Raspberry Pi configured
- [ ] Wiring completed
- [ ] Lab rig operational

### Phase 2: Edge Software (Weeks 3-4)
- [ ] Sensor polling working
- [ ] Local database operational
- [ ] MQTT publishing working
- [ ] Actuator control tested
- [ ] Watchdog timer configured

### Phase 3: AI Models (Weeks 5-8)
- [ ] MOMENT fine-tuned for 4 params
- [ ] ONNX export successful
- [ ] Inference <500ms on Raspberry Pi
- [ ] LightGBM dosing predictor trained
- [ ] Hybrid controller implemented
- [ ] Safety checks in place

### Phase 4: Cloud (Weeks 9-12)
- [ ] EC2 instance launched
- [ ] PostgreSQL + TimescaleDB setup
- [ ] MQTT broker operational
- [ ] FastAPI deployed
- [ ] Dashboard deployed
- [ ] End-to-end testing complete

### Phase 5: Pilot (Weeks 13-16)
- [ ] Lab testing 7 days successful
- [ ] Pilot site selected
- [ ] Baseline data collected
- [ ] Hardware installed at site
- [ ] Shadow mode validation (3 days)
- [ ] Full autonomous mode activated
- [ ] 30-day pilot begins

---

## 🎯 MVP SUCCESS = PROOF OF CONCEPT

**MVP Goal**: Prove that autonomous cooling tower control works in reality, not just theory.

**Success Looks Like:**
- Engineer visits pilot site after 30 days
- Checks dashboard: "All parameters stable"
- Talks to operator: "I haven't touched anything in a month"
- Checks chemical inventory: "We've used 15% less"
- Customer says: "This is amazing, when can we deploy 10 more towers?"

**Then and only then do we add:**
- Advanced models (VTT, RRCF, TransNAS)
- Reinforcement learning (SAC)
- XAI (SHAP/LIME)
- Multi-tower support
- Transfer learning
- All the fancy features

**MVP Philosophy**: 
> "First, make it work. Then, make it better."

---

**Document Status**: TRUE MVP DEFINITION
**Next Step**: Get approval, order hardware, start Week 1
**Decision Point**: After 30-day pilot (Go/No-Go to full product)
**Timeline**: 3 months to pilot, 4 months to production if successful
**Budget**: $50K development + $12K per tower + $170/month operating
